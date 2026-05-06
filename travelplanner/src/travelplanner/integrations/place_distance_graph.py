"""Build a **place distance graph**: clusters, hub matrix legs, and directed edges between stops.

Human-oriented overview: ``docs/routing.md``. This module implements :func:`build_place_distance_graph`:
walk-linked **clusters**, one **medoid hub** per cluster, a **full hub×hub** Route Matrix (**BICYCLE**,
**TRANSIT** when enabled, **DRIVE** for every ordered hub pair), pairwise **edges**, and **GeoJSON**.

**Within a cluster** only walk approximations are used (short direct walk or walk-via-hub for long
same-pocket paths). **Between clusters** each ``edge`` is a **hub chain** (walk to hub → one primary
matrix mode → walk from hub); all modes are still listed in ``hub_hub_legs`` for lookups.
Matrix responses may be **disk-cached** (:mod:`travelplanner.integrations.routing_api_cache`).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Literal, cast

import numpy as np

from travelplanner.integrations.google_route_matrix import (
    RouteMatrixElement,
    compute_all_travel_modes_hub_matrices,
    haversine_meters,
    hub_pair_mode_matrix,
    matrix_travel_mode_for_hub_separation_m,
)
from travelplanner.integrations.google_routes import (
    TravelMode,
    geocode_address_to_lat_lng,
)
from travelplanner.integrations.routing_api_cache import (
    ROUTING_CACHE_INTEGRATION_VERSION,
    reset_routing_cache_metrics,
    routing_cache_metrics,
)
from travelplanner.schema.place_distance_graph import (
    ClusterContext,
    ClusterSummaryModel,
    EdgeEstimateModel,
    EdgeLegModel,
    EdgeQuality,
    GraphBuildStatsModel,
    HubHubLegModel,
    MapPlaceInputModel,
    MatrixPolicyBand,
    PlaceDistanceGraphBuildConfig,
    PlaceDistanceGraphModel,
    PlaceNodeModel,
    TravelModeLiteral,
)

_CANONICAL_CLUSTER_CONTEXTS: frozenset[str] = frozenset(
    {"dense_urban", "mixed", "sparse"}
)


def normalize_cluster_context(value: object) -> ClusterContext:
    """Return a valid walk-cluster preset; unknown or empty values become ``mixed``."""
    if isinstance(value, str) and value.strip():
        s = value.strip().lower()
        if s in _CANONICAL_CLUSTER_CONTEXTS:
            return cast(ClusterContext, s)
    return "mixed"


def place_distance_graph_config_for_context(
    context: ClusterContext | str | None,
) -> PlaceDistanceGraphBuildConfig:
    """Build graph config from ``cluster_context`` preset."""
    ctx = (context or "mixed").strip().lower()
    if ctx not in ("dense_urban", "mixed", "sparse"):
        msg = f"cluster_context must be dense_urban, mixed, or sparse, got {context!r}"
        raise ValueError(msg)

    if ctx == "dense_urban":
        return PlaceDistanceGraphBuildConfig(
            cluster_link_m=420.0,
            cluster_link_adaptive=True,
            cluster_link_floor_m=180.0,
            cluster_link_nn_multiplier=2.0,
            use_transit_for_hub_pairs=True,
            hub_pair_bicycle_max_m=3000.0,
        )
    if ctx == "sparse":
        return PlaceDistanceGraphBuildConfig(
            cluster_link_m=900.0,
            cluster_link_adaptive=True,
            cluster_link_floor_m=350.0,
            cluster_link_nn_multiplier=2.15,
            use_transit_for_hub_pairs=True,
            hub_pair_bicycle_max_m=3000.0,
        )
    return PlaceDistanceGraphBuildConfig(
        cluster_link_adaptive=True,
        use_transit_for_hub_pairs=True,
        hub_pair_bicycle_max_m=3000.0,
    )


def parse_places_input_payload(
    raw: object,
) -> tuple[str, list[MapPlaceInputModel], dict[str, str]]:
    """Parse places JSON: ``{stops: [{address}]}`` or ``{cluster_context, stops}`` or legacy array.

    Returns ``(cluster_context, places, id_to_address)`` where ``id_to_address`` maps place ids
    to their original address strings for transit lookups.
    """
    cluster_context = "mixed"
    rows: list[object]
    if isinstance(raw, dict) and "stops" in raw:
        cluster_context = normalize_cluster_context(raw.get("cluster_context"))
        stops = raw["stops"]
        if not isinstance(stops, list):
            msg = "stops must be a JSON array"
            raise ValueError(msg)
        rows = stops
    elif isinstance(raw, list):
        rows = raw
    else:
        msg = "places JSON must be an array of stops or an object with a stops array"
        raise ValueError(msg)

    places: list[MapPlaceInputModel] = []
    id_to_address: dict[str, str] = {}
    for i, row in enumerate(rows):
        if isinstance(row, str):
            row = {"address": row.strip()}
        if not isinstance(row, dict):
            msg = f"stops[{i}] must be an object or address string"
            raise ValueError(msg)
        rid = str(row.get("id") or "").strip() or f"stop_{i}"
        addr = str(row.get("address") or "").strip()
        name = str(row.get("name") or row.get("label") or "").strip()
        category = str(row.get("category") or "other").strip() or "other"
        if addr:
            id_to_address[rid] = addr
        lat = row.get("latitude") if row.get("latitude") is not None else row.get("lat")
        lon = (
            row.get("longitude") if row.get("longitude") is not None else row.get("lng")
        )
        places.append(
            MapPlaceInputModel.model_validate(
                {
                    "id": rid,
                    "name": name,
                    "category": category,
                    "address": addr if addr else None,
                    "latitude": lat,
                    "longitude": lon,
                }
            )
        )
    return cluster_context, places, id_to_address


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._p = list(range(n))

    def find(self, i: int) -> int:
        while self._p[i] != i:
            self._p[i] = self._p[self._p[i]]
            i = self._p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[rb] = ra


def _haversine_matrix_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    r = 6371000.0
    latr = np.radians(lat)
    lonr = np.radians(lon)
    dlat = latr[:, None] - latr[None, :]
    dlon = lonr[:, None] - lonr[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(latr)[:, None] * np.cos(latr)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * r * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _medoid_global_index(member_indices: list[int], dist_m: np.ndarray) -> int:
    best = member_indices[0]
    best_sum = math.inf
    for j in member_indices:
        s = float(dist_m[j, member_indices].sum())
        if s < best_sum:
            best_sum = s
            best = j
    return best


def _matrix_element_ok(el: RouteMatrixElement) -> bool:
    if el.duration_seconds is None or el.duration_seconds <= 0:
        return False
    if el.condition is not None and el.condition != "ROUTE_EXISTS":
        return False
    return True


def _walk_approx_duration_s(
    dist_m: float,
    *,
    detour: float,
    speed_m_s: float,
) -> tuple[float, float]:
    dm = max(0.0, dist_m) * detour
    return dm, dm / speed_m_s if speed_m_s > 0 else 0.0


def cast_mode(mode: TravelMode) -> TravelModeLiteral:
    if mode == "DRIVE":
        return "DRIVE"
    if mode == "WALK":
        return "WALK"
    if mode == "BICYCLE":
        return "BICYCLE"
    return "TRANSIT"


def _matrix_band_for_travel_mode(mode: TravelMode) -> MatrixPolicyBand:
    """Tag for a matrix row: which ``travelMode`` was queried (bicycle / transit / drive)."""
    if mode == "BICYCLE":
        return "bicycle"
    if mode == "TRANSIT":
        return "transit"
    return "drive"


def _pick_primary_hub_matrix_cell(
    triples: dict[tuple[int, int, TravelMode], RouteMatrixElement],
    *,
    hi: int,
    hj: int,
    h_h: float,
    cfg: PlaceDistanceGraphBuildConfig,
    use_transit_eff: bool,
) -> tuple[RouteMatrixElement, TravelMode] | None:
    """Hub-chain middle leg: prefer Haversine-band mode, then DRIVE / BICYCLE / TRANSIT fallback."""
    if not triples:
        return None
    preferred = matrix_travel_mode_for_hub_separation_m(
        h_h,
        bicycle_max_m=cfg.hub_pair_bicycle_max_m,
        transit_max_m=cfg.hub_pair_transit_max_m,
        use_transit=use_transit_eff,
    )
    for mode in (
        preferred,
        cast(TravelMode, "DRIVE"),
        cast(TravelMode, "BICYCLE"),
        cast(TravelMode, "TRANSIT"),
    ):
        if mode == "TRANSIT" and not use_transit_eff:
            continue
        el = triples.get((hi, hj, mode))
        if el is not None and _matrix_element_ok(el):
            return el, mode
    return None


def _resolve_map_places_with_geocoding(
    places: list[MapPlaceInputModel], api_key: str
) -> list[MapPlaceInputModel]:
    """Fill missing coordinates from ``address`` via Google Geocoding API."""
    out: list[MapPlaceInputModel] = []
    for p in places:
        if p.latitude is not None and p.longitude is not None:
            out.append(p)
            continue
        addr = (p.address or "").strip()
        if not addr:
            msg = f"place {p.id!r} has no coordinates and no address"
            raise ValueError(msg)
        lat, lng = geocode_address_to_lat_lng(addr, api_key=api_key)
        out.append(
            MapPlaceInputModel(
                id=p.id,
                name=p.name,
                category=p.category,
                address=p.address,
                latitude=lat,
                longitude=lng,
            )
        )
    return out


def _geojson_for_places(
    place_nodes: list[PlaceNodeModel],
    hub_legs: list[HubHubLegModel],
    id_to_latlon: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for p in place_nodes:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [p.longitude, p.latitude]},
                "properties": {
                    "id": p.id,
                    "name": p.name,
                    "category": p.category,
                    "cluster_id": p.cluster_id,
                    "cluster_hub_id": p.cluster_hub_id,
                },
            }
        )
    for leg in hub_legs:
        a = id_to_latlon.get(leg.from_hub_id)
        b = id_to_latlon.get(leg.to_hub_id)
        if not a or not b:
            continue
        if leg.distance_meters is None:
            continue
        # id_to_latlon values are (latitude, longitude); GeoJSON is [lon, lat]
        alat, alon = a[0], a[1]
        blat, blon = b[0], b[1]
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[alon, alat], [blon, blat]],
                },
                "properties": {
                    "kind": "hub_leg",
                    "from_hub_id": leg.from_hub_id,
                    "to_hub_id": leg.to_hub_id,
                    "travel_mode": leg.travel_mode,
                    "matrix_policy_band": leg.matrix_policy_band,
                    "distance_meters": leg.distance_meters,
                    "duration_seconds": leg.duration_seconds,
                    "haversine_meters": leg.haversine_meters,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def build_place_distance_graph(
    places: list[MapPlaceInputModel],
    api_key: str,
    *,
    config: PlaceDistanceGraphBuildConfig | None = None,
) -> PlaceDistanceGraphModel:
    """Build clusters, hub matrix legs, and pairwise edges for plotting / later analysis."""
    cfg = config or PlaceDistanceGraphBuildConfig()
    if len(places) < 2:
        msg = "need at least two places"
        raise ValueError(msg)
    if len(places) > cfg.max_places_all_pairs:
        msg = (
            f"{len(places)} places exceeds max_places_all_pairs={cfg.max_places_all_pairs}; "
            "raise the cap or pre-filter POIs / add a sparse mode."
        )
        raise ValueError(msg)

    ids = [p.id for p in places]
    if len(set(ids)) != len(ids):
        msg = "place ids must be unique"
        raise ValueError(msg)

    places = _resolve_map_places_with_geocoding(places, api_key)

    reset_routing_cache_metrics()

    n = len(places)
    lat = np.array([p.latitude for p in places], dtype=np.float64)
    lon = np.array([p.longitude for p in places], dtype=np.float64)
    dist_m = _haversine_matrix_m(lat, lon)

    link_m = float(cfg.cluster_link_m)
    if cfg.cluster_link_adaptive and n >= 2:
        d2 = dist_m.copy()
        np.fill_diagonal(d2, np.inf)
        nn = np.min(d2, axis=1)
        med_nn = float(np.median(nn))
        link_m = min(
            link_m,
            max(cfg.cluster_link_floor_m, med_nn * cfg.cluster_link_nn_multiplier),
        )

    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if float(dist_m[i, j]) <= link_m:
                uf.union(i, j)

    root_to_members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        root_to_members[uf.find(i)].append(i)

    roots_sorted = sorted(
        root_to_members.keys(),
        key=lambda r: (
            float(lat[root_to_members[r][0]]),
            float(lon[root_to_members[r][0]]),
        ),
    )
    cluster_id_by_root: dict[int, int] = {r: k for k, r in enumerate(roots_sorted)}
    cluster_id_of_vertex = [cluster_id_by_root[uf.find(i)] for i in range(n)]

    medoid_global: dict[int, int] = {}
    hub_place_id_by_cluster: dict[int, str] = {}
    for root in roots_sorted:
        cid = cluster_id_by_root[root]
        members = root_to_members[root]
        med = _medoid_global_index(members, dist_m)
        medoid_global[cid] = med
        hub_place_id_by_cluster[cid] = places[med].id

    hub_coords: list[tuple[float, float]] = []
    hub_id_order: list[str] = []
    for cid in range(len(roots_sorted)):
        g = medoid_global[cid]
        hub_coords.append((float(lat[g]), float(lon[g])))
        hub_id_order.append(places[g].id)

    triple_matrix: dict[tuple[int, int, TravelMode], RouteMatrixElement] = {}
    matrix_http = 0
    matrix_elems = 0
    matrix_tiles_skipped = 0
    use_transit_eff = bool(cfg.use_transit_for_hub_pairs)
    transit_policy_note = "transit_user_off" if not use_transit_eff else "transit_on"
    if len(hub_coords) > 1:
        if use_transit_eff:
            em_probe = hub_pair_mode_matrix(
                hub_coords,
                bicycle_max_m=cfg.hub_pair_bicycle_max_m,
                transit_max_m=cfg.hub_pair_transit_max_m,
                use_transit=True,
            )
            n_transit_cells = int(np.count_nonzero(em_probe == 2))
            if len(hub_coords) > cfg.disable_transit_if_hub_clusters_exceed:
                use_transit_eff = False
                transit_policy_note = f"transit_off_hubs({len(hub_coords)}>{cfg.disable_transit_if_hub_clusters_exceed})"
            elif n_transit_cells > cfg.max_transit_hub_pair_cells:
                use_transit_eff = False
                transit_policy_note = f"transit_off_budget(transit_pairs={n_transit_cells}>{cfg.max_transit_hub_pair_cells})"
        triple_matrix, matrix_http, matrix_elems = (
            compute_all_travel_modes_hub_matrices(
                hubs=hub_coords,
                api_key=api_key,
                use_transit=use_transit_eff,
                departure_time_rfc3339=cfg.departure_time_rfc3339,
                sleep_between_requests_s=cfg.sleep_between_matrix_requests_s,
            )
        )

    _mode_order = {"BICYCLE": 0, "TRANSIT": 1, "DRIVE": 2}
    hub_hub_legs: list[HubHubLegModel] = []
    for (i, j, mode), el in sorted(
        triple_matrix.items(),
        key=lambda it: (it[0][0], it[0][1], _mode_order.get(it[0][2], 9)),
    ):
        hlat1, hlon1 = hub_coords[i]
        hlat2, hlon2 = hub_coords[j]
        h_h = haversine_meters(hlat1, hlon1, hlat2, hlon2)
        hub_hub_legs.append(
            HubHubLegModel(
                from_hub_id=hub_id_order[i],
                to_hub_id=hub_id_order[j],
                travel_mode=cast_mode(mode),
                matrix_policy_band=_matrix_band_for_travel_mode(mode),
                distance_meters=el.distance_meters,
                duration_seconds=el.duration_seconds,
                haversine_meters=h_h,
                condition=el.condition,
            )
        )

    clusters_out: list[ClusterSummaryModel] = []
    for cid in range(len(roots_sorted)):
        root = roots_sorted[cid]
        members = root_to_members[root]
        mids = members
        centroid_lat = float(lat[mids].mean())
        centroid_lon = float(lon[mids].mean())
        clusters_out.append(
            ClusterSummaryModel(
                cluster_id=cid,
                member_place_ids=[places[g].id for g in mids],
                hub_place_id=hub_place_id_by_cluster[cid],
                centroid_latitude=centroid_lat,
                centroid_longitude=centroid_lon,
            )
        )

    place_nodes: list[PlaceNodeModel] = []
    for i in range(n):
        cid = cluster_id_of_vertex[i]
        place_nodes.append(
            PlaceNodeModel(
                id=places[i].id,
                name=places[i].name,
                category=places[i].category,
                latitude=float(lat[i]),
                longitude=float(lon[i]),
                cluster_id=cid,
                cluster_hub_id=hub_place_id_by_cluster[cid],
            )
        )

    id_to_latlon = {places[i].id: (float(lat[i]), float(lon[i])) for i in range(n)}

    def hub_pos_for_place_index(pi: int) -> int:
        return cluster_id_of_vertex[pi]

    edges: list[EdgeEstimateModel] = []
    n_g = 0
    n_hw = 0
    n_hc = 0
    n_fb = 0

    for pi in range(n):
        for pj in range(n):
            if pi == pj:
                continue
            pid, qid = places[pi].id, places[pj].id
            ci, cj = cluster_id_of_vertex[pi], cluster_id_of_vertex[pj]
            d_direct = float(dist_m[pi, pj])

            if ci == cj:
                if d_direct <= cfg.straight_walk_approx_max_m:
                    dm, dur = _walk_approx_duration_s(
                        d_direct,
                        detour=cfg.walk_detour_factor,
                        speed_m_s=cfg.walk_speed_m_s,
                    )
                    edges.append(
                        EdgeEstimateModel(
                            from_place_id=pid,
                            to_place_id=qid,
                            distance_meters=dm,
                            duration_seconds=dur,
                            travel_mode_effective="WALK",
                            quality="haversine_walk",
                            primary_hub_travel_mode=None,
                            detail="same cluster; straight-line walk with detour factor",
                            legs=[
                                EdgeLegModel(
                                    kind="walk_approx",
                                    from_place_id=pid,
                                    to_place_id=qid,
                                    travel_mode="WALK",
                                    distance_meters=dm,
                                    duration_seconds=dur,
                                    quality="haversine_walk",
                                )
                            ],
                        )
                    )
                    n_hw += 1
                else:
                    hix = medoid_global[ci]
                    d1 = float(dist_m[pi, hix])
                    d2 = float(dist_m[hix, pj])
                    dm1, dur1 = _walk_approx_duration_s(
                        d1,
                        detour=cfg.walk_detour_factor,
                        speed_m_s=cfg.walk_speed_m_s,
                    )
                    dm2w, dur2w = _walk_approx_duration_s(
                        d2,
                        detour=cfg.walk_detour_factor,
                        speed_m_s=cfg.walk_speed_m_s,
                    )
                    dm = dm1 + dm2w
                    dur = dur1 + dur2w
                    edges.append(
                        EdgeEstimateModel(
                            from_place_id=pid,
                            to_place_id=qid,
                            distance_meters=dm,
                            duration_seconds=dur,
                            travel_mode_effective="WALK",
                            quality="haversine_walk",
                            primary_hub_travel_mode=None,
                            detail="same cluster; medoid chain walk approximation",
                            legs=[
                                EdgeLegModel(
                                    kind="walk_approx",
                                    from_place_id=pid,
                                    to_place_id=places[hix].id,
                                    travel_mode="WALK",
                                    distance_meters=dm1,
                                    duration_seconds=dur1,
                                    quality="haversine_walk",
                                ),
                                EdgeLegModel(
                                    kind="walk_approx",
                                    from_place_id=places[hix].id,
                                    to_place_id=qid,
                                    travel_mode="WALK",
                                    distance_meters=dm2w,
                                    duration_seconds=dur2w,
                                    quality="haversine_walk",
                                ),
                            ],
                        )
                    )
                    n_hw += 1
                continue

            hi = hub_pos_for_place_index(pi)
            hj = hub_pos_for_place_index(pj)
            hix = medoid_global[ci]
            hjx = medoid_global[cj]
            d_to_hub = float(dist_m[pi, hix])
            d_from_hub = float(dist_m[hjx, pj])
            dm1, dur1 = _walk_approx_duration_s(
                d_to_hub,
                detour=cfg.walk_detour_factor,
                speed_m_s=cfg.walk_speed_m_s,
            )
            dm3, dur3 = _walk_approx_duration_s(
                d_from_hub,
                detour=cfg.walk_detour_factor,
                speed_m_s=cfg.walk_speed_m_s,
            )

            hlat1, hlon1 = hub_coords[hi]
            hlat2, hlon2 = hub_coords[hj]
            h_h = haversine_meters(hlat1, hlon1, hlat2, hlon2)
            picked = _pick_primary_hub_matrix_cell(
                triple_matrix,
                hi=hi,
                hj=hj,
                h_h=h_h,
                cfg=cfg,
                use_transit_eff=use_transit_eff,
            )
            el = picked[0] if picked else None
            mid_mode = (
                cast_mode(picked[1])
                if picked
                else cast_mode(
                    matrix_travel_mode_for_hub_separation_m(
                        h_h,
                        bicycle_max_m=cfg.hub_pair_bicycle_max_m,
                        transit_max_m=cfg.hub_pair_transit_max_m,
                        use_transit=use_transit_eff,
                    )
                )
            )
            dm2: float
            dur2: float
            mid_quality: EdgeQuality
            if (
                el is not None
                and _matrix_element_ok(el)
                and el.distance_meters is not None
                and el.duration_seconds is not None
            ):
                dm2 = float(el.distance_meters)
                dur2 = float(el.duration_seconds)
                mid_quality = "google_matrix"
                n_g += 1
            else:
                dm2 = h_h
                dur2 = (
                    h_h / cfg.drive_fallback_speed_m_s
                    if cfg.drive_fallback_speed_m_s > 0
                    else 0.0
                )
                mid_quality = "fallback_estimate"
                n_fb += 1

            total_dm = dm1 + float(dm2) + dm3
            total_dur = dur1 + float(dur2) + dur3
            legs = [
                EdgeLegModel(
                    kind="walk_approx",
                    from_place_id=pid,
                    to_place_id=places[hix].id,
                    travel_mode="WALK",
                    distance_meters=dm1,
                    duration_seconds=dur1,
                    quality="haversine_walk",
                ),
                EdgeLegModel(
                    kind="matrix",
                    from_place_id=places[hix].id,
                    to_place_id=places[hjx].id,
                    travel_mode=mid_mode,
                    distance_meters=float(dm2),
                    duration_seconds=float(dur2),
                    quality=mid_quality,
                ),
                EdgeLegModel(
                    kind="walk_approx",
                    from_place_id=places[hjx].id,
                    to_place_id=qid,
                    travel_mode="WALK",
                    distance_meters=dm3,
                    duration_seconds=dur3,
                    quality="haversine_walk",
                ),
            ]
            edges.append(
                EdgeEstimateModel(
                    from_place_id=pid,
                    to_place_id=qid,
                    distance_meters=total_dm,
                    duration_seconds=total_dur,
                    travel_mode_effective="COMPOSED",
                    quality="hub_chain",
                    primary_hub_travel_mode=mid_mode,
                    detail=f"walk to hub ({places[hix].id}) + {mid_mode} hub leg + walk from hub ({places[hjx].id})",
                    legs=legs,
                )
            )
            n_hc += 1

    cache_hits, cache_misses = routing_cache_metrics()
    notes = (
        f"effective_cluster_link_m={link_m:.1f} (cap {cfg.cluster_link_m}); "
        f"hub primary-leg bands {cfg.hub_pair_bicycle_max_m}/{cfg.hub_pair_transit_max_m} m; "
        f"full hub matrix (BICYCLE + optional TRANSIT + DRIVE per ordered pair); "
        f"matrix HTTP={matrix_http}, matrix elements≈{matrix_elems}, matrix_tiles_skipped={matrix_tiles_skipped}; "
        f"transit_policy={transit_policy_note}; "
        f"routing API cache iv={ROUTING_CACHE_INTEGRATION_VERSION} hits={cache_hits} misses={cache_misses}; "
        "edge qualities: google_matrix / haversine_walk / hub_chain / fallback_estimate."
    )

    stats = GraphBuildStatsModel(
        place_count=n,
        cluster_count=len(roots_sorted),
        hub_hub_matrix_elements=matrix_elems,
        matrix_http_requests=matrix_http,
        edges_stored=len(edges),
        edges_google_matrix=n_g,
        edges_haversine_walk=n_hw,
        edges_hub_chain=n_hc,
        edges_fallback=n_fb,
        routing_cache_hits=cache_hits,
        routing_cache_misses=cache_misses,
        routing_cache_integration_version=ROUTING_CACHE_INTEGRATION_VERSION,
        effective_cluster_link_m=link_m,
        matrix_tiles_skipped=matrix_tiles_skipped,
        transit_matrix_policy=transit_policy_note,
    )

    gj = _geojson_for_places(place_nodes, hub_hub_legs, id_to_latlon)

    return PlaceDistanceGraphModel(
        strategy="cluster_medoids_hub_matrix_adaptive",
        notes=notes,
        places=place_nodes,
        clusters=clusters_out,
        hub_hub_legs=hub_hub_legs,
        edges=edges,
        stats=stats,
        geojson=gj,
    )
