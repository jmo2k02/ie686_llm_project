"""Many-place distance / duration graph for map plotting and downstream analysis.

Built by :func:`travelplanner.integrations.place_distance_graph.build_place_distance_graph`
using haversine clustering, hub medoids, and batched Google ``computeRouteMatrix`` (not N×N
single-route calls). Pairwise outputs tag **quality** so later pipelines can weight or
re-fetch exact routes where needed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

PLACE_DISTANCE_GRAPH_SCHEMA_VERSION = "1.5"

EdgeQuality = Literal[
    "google_matrix", "haversine_walk", "hub_chain", "fallback_estimate"
]
TravelModeLiteral = Literal["WALK", "DRIVE", "BICYCLE", "TRANSIT", "COMPOSED"]
# Haversine band for hub×hub matrix mode (see hub_pair_bicycle_max_m / hub_pair_transit_max_m).
MatrixPolicyBand = Literal["bicycle", "transit", "drive"]

ClusterContext = Literal["dense_urban", "mixed", "sparse"]


class PlaceDistanceGraphBuildConfig(BaseModel):
    """Advanced tuning for deterministic graph builds (clustering, walk approximations, primary hub-chain mode).

    **Typical product path:** addresses in a JSON file + ``cluster_context`` from the **routing-check
    task** (or defaults); callers rarely need to construct this model by hand — see
    :func:`~travelplanner.integrations.place_distance_graph.place_distance_graph_config_for_context`.
    The hub matrix always requests every non-walk mode per ordered hub pair.
    """

    cluster_link_m: float = Field(
        default=450.0,
        description="Upper cap (metres) for walk-cluster union; effective link may be lower when adaptive.",
        ge=50.0,
        le=5000.0,
    )
    cluster_link_adaptive: bool = Field(
        default=True,
        description="If true, tighten/loosen walk-link using median nearest-neighbour spacing (denser cities → smaller micro-clusters).",
    )
    cluster_link_floor_m: float = Field(
        default=200.0,
        description="Minimum effective walk-link when adaptive (metres).",
        ge=50.0,
        le=2000.0,
    )
    cluster_link_nn_multiplier: float = Field(
        default=2.15,
        description="effective_link = min(cluster_link_m, max(cluster_link_floor_m, median_nn * multiplier)).",
        ge=1.0,
        le=5.0,
    )
    straight_walk_approx_max_m: float = Field(
        default=1300.0,
        description="Within a cluster, pairs closer than this use a single haversine×detour walk estimate (no Google).",
        ge=100.0,
        le=5000.0,
    )
    walk_detour_factor: float = Field(
        default=1.28,
        description="Multiply straight-line metres to approximate network walking distance.",
        ge=1.0,
        le=2.5,
    )
    walk_speed_m_s: float = Field(default=1.35, ge=0.8, le=2.0)
    drive_fallback_speed_m_s: float = Field(
        default=9.0,
        description="Used only when a matrix cell fails — haversine / speed as a coarse drive proxy.",
        ge=3.0,
        le=40.0,
    )
    hub_pair_bicycle_max_m: float = Field(
        default=3500.0,
        validation_alias=AliasChoices("hub_pair_bicycle_max_m", "hub_pair_walk_max_m"),
        description=(
            "Hub-to-hub haversine ≤ this ⇒ Route Matrix **BICYCLE** band. "
            "Shorter separations use bike; longer use TRANSIT then DRIVE (no WALK on hub legs). "
            "Legacy JSON may still use ``hub_pair_walk_max_m`` as an alias for this field."
        ),
        ge=200.0,
        le=50_000.0,
    )
    hub_pair_transit_max_m: float = Field(
        default=18000.0,
        description="Between walk and this haversine ⇒ TRANSIT matrix (if enabled); beyond ⇒ DRIVE.",
        ge=1000.0,
        le=800_000.0,
    )
    use_transit_for_hub_pairs: bool = True
    disable_transit_if_hub_clusters_exceed: int = Field(
        default=14,
        description="If cluster count (hubs) exceeds this, skip TRANSIT matrix legs entirely (fewer slow 100-cap calls).",
        ge=2,
        le=500,
    )
    max_transit_hub_pair_cells: int = Field(
        default=48,
        description="If ordered hub pairs that would use TRANSIT exceed this, skip TRANSIT for the whole build (API + cost guard).",
        ge=0,
        le=5000,
    )
    departure_time_rfc3339: str | None = Field(
        default=None,
        description="Optional departure time for TRANSIT matrix legs.",
    )
    sleep_between_matrix_requests_s: float = Field(default=0.12, ge=0.0, le=5.0)
    max_places_all_pairs: int = Field(
        default=350,
        description="Safety cap: full pairwise edge list grows as O(n²); raise above this unless you add a sparse mode.",
        ge=2,
        le=10_000,
    )


class MapPlaceInputModel(BaseModel):
    """One POI (hotel, restaurant, attraction, …) for routing.

    Supply **either** ``latitude`` + ``longitude`` **or** a non-empty ``address`` (resolved to
    coordinates before clustering / matrix calls). If both are set, coordinates win and no
    geocoding is performed.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(description="Stable id (unique across the batch)")
    name: str = ""
    category: str = Field(
        default="other",
        description="e.g. hotel, restaurant, attraction, coworking — for styling / filters",
    )
    address: str | None = Field(
        default=None,
        description="Free-text address (city + street). Used when latitude/longitude are omitted.",
    )
    latitude: float | None = Field(
        default=None,
        validation_alias=AliasChoices("latitude", "lat"),
    )
    longitude: float | None = Field(
        default=None,
        validation_alias=AliasChoices("longitude", "lng"),
    )

    @model_validator(mode="after")
    def _coords_or_address(self) -> MapPlaceInputModel:
        lat_none = self.latitude is None
        lon_none = self.longitude is None
        if lat_none ^ lon_none:
            msg = "latitude and longitude must both be set or both omitted"
            raise ValueError(msg)
        has_coords = self.latitude is not None
        has_addr = bool(self.address and self.address.strip())
        if not has_coords and not has_addr:
            msg = "each place needs latitude and longitude, or a non-empty address"
            raise ValueError(msg)
        return self


class PlaceNodeModel(BaseModel):
    """Place with clustering assignment (for analysis + GeoJSON)."""

    id: str
    name: str = ""
    category: str = ""
    latitude: float
    longitude: float
    cluster_id: int = Field(ge=0)
    cluster_hub_id: str = Field(description="Medoid place id for this place's cluster")


class ClusterSummaryModel(BaseModel):
    cluster_id: int = Field(ge=0)
    member_place_ids: list[str] = Field(default_factory=list)
    hub_place_id: str
    centroid_latitude: float
    centroid_longitude: float


class HubHubLegModel(BaseModel):
    """One Google Route Matrix cell: directed **hub → hub** for a single ``travel_mode``.

    The builder always requests **BICYCLE**, **TRANSIT** (unless disabled by policy), and **DRIVE**
    for every ordered hub pair (no walking between hubs). Use :meth:`PlaceDistanceGraphModel.hub_hub_leg`
    or filter ``hub_hub_legs`` by ``from_hub_id``, ``to_hub_id``, and ``travel_mode`` for lookups.
    """

    from_hub_id: str
    to_hub_id: str
    travel_mode: TravelModeLiteral
    matrix_policy_band: MatrixPolicyBand | None = Field(
        default=None,
        description=(
            "Which **matrix travel mode** this row came from (``bicycle`` / ``transit`` / ``drive``); "
            "mirrors ``travel_mode`` for non-walk legs. Haversine bands on "
            "``PlaceDistanceGraphBuildConfig`` only affect which cell is chosen as the **default** "
            "mid-leg inside ``edges`` (hub_chain), not which rows exist here."
        ),
    )
    distance_meters: int | None = None
    duration_seconds: float | None = None
    haversine_meters: float = Field(
        ge=0, description="Great-circle distance for sanity checks"
    )
    source: Literal["google_route_matrix"] = "google_route_matrix"
    condition: str | None = Field(
        default=None,
        description="Google matrix element condition, e.g. ROUTE_EXISTS",
    )


class EdgeLegModel(BaseModel):
    """One leg inside a composed or matrix-backed estimate."""

    kind: Literal["walk_approx", "matrix"]
    from_place_id: str
    to_place_id: str
    travel_mode: TravelModeLiteral
    distance_meters: float | None = None
    duration_seconds: float | None = None
    quality: EdgeQuality


class EdgeEstimateModel(BaseModel):
    """Directed stop-to-stop summary (builder emits every ordered pair A≠B)."""

    from_place_id: str
    to_place_id: str
    distance_meters: float | None = None
    duration_seconds: float | None = None
    travel_mode_effective: TravelModeLiteral
    quality: EdgeQuality
    primary_hub_travel_mode: TravelModeLiteral | None = Field(
        default=None,
        description=(
            "When ``quality`` is ``hub_chain``, the **BICYCLE** / **TRANSIT** / **DRIVE** matrix row "
            "used for the middle leg (same as ``legs`` mid entry when present). ``None`` for walk-only "
            "or non-composed edges."
        ),
    )
    detail: str | None = Field(
        default=None,
        description="Human-readable provenance, e.g. hub-chain decomposition",
    )
    legs: list[EdgeLegModel] = Field(default_factory=list)


class GraphBuildStatsModel(BaseModel):
    place_count: int = Field(ge=0)
    cluster_count: int = Field(ge=0)
    hub_hub_matrix_elements: int = Field(
        ge=0, description="Billable matrix cells requested"
    )
    matrix_http_requests: int = Field(ge=0)
    edges_stored: int = Field(ge=0)
    edges_google_matrix: int = Field(ge=0)
    edges_haversine_walk: int = Field(ge=0)
    edges_hub_chain: int = Field(ge=0)
    edges_fallback: int = Field(ge=0)
    routing_cache_hits: int = Field(
        default=0,
        ge=0,
        description="Successful reads from disk cache during this build (matrix + routes).",
    )
    routing_cache_misses: int = Field(
        default=0,
        ge=0,
        description="Cache lookups that fell through to HTTP when caching is enabled.",
    )
    routing_cache_integration_version: str = Field(
        default="",
        description="Cache namespace; bump in routing_api_cache when invalidating stored responses.",
    )
    effective_cluster_link_m: float | None = Field(
        default=None,
        description="Walk-link distance actually used after adaptive adjustment (metres).",
    )
    matrix_tiles_skipped: int = Field(
        default=0,
        ge=0,
        description="Matrix subtiles skipped because no hub pair in that tile needed that travel mode.",
    )
    transit_matrix_policy: str = Field(
        default="",
        description="Why TRANSIT was on/off for hub legs (e.g. budget, hub count).",
    )


class PlaceDistanceGraphModel(BaseModel):
    """Full artifact for routing + execution: places, clusters, hub legs, pairwise edges, GeoJSON.

    **Hub matrix:** ``hub_hub_legs`` lists **every** ordered hub pair × **BICYCLE**, **TRANSIT** (if
    enabled for the build), and **DRIVE** — rough network **distance** / **duration** from Google’s
    Route Matrix. **Within-cluster** movement uses walk approximations only (see ``edges`` with
    ``haversine_walk``).

    Pairwise ``edges`` are **directed** and tagged with ``quality``. For cross-cluster trips,
    ``hub_chain`` totals use one **primary** matrix mode (Haversine bands on the build config);
    inspect ``primary_hub_travel_mode`` or the middle ``legs`` entry to see which mode was used.

    **Lookup:** :meth:`hub_hub_leg` returns the matrix row for a hub pair and mode, or ``None`` if
    missing (e.g. TRANSIT disabled) or unroutable.
    """

    schema_version: str = PLACE_DISTANCE_GRAPH_SCHEMA_VERSION
    strategy: str = Field(
        default="cluster_medoids_hub_matrix_adaptive",
        description="Algorithm id: medoid hubs, mode-aware matrix with tile/mode skipping + adaptive walk-link + transit budget.",
    )
    notes: str = Field(
        default="",
        description="Design rationale: clustering eps, matrix limits, composed pairs, etc.",
    )
    places: list[PlaceNodeModel] = Field(default_factory=list)
    clusters: list[ClusterSummaryModel] = Field(default_factory=list)
    hub_hub_legs: list[HubHubLegModel] = Field(default_factory=list)
    edges: list[EdgeEstimateModel] = Field(default_factory=list)
    stats: GraphBuildStatsModel | None = None
    inter_cluster_transit: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional ordered hub-pair TRANSIT results: primary route plus ``transit_alternatives`` "
            "(``duration_minutes``, ``distance_km``, ``step_summaries`` per option) when stop addresses exist."
        ),
    )
    geojson: dict[str, Any] | None = Field(
        default=None,
        description="GeoJSON FeatureCollection (Points); optional LineStrings for hub legs",
    )

    def hub_hub_leg(
        self, from_hub_id: str, to_hub_id: str, mode: TravelModeLiteral
    ) -> HubHubLegModel | None:
        """Return the hub→hub matrix row for ``mode``, or ``None`` if not in this graph."""
        for leg in self.hub_hub_legs:
            if (
                leg.from_hub_id == from_hub_id
                and leg.to_hub_id == to_hub_id
                and leg.travel_mode == mode
            ):
                return leg
        return None
