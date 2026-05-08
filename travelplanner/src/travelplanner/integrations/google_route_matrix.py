"""Google Routes API ``computeRouteMatrix`` (distanceMatrix v2) — batched OD pairs.

Use for hub-to-hub segments instead of N×N :func:`~travelplanner.integrations.google_routes.compute_route_plan`
calls. Respects element limits (100 for ``TRANSIT``, 625 for other modes).

See: https://developers.google.com/maps/documentation/routes/compute_route_matrix
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from travelplanner.integrations.google_routes import TravelMode, _summarize_routes_http_error
from travelplanner.integrations.routing_api_cache import (
    cache_key,
    read_cached_body,
    record_cache_miss,
    routing_cache_disabled,
    write_cached_body,
)

ROUTES_MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"

_MATRIX_FIELD_MASK = "originIndex,destinationIndex,distanceMeters,duration,status,condition"

# Google documented caps (elements = len(origins) * len(destinations))
_MAX_ELEMENTS_TRANSIT = 100
_MAX_ELEMENTS_OTHER = 625


def _parse_duration_seconds(value: object) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    if not s.endswith("s"):
        return None
    try:
        return float(s[:-1])
    except ValueError:
        return None


def _waypoint_lat_lng(lat: float, lng: float) -> dict[str, Any]:
    return {
        "waypoint": {
            "location": {
                "latLng": {
                    "latitude": lat,
                    "longitude": lng,
                },
            },
        },
    }


@dataclass(frozen=True)
class RouteMatrixElement:
    """One origin×destination cell from ``computeRouteMatrix``."""

    origin_index: int
    destination_index: int
    distance_meters: int | None
    duration_seconds: float | None
    condition: str | None
    raw: dict[str, Any]


def _parse_matrix_response(body: str) -> list[RouteMatrixElement]:
    top: object = json.loads(body)
    rows: list[dict[str, Any]]
    if isinstance(top, list):
        rows = [r for r in top if isinstance(r, dict)]
    else:
        msg = f"unexpected matrix response type: {type(top)}"
        raise RuntimeError(msg)

    out: list[RouteMatrixElement] = []
    for row in rows:
        oi = row.get("originIndex")
        di = row.get("destinationIndex")
        if not isinstance(oi, int) or not isinstance(di, int):
            continue
        dist = row.get("distanceMeters")
        dist_i = int(dist) if isinstance(dist, (int, float)) else None
        dur = _parse_duration_seconds(row.get("duration"))
        cond = row.get("condition")
        cond_s = cond.strip() if isinstance(cond, str) and cond.strip() else None
        out.append(
            RouteMatrixElement(
                origin_index=oi,
                destination_index=di,
                distance_meters=dist_i,
                duration_seconds=dur,
                condition=cond_s,
                raw=row,
            )
        )
    return out


def compute_route_matrix(
    *,
    origins: list[tuple[float, float]],
    destinations: list[tuple[float, float]],
    api_key: str,
    travel_mode: TravelMode,
    departure_time_rfc3339: str | None = None,
) -> list[RouteMatrixElement]:
    """Single HTTP POST; caller must keep ``len(origins)*len(destinations)`` within API limits."""
    trimmed = api_key.strip()
    if not trimmed:
        msg = "GOOGLE_MAPS_API_KEY is empty"
        raise ValueError(msg)
    if not origins or not destinations:
        msg = "origins and destinations must be non-empty"
        raise ValueError(msg)

    body: dict[str, Any] = {
        "origins": [_waypoint_lat_lng(lat, lng) for lat, lng in origins],
        "destinations": [_waypoint_lat_lng(lat, lng) for lat, lng in destinations],
        "travelMode": travel_mode,
    }
    if travel_mode == "DRIVE":
        body["routingPreference"] = "TRAFFIC_UNAWARE"
    if travel_mode == "TRANSIT":
        if departure_time_rfc3339:
            body["departureTime"] = departure_time_rfc3339

    key_hex = cache_key(
        kind="computeRouteMatrix",
        payload=cast(dict[str, Any], body),
        extra=_MATRIX_FIELD_MASK,
        api_key=trimmed,
    )
    if not routing_cache_disabled():
        cached = read_cached_body(key_hex=key_hex)
        if cached is not None:
            return _parse_matrix_response(cached)
        record_cache_miss()

    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        ROUTES_MATRIX_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": trimmed,
            "X-Goog-FieldMask": _MATRIX_FIELD_MASK,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw_body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(_summarize_routes_http_error(exc.code, detail)) from exc

    parsed = _parse_matrix_response(raw_body)
    if not routing_cache_disabled():
        write_cached_body(key_hex=key_hex, body_text=raw_body)
    return parsed


def max_tile_side(travel_mode: TravelMode) -> int:
    """Largest ``s`` with ``s*s`` not exceeding Google's element cap for ``travel_mode``."""
    cap = _MAX_ELEMENTS_TRANSIT if travel_mode == "TRANSIT" else _MAX_ELEMENTS_OTHER
    return int(cap**0.5)


def compute_square_hub_matrix(
    *,
    hubs: list[tuple[float, float]],
    api_key: str,
    travel_mode: TravelMode,
    departure_time_rfc3339: str | None,
    sleep_between_requests_s: float,
) -> dict[tuple[int, int], RouteMatrixElement]:
    """All ordered pairs of hub indices using square tiles (covers full C×C grid)."""
    c = len(hubs)
    out: dict[tuple[int, int], RouteMatrixElement] = {}
    if c == 0:
        return out
    side = max_tile_side(travel_mode)
    oi0 = 0
    while oi0 < c:
        oi1 = min(oi0 + side, c)
        di0 = 0
        while di0 < c:
            di1 = min(di0 + side, c)
            origins = hubs[oi0:oi1]
            destinations = hubs[di0:di1]
            cells = compute_route_matrix(
                origins=origins,
                destinations=destinations,
                api_key=api_key,
                travel_mode=travel_mode,
                departure_time_rfc3339=departure_time_rfc3339,
            )
            for el in cells:
                gi = oi0 + el.origin_index
                gj = di0 + el.destination_index
                out[(gi, gj)] = el
            di0 = di1
            if sleep_between_requests_s > 0 and (oi0 + side < c or di0 < c):
                time.sleep(sleep_between_requests_s)
        oi0 = oi1
        if sleep_between_requests_s > 0 and oi0 < c:
            time.sleep(sleep_between_requests_s)
    return out


def matrix_travel_mode_for_hub_separation_m(
    hub_separation_m: float,
    *,
    bicycle_max_m: float,
    transit_max_m: float,
    use_transit: bool,
) -> TravelMode:
    """Return the Route Matrix **travelMode** for one ordered hub pair.

    ``hub_separation_m`` is the great-circle distance between hubs (metres). Bands:
    ≤ ``bicycle_max_m`` → ``BICYCLE``; then ≤ ``transit_max_m`` (if ``use_transit``) → ``TRANSIT``;
    else ``DRIVE``. Walking between hubs is never selected here (walk stays inside clusters in
    :mod:`travelplanner.integrations.place_distance_graph`).
    """
    if hub_separation_m <= bicycle_max_m:
        return cast(TravelMode, "BICYCLE")
    if use_transit and hub_separation_m <= transit_max_m:
        return cast(TravelMode, "TRANSIT")
    return cast(TravelMode, "DRIVE")


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


_MODE_CODE: dict[TravelMode, int] = {"BICYCLE": 1, "TRANSIT": 2, "DRIVE": 3}


def hub_pair_mode_matrix(
    hubs: list[tuple[float, float]],
    *,
    bicycle_max_m: float,
    transit_max_m: float,
    use_transit: bool,
) -> np.ndarray:
    """Per ordered hub pair (i,j), i≠j: int 1=BICYCLE, 2=TRANSIT, 3=DRIVE (no WALK)."""
    c = len(hubs)
    em = np.zeros((c, c), dtype=np.int8)
    for i in range(c):
        lat1, lon1 = hubs[i]
        for j in range(c):
            if i == j:
                continue
            lat2, lon2 = hubs[j]
            h_m = haversine_meters(lat1, lon1, lat2, lon2)
            if h_m <= bicycle_max_m:
                em[i, j] = 1
            elif use_transit and h_m <= transit_max_m:
                em[i, j] = 2
            else:
                em[i, j] = 3
    return em


def compute_mode_bucket_matrices(
    *,
    hubs: list[tuple[float, float]],
    api_key: str,
    bicycle_max_m: float,
    transit_max_m: float,
    use_transit: bool,
    departure_time_rfc3339: str | None,
    sleep_between_requests_s: float,
) -> tuple[dict[tuple[int, int], RouteMatrixElement], int, int, int]:
    """Compute hub×hub matrix cells using per-pair mode, **skipping HTTP** when a tile has no matching pairs.

    Skips an entire ``travelMode`` if no hub pair needs that mode. Returns
    ``(cells, http_requests, billable_elements_requested, matrix_tiles_skipped)``.
    """
    c = len(hubs)
    out: dict[tuple[int, int], RouteMatrixElement] = {}
    if c <= 1:
        return out, 0, 0, 0

    em = hub_pair_mode_matrix(
        hubs,
        bicycle_max_m=bicycle_max_m,
        transit_max_m=transit_max_m,
        use_transit=use_transit,
    )

    modes: list[TravelMode] = ["BICYCLE", "DRIVE"]
    if use_transit:
        modes.insert(1, "TRANSIT")

    http_requests = 0
    billable_elements = 0
    tiles_skipped = 0

    for mode in modes:
        code = _MODE_CODE[mode]
        if not np.any(em == code):
            continue
        side = max_tile_side(mode)
        oi0 = 0
        while oi0 < c:
            oi1 = min(oi0 + side, c)
            di0 = 0
            while di0 < c:
                di1 = min(di0 + side, c)
                gii = np.arange(oi0, oi1, dtype=np.int32)[:, None]
                gjj = np.arange(di0, di1, dtype=np.int32)[None, :]
                sub = em[oi0:oi1, di0:di1]
                if not np.any((sub == code) & (gii != gjj)):
                    tiles_skipped += 1
                    di0 = di1
                    continue
                origins = hubs[oi0:oi1]
                destinations = hubs[di0:di1]
                cells = compute_route_matrix(
                    origins=origins,
                    destinations=destinations,
                    api_key=api_key,
                    travel_mode=mode,
                    departure_time_rfc3339=departure_time_rfc3339 if mode == "TRANSIT" else None,
                )
                http_requests += 1
                billable_elements += len(origins) * len(destinations)
                for el in cells:
                    gi = oi0 + el.origin_index
                    gj = di0 + el.destination_index
                    if gi == gj:
                        continue
                    h_m = haversine_meters(hubs[gi][0], hubs[gi][1], hubs[gj][0], hubs[gj][1])
                    expected = matrix_travel_mode_for_hub_separation_m(
                        h_m,
                        bicycle_max_m=bicycle_max_m,
                        transit_max_m=transit_max_m,
                        use_transit=use_transit,
                    )
                    if expected != mode:
                        continue
                    out[(gi, gj)] = el
                di0 = di1
                if sleep_between_requests_s > 0:
                    time.sleep(sleep_between_requests_s)
            oi0 = oi1
            if sleep_between_requests_s > 0:
                time.sleep(sleep_between_requests_s)
    return out, http_requests, billable_elements, tiles_skipped


def compute_all_travel_modes_hub_matrices(
    *,
    hubs: list[tuple[float, float]],
    api_key: str,
    use_transit: bool,
    departure_time_rfc3339: str | None,
    sleep_between_requests_s: float,
) -> tuple[dict[tuple[int, int, TravelMode], RouteMatrixElement], int, int]:
    """Route Matrix for **every** ordered hub pair (i≠j) in **BICYCLE**, optional **TRANSIT**, and **DRIVE**.

    Keys are ``(origin_index, destination_index, travel_mode)``. Diagonal pairs are skipped.
    Returns ``(cells, http_requests, billable_elements)``.
    """
    c = len(hubs)
    out: dict[tuple[int, int, TravelMode], RouteMatrixElement] = {}
    if c <= 1:
        return out, 0, 0

    modes: list[TravelMode] = ["BICYCLE", "DRIVE"]
    if use_transit:
        modes.insert(1, "TRANSIT")

    http_requests = 0
    billable_elements = 0

    for mode in modes:
        side = max_tile_side(mode)
        oi0 = 0
        while oi0 < c:
            oi1 = min(oi0 + side, c)
            di0 = 0
            while di0 < c:
                di1 = min(di0 + side, c)
                origins = hubs[oi0:oi1]
                destinations = hubs[di0:di1]
                cells = compute_route_matrix(
                    origins=origins,
                    destinations=destinations,
                    api_key=api_key,
                    travel_mode=mode,
                    departure_time_rfc3339=departure_time_rfc3339 if mode == "TRANSIT" else None,
                )
                http_requests += 1
                billable_elements += len(origins) * len(destinations)
                for el in cells:
                    gi = oi0 + el.origin_index
                    gj = di0 + el.destination_index
                    if gi == gj:
                        continue
                    out[(gi, gj, mode)] = el
                di0 = di1
                if sleep_between_requests_s > 0:
                    time.sleep(sleep_between_requests_s)
            oi0 = oi1
            if sleep_between_requests_s > 0:
                time.sleep(sleep_between_requests_s)

    return out, http_requests, billable_elements
