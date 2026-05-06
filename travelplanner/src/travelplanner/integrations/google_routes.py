"""Google Routes API: single-route planner and geocoding.

This module provides:
- ``TravelMode`` literal type
- ``geocode_address_to_lat_lng()`` — Google Geocoding API
- ``compute_route_plan()`` — Google Routes API (computeRoutes)
- ``resolve_travel_mode()`` — string → TravelMode conversion
- ``route_plan_to_jsonable()`` — RoutePlanModel → dict

See: https://developers.google.com/maps/documentation/routes
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Literal, cast

from travelplanner.schema.route_plan import (
    AlternativeTransitRouteModel,
    RouteLegModel,
    RouteMetricModel,
    RoutePlanModel,
    RouteRequestModel,
    RouteStepModel,
    RouteStepDetailModel,
)

TravelMode = Literal["BICYCLE", "DRIVE", "TRANSIT", "WALK"]

_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

_ROUTE_FIELD_MASK = (
    "routes.duration,routes.distanceMeters,routes.legs.steps.transitDetails,"
    "routes.legs.steps.distanceMeters,routes.legs.steps.duration,routes.legs.steps.htmlInstructions,"
    "routes.legs.distanceMeters,routes.legs.duration,routes.legs.startLocation.latLng,"
    "routes.legs.endLocation.latLng,routes.legs.startAddress,routes.legs.endAddress"
)


def _summarize_routes_http_error(code: int, detail: str) -> str:
    try:
        body = json.loads(detail)
        msg = body.get("error", {}).get("message", detail)
    except Exception:
        msg = detail
    return f"Routes API HTTP {code}: {msg}"


def resolve_travel_mode(mode: str) -> TravelMode:
    """Coerce a travel-mode string to a canonical ``TravelMode``."""
    upper = mode.strip().upper()
    if upper in ("BICYCLE", "DRIVE", "TRANSIT", "WALK"):
        return cast(TravelMode, upper)
    if upper in ("BIKE", "CYCLING", "BIKING"):
        return "BICYCLE"
    if upper in ("CAR", "DRIVING", "DRIVE"):
        return "DRIVE"
    if upper in ("TRANSIT", "PUBLIC", "BUS", "TRAIN", "SUBWAY", "METRO"):
        return "TRANSIT"
    if upper in ("WALK", "WALKING", "FOOT"):
        return "WALK"
    msg = f"unknown travel mode: {mode!r}"
    raise ValueError(msg)


def geocode_address_to_lat_lng(address: str, api_key: str) -> tuple[float, float]:
    """Geocode a free-text address to ``(latitude, longitude)`` via Google Geocoding API."""
    addr = address.strip()
    if not addr:
        raise ValueError("address is empty")
    key = api_key.strip()
    if not key:
        raise ValueError("api_key is empty")

    params = urllib.parse.urlencode({"address": addr, "key": key})
    url = f"{_GEOCODE_URL}?{params}"
    req = urllib.request.Request(url, headers={"X-Requested-With": "XMLHttpRequest"})

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Geocoding HTTP {exc.code}: {detail}") from exc

    data = json.loads(body)
    status = data.get("status", "")
    if status != "OK":
        msg = data.get("error_message", status)
        raise RuntimeError(f"Geocoding status={status}: {msg}")

    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"No results for address: {addr!r}")

    loc = results[0].get("geometry", {}).get("location", {})
    lat = loc.get("lat")
    lng = loc.get("lng")
    if lat is None or lng is None:
        raise RuntimeError(f"Missing location in geocoding response for: {addr!r}")
    return float(lat), float(lng)


def _parse_duration_seconds(value: str | None) -> float | None:
    if not value or not value.endswith("s"):
        return None
    try:
        return float(value[:-1])
    except ValueError:
        return None


def _build_step_summary(step: dict[str, Any]) -> str:
    html = step.get("htmlInstructions", "") or ""
    import re

    text = re.sub(r"<[^>]+>", "", html)
    return text.strip()


def _parse_steps(legs: list[dict[str, Any]]) -> list[RouteStepModel]:
    steps_out: list[RouteStepModel] = []
    for leg in legs:
        for step in leg.get("steps", []):
            si: dict[str, Any] = step
            dist = si.get("distanceMeters")
            dur = _parse_duration_seconds(si.get("duration", {}).get("value"))
            travel_mode = si.get("travelMode", "").upper()
            detail = RouteStepDetailModel(
                distance_meters=dist,
                duration_seconds=dur,
                start_lat_lng=si.get("startLocation", {}).get("latLng"),
                end_lat_lng=si.get("endLocation", {}).get("latLng"),
                html_instructions=si.get("htmlInstructions", ""),
                travel_mode=travel_mode,
            )
            steps_out.append(
                RouteStepModel(
                    kind=travel_mode,
                    summary=_build_step_summary(si),
                    distance_meters=dist,
                    duration_seconds=dur,
                    detail=detail,
                )
            )
    return steps_out


def _parse_route(route: dict[str, Any]) -> RoutePlanModel:
    legs = route.get("legs", [])
    steps = _parse_steps(legs)

    total_dist = route.get("distanceMeters", 0)
    total_dur_raw = route.get("duration", "")
    total_dur = _parse_duration_seconds(total_dur_raw)

    start_addr = legs[0].get("startAddress", "") if legs else ""
    end_addr = legs[-1].get("endAddress", "") if legs else ""

    metrics = RouteMetricModel(
        distance_meters=total_dist,
        distance_km=round(total_dist / 1000.0, 3),
        duration_seconds=total_dur or 0.0,
    )

    transit_alts: list[AlternativeTransitRouteModel] = []
    for raw_alt in route.get("transitAlternatives", []):
        alt_legs = raw_alt.get("legs", [])
        alt_steps = _parse_steps(alt_legs)
        alt_dist = raw_alt.get("distanceMeters", 0)
        alt_dur_raw = raw_alt.get("duration", "")
        alt_dur = _parse_duration_seconds(alt_dur_raw)
        alt_metrics = RouteMetricModel(
            distance_meters=alt_dist,
            distance_km=round(alt_dist / 1000.0, 3),
            duration_seconds=alt_dur or 0.0,
        )
        transit_alts.append(
            AlternativeTransitRouteModel(
                rank=raw_alt.get("routeIndex", 0),
                metrics=alt_metrics,
                steps=alt_steps,
            )
        )

    return RoutePlanModel(
        request=RouteRequestModel(
            origin=start_addr,
            destination=end_addr,
            travel_mode="TRANSIT",
        ),
        metrics=metrics,
        steps=steps,
        transit_alternatives=transit_alts,
    )


def compute_route_plan(
    *,
    origin: str,
    destination: str,
    api_key: str,
    travel_mode: TravelMode,
    departure_time_rfc3339: str | None = None,
    detail_level: str = "standard",
    include_transit_alternatives: bool = False,
) -> RoutePlanModel:
    """Compute a single route via Google Routes API (``computeRoutes``)."""
    key = api_key.strip()
    if not key:
        raise ValueError("api_key is empty")
    if not origin.strip() or not destination.strip():
        raise ValueError("origin and destination must be non-empty")

    body: dict[str, Any] = {
        "origin": {"location": {"latLng": {"latitude": 0, "longitude": 0}}},
        "destination": {"location": {"latLng": {"latitude": 0, "longitude": 0}}},
        "travelMode": travel_mode,
    }

    body["origin"] = {"address": origin.strip()}
    body["destination"] = {"address": destination.strip()}

    if travel_mode == "DRIVE":
        body["routingPreference"] = "TRAFFIC_UNAWARE"
    if travel_mode == "TRANSIT":
        body["transitRoutingPreferences"] = ["LESS_WALKING", "FEWER_TRANSFERS"]
        if departure_time_rfc3339:
            body["departureTime"] = departure_time_rfc3339

    extra_mask = ""
    if include_transit_alternatives and travel_mode == "TRANSIT":
        body["computeAlternativeRoutes"] = True
        extra_mask = ",routes.transitAlternatives"

    field_mask = _ROUTE_FIELD_MASK + extra_mask

    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        _ROUTES_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": key,
            "X-Goog-FieldMask": field_mask,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(_summarize_routes_http_error(exc.code, detail)) from exc

    data = json.loads(raw)
    routes = data.get("routes", [])
    if not routes:
        raise RuntimeError(f"No routes returned for {origin!r} → {destination!r}")
    return _parse_route(routes[0])


def route_plan_to_jsonable(plan: RoutePlanModel) -> dict[str, Any]:
    """Serialize ``RoutePlanModel`` to a JSON-serializable dict (for artifact storage)."""
    out: dict[str, Any] = {
        "request": {
            "origin": plan.request.origin,
            "destination": plan.request.destination,
            "travel_mode": plan.request.travel_mode,
        },
        "metrics": {
            "distance_meters": plan.metrics.distance_meters,
            "distance_km": plan.metrics.distance_km,
            "duration_seconds": plan.metrics.duration_seconds,
        },
        "steps": [
            {
                "kind": s.kind,
                "summary": s.summary,
                "distance_meters": s.distance_meters,
                "duration_seconds": s.duration_seconds,
            }
            for s in plan.steps
        ],
    }
    if plan.transit_alternatives:
        out["transit_alternatives"] = [
            {
                "rank": a.rank,
                "metrics": {
                    "distance_meters": a.metrics.distance_meters,
                    "distance_km": a.metrics.distance_km,
                    "duration_seconds": a.metrics.duration_seconds,
                },
                "step_summaries": [s.summary for s in a.steps],
            }
            for a in plan.transit_alternatives
        ]
    return out
