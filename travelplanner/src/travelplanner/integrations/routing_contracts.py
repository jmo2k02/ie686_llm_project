"""Routing **contracts**: artifact types, task JSON shapes, and parsing (no HTTP, no LLM).

This module is the stable boundary between **LLM-produced plans** (``TaskModel`` with
``type=\"routing-check\"``) and **deterministic routing tooling** (Google APIs, clustering).

**Where things live**

* **Deterministic / preferred for data quality:** :func:`travelplanner.integrations.routing_execution.execute_routing_check_task`, :func:`travelplanner.integrations.place_distance_graph.build_place_distance_graph`, CLI ``tp routing-check …``.
* **Thin LangGraph wrapper (optional):** :mod:`travelplanner.integrations.routing_check_agent` — delegates single-OD runs to the same execution path; it does **not** parse natural language.

**Determinism**

* Same validated JSON payload + same Google response (or disk cache hit) → same artifact ``content``.
* Non-determinism is isolated to **Google** (traffic, transit schedules) and **cache eviction**; bump
  :data:`travelplanner.integrations.routing_api_cache.ROUTING_CACHE_INTEGRATION_VERSION` when parsing changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ROUTING_CHECK_TASK_TYPE: str = "routing-check"
"""Must match ``TaskModel.type`` literal."""

ARTIFACT_TYPE_ROUTE_TIMING: str = "route_timing_artifact"
"""Single origin→destination :class:`~travelplanner.schema.route_plan.RoutePlanModel` JSON."""

ARTIFACT_TYPE_PLACE_DISTANCE_GRAPH: str = "place_distance_graph"
"""Many-place graph: :class:`~travelplanner.schema.place_distance_graph.PlaceDistanceGraphModel` JSON."""

ROUTING_REVIEWER_INSTRUCTION_BLOCK: str = """
- For ``type: "routing-check"``: set ``is_valid`` true only if ``text`` is **one JSON object** (no prose)
  with ``kind`` either ``single_od`` or ``place_graph_file``, matching the shapes described for the planner.
  Reject vague instructions like "check distances between hotels" without structured endpoints or file path.
  For ``place_graph_file``, optional ``cluster_context`` on the task must be one of ``dense_urban`` | ``mixed`` | ``sparse`` if present.
""".strip()


ROUTING_PLANNER_INSTRUCTION_BLOCK: str = """
Routing-check tasks (downstream execution is **deterministic** — no prose in ``text``):
- Use ``type``: ``routing-check`` when fixed endpoints or a place list are needed for scheduling.
- ``text`` MUST be **one JSON object** (no markdown) — exactly one of:
  1) **Single route:** ``{"kind":"single_od","origin_address":"…","destination_address":"…"}`` — optional: ``travel_mode`` (default ``drive``), ``departure_time_rfc3339``, ``detail_level``, ``include_transit_alternatives``.
  2) **Many-place graph:** ``{"kind":"place_graph_file","places_json_path":"…"}`` — optional: ``cluster_context``: ``dense_urban`` | ``mixed`` | ``sparse`` (**you** pick this from the user’s trip: city centre → ``dense_urban``, road-trip spacing → ``sparse``, else ``mixed``). When set on the task it **overrides** any preset in the file.
- **Places file (keep dumb):** prefer ``{"stops":[{"address":"…"},…]}`` only — addresses (or legacy ``id``+coords rows). Do **not** require travellers to tune clustering; optional ``cluster_context`` inside the file is still allowed but defaults to ``mixed`` if omitted or invalid.
""".strip()


class SingleOdTaskPayload(BaseModel):
    """Machine-readable payload for one OD route (``computeRoutes``)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["single_od"] = "single_od"
    origin_address: str = Field(min_length=1)
    destination_address: str = Field(min_length=1)
    travel_mode: str = Field(default="drive")
    departure_time_rfc3339: str | None = None
    detail_level: Literal["route_summary", "standard", "full"] = "standard"
    include_transit_alternatives: bool = True


class PlaceGraphFileTaskPayload(BaseModel):
    """Reference a JSON file of places; executor reads the file then builds the distance graph."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["place_graph_file"] = "place_graph_file"
    places_json_path: str = Field(
        min_length=1,
        description='Path to places JSON: ``{"stops":[{"address":"…"}]}`` or legacy array of place objects.',
    )
    cluster_context: Literal["dense_urban", "mixed", "sparse"] | None = Field(
        default=None,
        description=(
            "Walk-cluster preset chosen by the planner. When set, overrides ``cluster_context`` "
            "in the places file (and is the right place for LLM policy — keep the file as addresses only)."
        ),
    )


def parse_routing_check_task_text(
    text: str,
) -> SingleOdTaskPayload | PlaceGraphFileTaskPayload:
    """Parse ``TaskModel.text`` for ``routing-check`` tasks. Raises ``ValueError`` on invalid JSON/shape."""
    raw = text.strip()
    if not raw:
        msg = "routing-check task text is empty"
        raise ValueError(msg)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"routing-check task text must be JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(data, dict):
        msg = "routing-check task JSON must be an object"
        raise ValueError(msg)
    k = data.get("kind")
    if k == "single_od":
        return SingleOdTaskPayload.model_validate(data)
    if k == "place_graph_file":
        return PlaceGraphFileTaskPayload.model_validate(data)
    msg = (
        f"routing-check task kind must be 'single_od' or 'place_graph_file', got {k!r}"
    )
    raise ValueError(msg)


def resolve_places_json_path(path_str: str, *, base_dir: Path | None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base_dir or Path.cwd()) / p
