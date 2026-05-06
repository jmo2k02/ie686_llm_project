"""Dead-simple A→B distance/duration lookup from a pre-computed place-distance graph.

This module reads a JSON artefact containing places, clusters, hub-to-hub legs,
and pairwise edges. No Google API calls — the graph is pure read-only lookup data.

Quickstart
----------
>>> from travelplanner.routing_lookup import load_routing_lookup, RouteOption
>>> lu = load_routing_lookup("/path/to/graph.json")
>>>
>>> # All places and their IDs
>>> for p in lu.places():
...     print(p["id"], p["name"], p["latitude"], p["longitude"])
>>>
>>> # All travel-mode options between two places
>>> options = lu.get("stop_0", "stop_5")
>>> for opt in options:
...     print(f"{opt.travel_mode:10s}  {opt.distance_meters:7.0f}m  {opt.duration_minutes:5.1f}min  [{opt.quality}]")
>>>
>>> # Filter to only TRANSIT options
>>> transit_options = lu.get("stop_0", "stop_5", travel_mode="TRANSIT")
>>>
>>> # Just the IDs
>>> ids = lu.place_ids()

NEW — Convenience API
---------------------
>>> lu = load_routing_lookup(graph)
>>> ids = lu.place_ids()
>>>
>>> # Human-readable table of all options for a pair
>>> print(lu.options(ids[0], ids[1]).table())
>>>
>>> # Pick the best option by preference
>>> lu.options(ids[0], ids[1]).choose("fastest")   # fastest TRANSIT
>>> lu.options(ids[0], ids[1]).choose("walk")     # first WALK option
>>> lu.options(ids[0], ids[1]).choose("cheapest") # WALK if available
>>>
>>> # One-liner comparisons
>>> from travelplanner.routing_lookup import compare, trip_summary, best_mode
>>> print(compare(graph, ids[0], ids[1]))
>>> print(trip_summary(graph, ids[:3]))
>>> opt = best_mode(graph, ids[0], ids[1])
>>> print(f"Take {opt.travel_mode}, {opt.distance_km:.1f}km, {opt.duration_minutes:.1f}min")
>>>
>>> # All-to-all matrix (rows=origins, cols=destinations, cell=best mode + duration)
>>> mat = lu.matrix()
>>> print(mat.as_table())
>>> # Matrix with ALL modes per pair
>>> mat_all = lu.matrix(all_modes=True)
>>> print(mat_all.duration_matrix())

Data model
----------
- **Same cluster**: only WALK (haversine approximation) — no Google API call made.
- **Cross cluster**: walks to hub → one hub-to-hub mode → walk from hub.
  The hub-to-hub legs come from Google's Route Matrix (BICYCLE / TRANSIT / DRIVE per ordered pair).
  All three modes are stored; ``edges`` holds the *selected* best option per pair.
  Use ``get(a, b)`` to get **all available modes**, or ``get(a, b, "TRANSIT")`` to filter.

Edge cases
----------
- ``get(id, id)`` returns one WALK option with 0m / 0s.
- Unknown place IDs return an empty list.
- ``travel_mode`` filter with no matching options returns an empty list.
- ``options()`` returns a RouteOptions with helpers; ``get()`` still returns raw list.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import json as _json
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WALK = "WALK"
BICYCLE = "BICYCLE"
TRANSIT = "TRANSIT"
DRIVE = "DRIVE"
COMPOSED = "COMPOSED"
ALL_MODES = [WALK, BICYCLE, TRANSIT, DRIVE, COMPOSED]

TravelModeLiteral = Literal["WALK", "BICYCLE", "TRANSIT", "DRIVE", "COMPOSED"]

__all__ = [
    "RoutingLookup",
    "RouteOption",
    "RouteOptions",
    "RoutingMatrix",
    "WALK",
    "BICYCLE",
    "TRANSIT",
    "DRIVE",
    "COMPOSED",
    "ALL_MODES",
    "TravelModeLiteral",
    "compare",
    "trip_summary",
    "best_mode",
    "distance_lookup",
    "load_routing_lookup",
    "PlaceResolutionError",
    "resolve_place_id",
    "distance_between",
    "closest_to",
]


# ---------------------------------------------------------------------------
# RouteOption — already defined, keep as-is
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouteOption:
    """One travel option for an A→B pair.

    Attributes
    ----------
    from_place_id:
        Origin place ID.
    to_place_id:
        Destination place ID.
    travel_mode:
        WALK | BICYCLE | TRANSIT | DRIVE | COMPOSED.
        WALK = same-cluster or hub-walk leg.
        COMPOSED = walk-to-hub + hub-leg + walk-from-hub (cross-cluster).
    distance_meters:
        Total distance in metres (walk legs + hub leg combined).
    duration_seconds:
        Total duration in seconds.
    quality:
        How this option was computed:

        - ``google_matrix`` — exact result from Google Route Matrix.
        - ``haversine_walk`` — straight-line × detour factor (same cluster only).
        - ``hub_chain`` — composed: walk → hub-matrix leg → walk (cross-cluster).
        - ``fallback_estimate`` — haversine / speed proxy when matrix missed.
    primary_mode:
        The hub mode used when ``travel_mode`` is COMPOSED
        (BICYCLE | TRANSIT | DRIVE | None).
    is_same_cluster:
        True when both places are in the same walkable cluster.
    legs:
        Breakdown of sub-legs:

        - ``walk_approx`` — same-cluster haversine walk.
        - ``matrix`` — one Google Route Matrix cell (hub→hub for cross-cluster,
          or the full route for same-cluster when available).
    """

    from_place_id: str
    to_place_id: str
    travel_mode: str
    distance_meters: float
    duration_seconds: float
    quality: str
    primary_mode: str | None
    is_same_cluster: bool
    legs: tuple[dict, ...]  # subgraph of the raw edge legs array

    @property
    def distance_km(self) -> float:
        """Distance in kilometres."""
        return self.distance_meters / 1000.0

    @property
    def duration_minutes(self) -> float:
        """Duration in minutes (rounded to 1 decimal)."""
        return self.duration_seconds / 60.0


# ---------------------------------------------------------------------------
# RouteOptions — result container with useful methods
# ---------------------------------------------------------------------------


@dataclass
class RouteOptions:
    """A list of RouteOption with helpers for ranking and filtering.

    Parameters
    ----------
    options:
        List of available RouteOption objects for a given pair.
    from_id:
        Origin place ID.
    to_id:
        Destination place ID.
    """

    options: list[RouteOption]
    from_id: str
    to_id: str

    def best(self, by: str = "duration") -> RouteOption | None:
        """Return best option by 'duration' or 'distance'.

        Parameters
        ----------
        by:
            ``"duration"`` (default) → fastest option.
            ``"distance"`` → shortest option.

        Returns
        -------
        RouteOption | None
            The best option by the given criterion, or None if no options.
        """
        if not self.options:
            return None
        key = "duration_seconds" if by == "duration" else "distance_meters"
        return min(self.options, key=lambda o: getattr(o, key))

    def filter(self, mode: str) -> "RouteOptions":
        """Return new RouteOptions with only this travel mode.

        Parameters
        ----------
        mode:
            Travel mode to keep, e.g. ``"WALK"``, ``"TRANSIT"``, ``"BICYCLE"``.

        Returns
        -------
        RouteOptions
            New container with only matching options (may be empty).
        """
        return RouteOptions(
            options=[o for o in self.options if o.travel_mode == mode],
            from_id=self.from_id,
            to_id=self.to_id,
        )

    def modes(self) -> list[str]:
        """List of available travel modes, e.g. ``['WALK', 'BICYCLE', 'TRANSIT']``."""
        seen: dict[str, bool] = {}
        for o in self.options:
            seen[o.travel_mode] = True
        return list(seen.keys())

    def table(self) -> str:
        """Human-readable ASCII table of all options.

        Returns
        -------
        str
            Formatted table with columns: Mode, Distance, Duration, Quality.
        """
        if not self.options:
            return f"{self.from_id} → {self.to_id}: (no options)"

        lines = [f"{self.from_id} → {self.to_id}"]
        lines.append("-" * 55)
        lines.append(
            f"{'Mode':<12} {'Distance':>18}  {'Duration (s/min)':>18}  {'Quality'}"
        )
        lines.append("-" * 55)
        for o in self.options:
            dist = f"{o.distance_meters:.0f}m  ({o.distance_km:.1f}km)"
            dur = f"{o.duration_seconds:.0f}s  ({o.duration_minutes:.1f}min)"
            lines.append(f"{o.travel_mode:<12} {dist:>18}  {dur:>18}  {o.quality}")
        lines.append("-" * 55)
        return "\n".join(lines)

    def choose(self, preference: str) -> RouteOption | None:
        """Pick based on preference string.

        Parameters
        ----------
        preference:
            One of:

            - ``"fastest"`` / ``"shortest"`` → best by duration/distance
            - ``"transit"`` / ``"bike"`` / ``"walk"`` / ``"drive"`` → first matching mode
            - ``"cheapest"`` → WALK if available, else cheapest by duration

        Returns
        -------
        RouteOption | None
            Selected option, or None if no match.
        """
        pref = preference.lower().strip()

        if pref in ("fastest", "duration"):
            return self.best(by="duration")
        if pref in ("shortest", "distance"):
            return self.best(by="distance")

        mode_map = {
            "transit": TRANSIT,
            "bike": BICYCLE,
            "bicycle": BICYCLE,
            "walk": WALK,
            "drive": DRIVE,
        }
        target = mode_map.get(pref)
        if target:
            filtered = self.filter(target).options
            return filtered[0] if filtered else None

        if pref == "cheapest":
            walk_opts = self.filter(WALK).options
            if walk_opts:
                return walk_opts[0]
            return self.best(by="duration")

        return None


# ---------------------------------------------------------------------------
# RoutingMatrix — all-to-all lookup table
# ---------------------------------------------------------------------------


class RoutingMatrix:
    """All-to-all lookup table for a graph.

    Supports two views:
    - ``all_modes=False`` (default): each A→B pair shows only the best (fastest) option.
    - ``all_modes=True``: each A→B pair shows all available mode options.

    Parameters
    ----------
    graph_data:
        dict (parsed JSON), Path to a .json file, or a JSON string.
    all_modes:
        False (default) → each cell shows only the best mode.
        True → each cell shows all available modes.
    """

    def __init__(
        self, graph_data: dict | Path | str, *, all_modes: bool = False
    ) -> None:
        self._lookup = RoutingLookup(graph_data)
        self._all_modes = all_modes
        self._place_ids: list[str] = self._lookup.place_ids()
        self._n = len(self._place_ids)
        # Pre-build the index maps
        self._id_to_idx: dict[str, int] = {
            pid: i for i, pid in enumerate(self._place_ids)
        }
        self._idx_to_id: list[str] = self._place_ids

    # ------------------------------------------------------------------
    # Core lookups
    # ------------------------------------------------------------------

    def best(self, from_id: str, to_id: str) -> RouteOption | None:
        """Best (fastest) option for this pair.

        Parameters
        ----------
        from_id:
            Origin place ID.
        to_id:
            Destination place ID.

        Returns
        -------
        RouteOption | None
            Fastest option, or None if either ID is unknown.
        """
        opts = self._lookup.get(from_id, to_id)
        if not opts:
            return None
        return min(opts, key=lambda o: o.duration_seconds)

    def all_options(self, from_id: str, to_id: str) -> list[RouteOption]:
        """All available mode options for this pair.

        Parameters
        ----------
        from_id:
            Origin place ID.
        to_id:
            Destination place ID.

        Returns
        -------
        list[RouteOption]
            All options (may be empty if either ID is unknown).
        """
        return list(self._lookup.get(from_id, to_id))

    def row(self, from_id: str) -> dict[str, RouteOption | None]:
        """Best option from this place to every other place.

        Parameters
        ----------
        from_id:
            Origin place ID.

        Returns
        -------
        dict[str, RouteOption | None]
            Mapping from each destination ID to its best option (or None).
        """
        result: dict[str, RouteOption | None] = {}
        for pid in self._place_ids:
            result[pid] = self.best(from_id, pid)
        return result

    def all_rows(self, from_id: str) -> dict[str, list[RouteOption]]:
        """All mode options from this place to every other place.

        Parameters
        ----------
        from_id:
            Origin place ID.

        Returns
        -------
        dict[str, list[RouteOption]]
            Mapping from each destination ID to all its options.
        """
        result: dict[str, list[RouteOption]] = {}
        for pid in self._place_ids:
            result[pid] = self.all_options(from_id, pid)
        return result

    # ------------------------------------------------------------------
    # Matrix views
    # ------------------------------------------------------------------

    def distance_matrix(self) -> list[list[float]]:
        """M × M matrix of distances (metres).

        Rows and columns are in ``place_ids()`` order.
        Diagonal is 0m (same place).
        """
        matrix: list[list[float]] = []
        for from_id in self._place_ids:
            row: list[float] = []
            for to_id in self._place_ids:
                if from_id == to_id:
                    row.append(0.0)
                else:
                    opt = self.best(from_id, to_id)
                    row.append(opt.distance_meters if opt else 0.0)
            matrix.append(row)
        return matrix

    def duration_matrix(self) -> list[list[float]]:
        """M × M matrix of durations (seconds).

        Rows and columns are in ``place_ids()`` order.
        Diagonal is 0s (same place).
        """
        matrix: list[list[float]] = []
        for from_id in self._place_ids:
            row: list[float] = []
            for to_id in self._place_ids:
                if from_id == to_id:
                    row.append(0.0)
                else:
                    opt = self.best(from_id, to_id)
                    row.append(opt.duration_seconds if opt else 0.0)
            matrix.append(row)
        return matrix

    def mode_matrix(self) -> list[list[str]]:
        """M × M matrix of best travel mode per pair (e.g. ``'BICYCLE'``).

        Rows and columns are in ``place_ids()`` order.
        Empty string ``''`` means no option available.
        """
        matrix: list[list[str]] = []
        for from_id in self._place_ids:
            row: list[str] = []
            for to_id in self._place_ids:
                if from_id == to_id:
                    row.append(WALK)
                else:
                    opt = self.best(from_id, to_id)
                    row.append(opt.travel_mode if opt else "")
            matrix.append(row)
        return matrix

    def as_table(self) -> str:
        """Human-readable matrix: rows=origins, cols=destinations, cell=best mode + duration.

        Returns
        -------
        str
            Formatted table with header row of destination IDs,
            then one row per origin showing best mode and duration for each destination.
        """
        ids = self._place_ids

        # Column widths
        id_width = max(len(str(i)) for i in ids) if ids else 4
        id_width = max(id_width, 6)

        lines: list[str] = []

        # Header row
        header = "FROM   " + "".join(f"{str(i):>{id_width + 14}} " for i in ids)
        lines.append(header)
        lines.append("-" * len(header))

        # Data rows
        for from_id in ids:
            cells = ""
            for to_id in ids:
                if from_id == to_id:
                    cells += f"{'=' * (id_width + 14):>{id_width + 14}} "
                else:
                    opt = self.best(from_id, to_id)
                    if opt:
                        label = f"{opt.travel_mode} {opt.duration_minutes:.0f}min"
                    else:
                        label = "-"
                    cells += f"{label:>{id_width + 14}} "
            lines.append(f"{from_id:<6} {cells}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def place_ids(self) -> list[str]:
        """All place IDs in insertion order."""
        return list(self._place_ids)


# ---------------------------------------------------------------------------
# RoutingLookup — enhanced with new convenience methods
# ---------------------------------------------------------------------------


class RoutingLookup:
    """Read-only lookup over a pre-computed place-distance graph.

    Parameters
    ----------
    graph_data:
        dict (parsed JSON), Path to a .json file, or a JSON string.
    """

    def __init__(self, graph_data: dict | Path | str) -> None:
        _graph = self._resolve(graph_data)
        self._data: dict[str, Any] = _graph if isinstance(_graph, Mapping) else {}
        self._places: list[dict[str, Any]] = self._data.get("places", [])
        self._clusters: list[dict[str, Any]] = self._data.get("clusters", [])
        self._edges: list[dict[str, Any]] = self._data.get("edges", [])
        self._hub_hub_legs: list[dict[str, Any]] = self._data.get("hub_hub_legs", [])

        # Indexes for fast lookups
        self._place_id_to_info: dict[str, dict[str, Any]] = {
            p["id"]: p for p in self._places
        }
        self._cluster_id_to_members: dict[int, list[str]] = {}
        self._cluster_id_to_hub: dict[int, str] = {}
        for c in self._clusters:
            cid = c["cluster_id"]
            self._cluster_id_to_members[cid] = c["member_place_ids"]
            self._cluster_id_to_hub[cid] = c["hub_place_id"]

        # edges indexed by (from, to)
        self._edge_map: dict[tuple[str, str], dict[str, Any]] = {
            (e["from_place_id"], e["to_place_id"]): e for e in self._edges
        }

        # hub_hub_legs indexed by (from_hub_id, to_hub_id, travel_mode)
        self._hub_leg_map: dict[tuple[str, str, str], dict[str, Any]] = {}
        for leg in self._hub_hub_legs:
            key = (leg["from_hub_id"], leg["to_hub_id"], leg["travel_mode"])
            self._hub_leg_map[key] = leg

    # ------------------------------------------------------------------
    # Input resolution
    # ------------------------------------------------------------------

    @classmethod
    def _resolve(cls, graph_data: dict | Path | str) -> dict[str, Any]:
        if isinstance(graph_data, Path):
            path = graph_data.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Graph file not found: {path}")
            return _json.loads(path.read_text())
        if isinstance(graph_data, str):
            p = Path(graph_data).expanduser().resolve()
            if p.is_file():
                return _json.loads(p.read_text())
            return _json.loads(graph_data)
        return graph_data

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def places(self) -> list[dict[str, Any]]:
        """All places.

        Each dict contains: ``id``, ``name``, ``category``, ``latitude``,
        ``longitude``, ``cluster_id``, ``cluster_hub_id``.
        """
        return list(self._places)

    def place_ids(self) -> list[str]:
        """All place IDs in insertion order."""
        return [p["id"] for p in self._places]

    def clusters(self) -> list[dict[str, Any]]:
        """All clusters.

        Each dict contains: ``cluster_id``, ``member_place_ids``,
        ``hub_place_id``, ``centroid_latitude``, ``centroid_longitude``.
        """
        return list(self._clusters)

    def is_same_cluster(self, place_a: str, place_b: str) -> bool:
        """True when both places are in the same cluster (walkable)."""
        if place_a == place_b:
            return True
        info_a = self._place_id_to_info.get(place_a, {})
        info_b = self._place_id_to_info.get(place_b, {})
        return info_a.get("cluster_id") == info_b.get("cluster_id")

    def hub_leg(
        self,
        from_hub_id: str,
        to_hub_id: str,
        mode: TravelModeLiteral,
    ) -> dict | None:
        """Return the raw hub-to-hub matrix row for a given mode, or None.

        Use this to inspect raw Google Route Matrix data directly.
        """
        return self._hub_leg_map.get((from_hub_id, to_hub_id, mode))

    # ------------------------------------------------------------------
    # NEW: options() — same as get() but returns RouteOptions container
    # ------------------------------------------------------------------

    def options(self, from_place_id: str, to_place_id: str) -> RouteOptions:
        """Same as ``get()`` but returns a RouteOptions container with helpers.

        Parameters
        ----------
        from_place_id:
            Origin place ID.
        to_place_id:
            Destination place ID.

        Returns
        -------
        RouteOptions
            Container wrapping the same list that ``get()`` returns,
            with additional helpers: ``best()``, ``filter()``, ``modes()``,
            ``table()``, and ``choose()``.

        Examples
        --------
        >>> lu = load_routing_lookup(graph)
        >>> ids = lu.place_ids()
        >>> opts = lu.options(ids[0], ids[1])
        >>> print(opts.table())
        >>> print(opts.best().travel_mode)
        >>> print(opts.choose("fastest"))
        """
        return RouteOptions(
            options=self.get(from_place_id, to_place_id),
            from_id=from_place_id,
            to_id=to_place_id,
        )

    # ------------------------------------------------------------------
    # NEW: matrix() — build an all-to-all routing matrix from this graph
    # ------------------------------------------------------------------

    def matrix(self, *, all_modes: bool = False) -> RoutingMatrix:
        """Build an all-to-all routing matrix from this graph.

        Parameters
        ----------
        all_modes:
            False (default) → each cell shows only the best (fastest) mode.
            True → each cell shows all available modes.

        Returns
        -------
        RoutingMatrix
            Pre-built matrix for fast M×M lookups. Reusing the same
            RoutingMatrix across many calls is more efficient than
            calling ``options()`` or ``get()`` repeatedly.

        Examples
        --------
        >>> lu = load_routing_lookup(graph)
        >>> mat = lu.matrix()
        >>> print(mat.as_table())
        >>> mat_all = lu.matrix(all_modes=True)
        >>> print(mat_all.duration_matrix())
        """
        return RoutingMatrix(self._data, all_modes=all_modes)

    # ------------------------------------------------------------------
    # NEW: compare() — one-line summary of all options for a pair
    # ------------------------------------------------------------------

    def compare(self, from_id: str, to_id: str) -> str:
        """One-line summary of all options for a pair.

        Example output:
        ``stop_0 → stop_5: WALK 200m 2min, BICYCLE 1km 4min, TRANSIT 1km 12min, DRIVE 2km 3min``

        Parameters
        ----------
        from_id:
            Origin place ID.
        to_id:
            Destination place ID.

        Returns
        -------
        str
            One-line summary, or ``"from_id → to_id: (no route)"`` if no options.
        """
        opts = self.get(from_id, to_id)
        if not opts:
            return f"{from_id} → {to_id}: (no route)"

        parts = []
        for o in opts:
            dist_str = (
                f"{o.distance_meters:.0f}m"
                if o.distance_meters < 1000
                else f"{o.distance_km:.1f}km"
            )
            dur_str = f"{o.duration_minutes:.0f}min"
            parts.append(f"{o.travel_mode} {dist_str} {dur_str}")

        return f"{from_id} → {to_id}: {', '.join(parts)}"

    # ------------------------------------------------------------------
    # NEW: trip_summary() — multi-stop leg-by-leg breakdown
    # ------------------------------------------------------------------

    def trip_summary(self, place_ids: list[str]) -> dict:
        """Summarize a multi-stop trip leg by leg.

        Parameters
        ----------
        place_ids:
            Ordered list of place IDs forming the trip (e.g. ``[a, b, c]``
            produces legs a→b and b→c).

        Returns
        -------
        dict
            With keys:
            - ``legs``: list of dicts, each with ``from``, ``to``, ``best_mode``,
              ``distance_m``, ``duration_s``.
            - ``total_distance_m``: sum of all leg distances.
            - ``total_duration_s``: sum of all leg durations.

        Examples
        --------
        >>> lu = load_routing_lookup(graph)
        >>> result = lu.trip_summary(ids[:3])
        >>> for leg in result["legs"]:
        ...     print(leg["from"], "→", leg["to"], leg["best_mode"])
        >>> print(f"Total: {result['total_distance_m']:.0f}m, {result['total_duration_s']:.0f}s")
        """
        legs = []
        total_dist = 0.0
        total_dur = 0.0

        for i in range(len(place_ids) - 1):
            from_id = place_ids[i]
            to_id = place_ids[i + 1]
            opts = self.get(from_id, to_id)
            if opts:
                best = min(opts, key=lambda o: o.duration_seconds)
                dist = best.distance_meters
                dur = best.duration_seconds
                mode = best.travel_mode
            else:
                dist = 0.0
                dur = 0.0
                mode = "?"
            legs.append(
                {
                    "from": from_id,
                    "to": to_id,
                    "best_mode": mode,
                    "distance_m": dist,
                    "duration_s": dur,
                }
            )
            total_dist += dist
            total_dur += dur

        return {
            "legs": legs,
            "total_distance_m": total_dist,
            "total_duration_s": total_dur,
        }

    # ------------------------------------------------------------------
    # Core lookup
    # ------------------------------------------------------------------

    def get(
        self,
        from_place_id: str,
        to_place_id: str,
        travel_mode: TravelModeLiteral | str | None = None,
    ) -> list[RouteOption]:
        """All travel options from ``from_place_id`` to ``to_place_id``.

        Returns a **list of all available modes** (not just the best one).
        Optionally filter to a specific ``travel_mode``.

        Parameters
        ----------
        from_place_id:
            Origin place ID. Must exist in the graph.
        to_place_id:
            Destination place ID. Must exist in the graph.
        travel_mode:
            Optional filter. Pass ``"WALK"``, ``"BICYCLE"``, ``"TRANSIT"``,
            ``"DRIVE"``, or ``"COMPOSED"`` to return only matching options.
            When omitted, all available modes are returned.

        Returns
        -------
        list[RouteOption]
            One entry per available travel mode (may be 1–4 depending on
            cluster configuration).  Empty list if either place ID is unknown
            or no mode matches the filter.

        Examples
        --------
        >>> lu = load_routing_lookup("/path/to/graph.json")
        >>> # Get all options for a cross-cluster trip
        >>> for opt in lu.get("stop_0", "stop_5"):
        ...     print(opt.travel_mode, opt.distance_km, opt.duration_minutes)
        >>> # Transit-only options
        >>> for opt in lu.get("stop_0", "stop_5", travel_mode="TRANSIT"):
        ...     print(opt.travel_mode, opt.duration_minutes)
        """
        # Unknown place → empty list
        if from_place_id not in self._place_id_to_info:
            return []
        if to_place_id not in self._place_id_to_info:
            return []

        # Same place: only WALK at zero cost
        if from_place_id == to_place_id:
            opt = RouteOption(
                from_place_id=from_place_id,
                to_place_id=to_place_id,
                travel_mode="WALK",
                distance_meters=0.0,
                duration_seconds=0.0,
                quality="haversine_walk",
                primary_mode=None,
                is_same_cluster=True,
                legs=(),
            )
            return [opt] if self._mode_matches("WALK", travel_mode) else []

        from_info = self._place_id_to_info[from_place_id]
        to_info = self._place_id_to_info[to_place_id]
        same_cluster = from_info["cluster_id"] == to_info["cluster_id"]

        options: list[RouteOption] = []

        if same_cluster:
            # Same cluster → only WALK available (haversine walk approximation)
            edge = self._edge_map.get((from_place_id, to_place_id))
            if edge:
                options.append(
                    self._edge_to_option(
                        from_place_id, to_place_id, edge, same_cluster=True
                    )
                )
        else:
            # Cross cluster → WALK (via composed edge) + all hub-matrix modes
            # The composed edge already captures the "best" hub mode selection
            edge = self._edge_map.get((from_place_id, to_place_id))
            if edge:
                options.append(
                    self._edge_to_option(
                        from_place_id, to_place_id, edge, same_cluster=False
                    )
                )

            # Additionally expose every hub-matrix mode directly
            from_hub = from_info["cluster_hub_id"]
            to_hub = to_info["cluster_hub_id"]

            for mode in ("BICYCLE", "TRANSIT", "DRIVE"):
                hub_leg = self._hub_leg_map.get((from_hub, to_hub, mode))
                if hub_leg is None:
                    continue

                # Reconstruct the composed option with this specific hub mode
                distance_meters = hub_leg.get("distance_meters") or 0
                duration_seconds = hub_leg.get("duration_seconds") or 0
                # Walk legs not stored in hub_leg — use edge's walk estimates as proxy
                walk_d = 0.0
                walk_t = 0.0
                if edge:
                    walk_d = next(
                        (
                            leg["distance_meters"]
                            for leg in edge.get("legs", [])
                            if leg.get("kind") == "walk_approx"
                        ),
                        0.0,
                    )
                    walk_t = next(
                        (
                            leg["duration_seconds"]
                            for leg in edge.get("legs", [])
                            if leg.get("kind") == "walk_approx"
                        ),
                        0.0,
                    )
                    distance_meters += 2 * (walk_d or 0)
                    duration_seconds += 2 * (walk_t or 0)

                options.append(
                    RouteOption(
                        from_place_id=from_place_id,
                        to_place_id=to_place_id,
                        travel_mode=mode,
                        distance_meters=float(distance_meters),
                        duration_seconds=float(duration_seconds),
                        quality="hub_chain",
                        primary_mode=mode,
                        is_same_cluster=False,
                        legs=(
                            {
                                "kind": "walk_approx",
                                "travel_mode": "WALK",
                                "distance_meters": walk_d,
                                "duration_seconds": walk_t,
                            },
                            {
                                "kind": "matrix",
                                "travel_mode": mode,
                                "distance_meters": hub_leg.get("distance_meters"),
                                "duration_seconds": hub_leg.get("duration_seconds"),
                            },
                            {
                                "kind": "walk_approx",
                                "travel_mode": "WALK",
                                "distance_meters": walk_d,
                                "duration_seconds": walk_t,
                            },
                        )
                        if edge
                        else (),
                    )
                )

        # Apply travel_mode filter
        if travel_mode is not None:
            options = [
                o for o in options if self._mode_matches(o.travel_mode, travel_mode)
            ]

        return options

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mode_matches(option_mode: str, filter_mode: str | None) -> bool:
        if filter_mode is None:
            return True
        return option_mode == filter_mode

    def _edge_to_option(
        self,
        from_place_id: str,
        to_place_id: str,
        edge: dict[str, Any],
        same_cluster: bool,
    ) -> RouteOption:
        travel_mode = str(edge.get("travel_mode_effective", "WALK"))
        quality = str(edge.get("quality", "fallback_estimate"))
        primary_mode: str | None = None

        if not same_cluster:
            primary_mode = str(edge.get("primary_hub_travel_mode")) or None
            if primary_mode and travel_mode == "WALK":
                travel_mode = "COMPOSED"

        legs = tuple(edge.get("legs", []))

        return RouteOption(
            from_place_id=from_place_id,
            to_place_id=to_place_id,
            travel_mode=travel_mode,
            distance_meters=float(edge.get("distance_meters", 0.0)),
            duration_seconds=float(edge.get("duration_seconds", 0.0)),
            quality=quality,
            primary_mode=primary_mode,
            is_same_cluster=same_cluster,
            legs=legs,
        )


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------


def compare(graph: dict | Path | str, from_id: str, to_id: str) -> str:
    """One-liner: human-readable summary of all options for a pair.

    Example output:
    ``stop_0 → stop_5: WALK 200m 2min, BICYCLE 1km 4min, TRANSIT 1km 12min, DRIVE 2km 3min``

    Parameters
    ----------
    graph:
        Graph source (dict, Path, or JSON string).
    from_id:
        Origin place ID.
    to_id:
        Destination place ID.

    Returns
    -------
    str
        One-line summary, or ``"from_id → to_id: (no route)"`` if no options.
    """
    return RoutingLookup(graph).compare(from_id, to_id)


def trip_summary(graph: dict | Path | str, stops: list[str]) -> dict:
    """One-liner for multi-stop trips. Returns leg-by-leg breakdown.

    Parameters
    ----------
    graph:
        Graph source (dict, Path, or JSON string).
    stops:
        Ordered list of place IDs forming the trip.

    Returns
    -------
    dict
        With keys:
        - ``legs``: list of dicts, each with ``from``, ``to``, ``best_mode``,
          ``distance_m``, ``duration_s``.
        - ``total_distance_m``: sum of all leg distances (float).
        - ``total_duration_s``: sum of all leg durations (float).

    Examples
    --------
    >>> result = trip_summary(graph, ["stop_0", "stop_3", "stop_7"])
    >>> for leg in result["legs"]:
    ...     print(leg["from"], "→", leg["to"], leg["best_mode"], leg["distance_m"])
    """
    return RoutingLookup(graph).trip_summary(stops)


def best_mode(
    graph: dict | Path | str,
    from_id: str,
    to_id: str,
    preference: str = "duration",
) -> RouteOption | None:
    """Quickest / shortest / or mode-specific option for a pair.

    Parameters
    ----------
    graph:
        Graph source (dict, Path, or JSON string).
    from_id:
        Origin place ID.
    to_id:
        Destination place ID.
    preference:
        ``"duration"`` (default) → fastest option.
        ``"distance"`` → shortest option.
        ``"transit"`` / ``"bike"`` / ``"walk"`` / ``"drive"`` → first matching mode.
        ``"cheapest"`` → WALK if available, else fastest.

    Returns
    -------
    RouteOption | None
        Selected option, or None if no match.

    Examples
    --------
    >>> opt = best_mode(graph, ids[0], ids[1])
    >>> print(f"Take {opt.travel_mode}, {opt.duration_minutes:.1f}min")
    >>> opt = best_mode(graph, ids[0], ids[1], preference="walk")
    """
    opts = RoutingLookup(graph).options(from_id, to_id)
    return opts.choose(preference)


def distance_lookup(
    graph: dict | Path | str,
    from_id: str,
    to_id: str,
    travel_mode: str | None = None,
) -> list[RouteOption]:
    """One-liner: load graph and return all A→B route options.

    Equivalent to: ``load_routing_lookup(graph).get(from_id, to_id, travel_mode)``

    Parameters
    ----------
    graph:
        Graph source (dict, Path, or JSON string).
    from_id:
        Origin place ID.
    to_id:
        Destination place ID.
    travel_mode:
        Optional mode filter (``"WALK"``, ``"BICYCLE"``, ``"TRANSIT"``,
        ``"DRIVE"``, ``"COMPOSED"``).

    Returns
    -------
    list[RouteOption]
        Available travel options (may be empty).
    """
    return RoutingLookup(graph).get(from_id, to_id, travel_mode)


def load_routing_lookup(source: dict | Path | str) -> RoutingLookup:
    """Build a :class:`RoutingLookup` from a dict, Path, or JSON string."""
    return RoutingLookup(source)


# ---------------------------------------------------------------------------
# Execution-agent-friendly name-based query helpers
# ---------------------------------------------------------------------------

from travelplanner.routing_lookup.queries import (  # noqa: E402
    PlaceResolutionError,
    closest_to,
    distance_between,
    resolve_place_id,
)
