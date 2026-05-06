"""Execution-agent-friendly query helpers on top of :mod:`travelplanner.routing_lookup`.

Goal: make a retained ``place_distance_graph`` artifact *directly usable* for Phase-2
composition per ``docs/workflow.md``:

- resolve human names (\"Hotel\", \"Rijksmuseum\") → stable place ids (\"stop_3\")
- answer deterministic queries like:
  - distance from X to Y
  - closest of {A,B,C} to target

No LLM. No Google API. Pure lookups over the existing routing artifact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from travelplanner.routing_lookup import RouteOption

PreferenceLiteral = Literal[
    "duration",
    "distance",
    "fastest",
    "shortest",
    "walk",
    "bike",
    "bicycle",
    "transit",
    "drive",
    "cheapest",
]


def _norm(s: str) -> str:
    s2 = s.strip().lower()
    s2 = re.sub(r"[^a-z0-9]+", " ", s2)
    return re.sub(r"\s+", " ", s2).strip()


def _token_score(a: str, b: str) -> float:
    """Lightweight similarity score in [0,1]."""
    na = _norm(a)
    nb = _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    # Common user behavior: short substring of a label ("rijks" → "rijksmuseum").
    if na in nb:
        # If it matches a full token prefix, boost further.
        tokens = nb.split()
        if any(t.startswith(na) for t in tokens):
            return 0.95
        return 0.85

    ratio = SequenceMatcher(a=na, b=nb).ratio()
    ta = set(na.split())
    tb = set(nb.split())
    jacc = (len(ta & tb) / len(ta | tb)) if (ta and tb) else 0.0
    # Favor SequenceMatcher but keep token overlap for multi-word labels.
    return 0.7 * ratio + 0.3 * jacc


@dataclass(frozen=True, slots=True)
class PlaceCandidate:
    place_id: str
    name: str
    score: float


@dataclass(frozen=True, slots=True)
class ResolvedPlace:
    query: str
    matched: PlaceCandidate
    alternatives: tuple[PlaceCandidate, ...] = ()


class PlaceResolutionError(ValueError):
    """Raised when a name cannot be resolved deterministically."""

    def __init__(
        self,
        message: str,
        *,
        query: str,
        candidates: list[PlaceCandidate] | None = None,
    ) -> None:
        super().__init__(message)
        self.query = query
        self.candidates = candidates or []


def resolve_place_id(
    graph: dict | Any,
    query: str,
    *,
    min_score: float = 0.62,
    ambiguity_delta: float = 0.06,
    max_alternatives: int = 5,
) -> ResolvedPlace:
    """Resolve a human label/name to a place id using fuzzy matching.

    Parameters
    ----------
    graph:
        place_distance_graph content dict, a Path-like string, or any graph source accepted by
        :func:`travelplanner.routing_lookup.load_routing_lookup`.
    query:
        Human reference: name/label/id (\"Hotel\", \"Rijksmuseum\", \"stop_3\").
    min_score:
        Minimum score to accept a match.
    ambiguity_delta:
        If the 2nd best score is within this delta of the best score, treat as ambiguous.
    max_alternatives:
        How many candidates to include in the error payload / response.
    """
    q = query.strip()
    if not q:
        raise PlaceResolutionError("empty query", query=query)

    from travelplanner.routing_lookup import load_routing_lookup

    lu = load_routing_lookup(graph)

    # If the user already provided a valid place id, accept it directly.
    for pid in lu.place_ids():
        if pid == q:
            info = next((p for p in lu.places() if p.get("id") == pid), None) or {}
            name = str(info.get("name") or info.get("label") or pid)
            cand = PlaceCandidate(place_id=pid, name=name, score=1.0)
            return ResolvedPlace(query=query, matched=cand, alternatives=())

    scored: list[PlaceCandidate] = []
    for p in lu.places():
        pid = str(p.get("id") or "")
        if not pid:
            continue
        label = str(p.get("name") or p.get("label") or pid)
        s = _token_score(q, label)
        scored.append(PlaceCandidate(place_id=pid, name=label, score=s))

    scored.sort(key=lambda c: (-c.score, c.name, c.place_id))
    top = scored[0] if scored else None
    if top is None or top.score < min_score:
        raise PlaceResolutionError(
            f"could not resolve place name {query!r}",
            query=query,
            candidates=scored[:max_alternatives],
        )

    second = scored[1] if len(scored) > 1 else None
    if second is not None and (top.score - second.score) <= ambiguity_delta:
        raise PlaceResolutionError(
            f"ambiguous place name {query!r}",
            query=query,
            candidates=scored[:max_alternatives],
        )

    return ResolvedPlace(
        query=query,
        matched=top,
        alternatives=tuple(scored[1:max_alternatives]),
    )


@dataclass(frozen=True, slots=True)
class DistanceResult:
    from_query: str
    to_query: str
    from_place: ResolvedPlace
    to_place: ResolvedPlace
    option: RouteOption
    explanation: str


def _pref_to_best_mode_preference(pref: PreferenceLiteral) -> str:
    # best_mode() implements these strings already.
    if pref in ("fastest",):
        return "duration"
    if pref in ("shortest",):
        return "distance"
    return str(pref)


def distance_between(
    graph: dict | Any,
    from_name: str,
    to_name: str,
    *,
    preference: PreferenceLiteral = "duration",
) -> DistanceResult:
    """Resolve place names and return a single chosen RouteOption + explanation."""
    from travelplanner.routing_lookup import best_mode

    a = resolve_place_id(graph, from_name)
    b = resolve_place_id(graph, to_name)
    pref = _pref_to_best_mode_preference(preference)
    opt = best_mode(graph, a.matched.place_id, b.matched.place_id, preference=pref)
    if opt is None:
        raise ValueError(
            f"no route option for {from_name!r} → {to_name!r} "
            f"({a.matched.place_id}→{b.matched.place_id})"
        )
    expl = opt.quality
    if opt.travel_mode == "COMPOSED" and opt.primary_mode:
        expl = f"{opt.quality} (primary={opt.primary_mode})"
    return DistanceResult(
        from_query=from_name,
        to_query=to_name,
        from_place=a,
        to_place=b,
        option=opt,
        explanation=expl,
    )


@dataclass(frozen=True, slots=True)
class ClosestItem:
    candidate_query: str
    candidate: ResolvedPlace
    option: RouteOption


@dataclass(frozen=True, slots=True)
class ClosestResult:
    target_query: str
    target: ResolvedPlace
    ranked: tuple[ClosestItem, ...]

    @property
    def winner(self) -> ClosestItem:
        return self.ranked[0]


def closest_to(
    graph: dict | Any,
    target_name: str,
    candidate_names: list[str],
    *,
    preference: PreferenceLiteral = "duration",
) -> ClosestResult:
    """Rank candidates by distance/duration to a target, using resolved names."""
    if not candidate_names:
        raise ValueError("candidate_names is empty")
    from travelplanner.routing_lookup import best_mode

    target = resolve_place_id(graph, target_name)
    pref = _pref_to_best_mode_preference(preference)

    items: list[ClosestItem] = []
    for cand_query in candidate_names:
        cand = resolve_place_id(graph, cand_query)
        opt = best_mode(graph, cand.matched.place_id, target.matched.place_id, preference=pref)
        if opt is None:
            continue
        items.append(
            ClosestItem(candidate_query=cand_query, candidate=cand, option=opt)
        )

    if not items:
        raise ValueError("no candidates produced a route option")

    if pref == "distance":
        items.sort(key=lambda it: (it.option.distance_meters, it.candidate.matched.name))
    else:
        items.sort(key=lambda it: (it.option.duration_seconds, it.candidate.matched.name))

    return ClosestResult(
        target_query=target_name,
        target=target,
        ranked=tuple(items),
    )


__all__ = [
    "PlaceCandidate",
    "ResolvedPlace",
    "PlaceResolutionError",
    "DistanceResult",
    "ClosestItem",
    "ClosestResult",
    "resolve_place_id",
    "distance_between",
    "closest_to",
]

