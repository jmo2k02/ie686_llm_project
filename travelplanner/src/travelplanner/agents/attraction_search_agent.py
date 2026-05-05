from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openai
import requests
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

from travelplanner.schema.attraction_search_artifact import (
    AttractionArtifactContentModel,
    AttractionCandidateModel,
    AttractionItemModel,
    AttractionParamsModel,
    AttractionSearchErrorModel,
    CandidateSelectionModel,
    GeneratedActivityModel,
)
from travelplanner.schema.system_state import (
    AgentArtifactModel,
    MessageHistoryModel,
    TaskModel,
)
from travelplanner.utils.llm import invoke_structured_model
from travelplanner.config import get_setting

_SEARCH_URL = get_setting("agents.attraction_search.search_url")
_EXPERIENCE_POOL_PATH: Path = Path(__file__).parent / get_setting("agents.attraction_search.experience_pool_path")
_EXPERIENCE_POOL: list[dict[str, Any]] = json.loads(_EXPERIENCE_POOL_PATH.read_text(encoding="utf-8"))

_DEFAULT_EMBEDDING_MODEL = get_setting("agents.attraction_search.default_embedding_model")
_DEFAULT_ANSWER_MODEL = get_setting("agents.attraction_search.default_answer_model")
_DEFAULT_MAX_CANDIDATES = get_setting("agents.attraction_search.default_max_candidates")
_DEFAULT_TOP_REVIEW_CANDIDATES = get_setting("agents.attraction_search.default_top_review_candidates")
_DEFAULT_TIMEOUT_SECONDS = get_setting("agents.attraction_search.default_timeout_seconds")

_PARAM_EXTRACTION_SYSTEM_PROMPT = """\
You are an attraction search parameter extractor for a travel planning assistant.

Given a structured task description, extract the search parameters.

Rules:
- budget is a float representing the per-activity budget in EUR
- destination is the city or region name
- traveller_profile is the full free-text description of the traveller's style and interests
- day is the trip day number (integer, defaults to 1 if not specified)
- previous_activities is a summary of activities done on prior days (empty string if none)
- orchestrator_hint is an optional hint about what type of activity is needed (null if absent)
- Return JSON only — no extra text.
"""

_GENERATION_SYSTEM_PROMPT = """\
You are an experience curator for a travel planning assistant generating deeply engaging, culturally immersive activity experiences.

Rules:
- Activities only — no food, no restaurants, no transport
- One activity covering a half-day slot (morning, afternoon, or evening)
- The activity must be realistic given typical opening hours for that type of venue
- Each activity must name the specific type of local person or community the traveller will interact with, and explain why this interaction falls outside the tourist bubble
- Consider previous activities to avoid repetition and ensure variety across days
- Output strict JSON only — no markdown fences, no preamble

Output format: a single JSON object:
{
  "day": int,
  "time_slot": "morning" | "afternoon" | "evening",
  "title": "string (max 8 words, evocative)",
  "description": "string (3-5 sentences, destination-specific and textured)",
  "local_touchpoint": "string (1-2 sentences: who the traveller meets and why it is not tourist-facing)",
  "search_keywords": ["keyword1", "keyword2"],
  "estimated_duration_hours": float,
  "has_specific_location": bool
}

The following are examples of the expected style — narrative specificity, local touchpoint depth, cultural engagement register. Generate an activity in this exact register for the actual destination:

{few_shot_examples}
"""

_SELECTION_SYSTEM_PROMPT = """\
You are a travel experience selector. Choose the venue that best fits the traveller profile.
Prefer locally-embedded venues over tourist-facing operators, even if the tourist operator has a higher rating.
Secondly, put more weight on recent reviews than old reviews, and prefer
venues with many reviews rather than few reviews even if the rating is slightly lower.
Make an estimation of how much it will cost based on the price level and the traveller's budget, 
but do not exclude venues that are above the traveller's stated budget as long as they have other strong signals.
Return JSON only: {"selected_index": int, "selection_reason": "one sentence"}
"""

# Module-level embedding cache — populated on first invocation, not at import time
_EMBEDDING_CACHE: dict[str, Any] | None = None


@dataclass(frozen=True)
class AttractionSearchConfig:
    openai_api_key: str = ""
    serpapi_api_key: str = ""
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL
    answer_model_name: str = _DEFAULT_ANSWER_MODEL
    temperature: float = 0.0
    max_candidates: int = _DEFAULT_MAX_CANDIDATES
    top_review_candidates: int = _DEFAULT_TOP_REVIEW_CANDIDATES
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS


def load_config_from_env() -> AttractionSearchConfig:
    return AttractionSearchConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        serpapi_api_key=os.getenv("SERPAPI_API_KEY", "").strip(),
        embedding_model=os.getenv("TRAVELPLANNER_ATTRACTION_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL).strip(),
        answer_model_name=os.getenv("TRAVELPLANNER_ATTRACTION_MODEL", _DEFAULT_ANSWER_MODEL).strip(),
        temperature=float(os.getenv("TRAVELPLANNER_ATTRACTION_TEMPERATURE", "0.0")),
        max_candidates=int(os.getenv("TRAVELPLANNER_ATTRACTION_MAX_CANDIDATES", str(_DEFAULT_MAX_CANDIDATES))),
        top_review_candidates=int(os.getenv("TRAVELPLANNER_ATTRACTION_TOP_REVIEW_CANDIDATES", str(_DEFAULT_TOP_REVIEW_CANDIDATES))),
        timeout_seconds=int(os.getenv("TRAVELPLANNER_ATTRACTION_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS))),
    )


class AttractionSearchAgentState(BaseModel):
    query: str
    model_name: str
    temperature: float = 0.0
    task_list: list[TaskModel] = Field(default_factory=list)
    agent_artifacts: dict[str, list[AgentArtifactModel]] = Field(default_factory=dict)
    message_history: MessageHistoryModel | None = None


# ── Param extraction ──────────────────────────────────────────────────────────

def _build_param_extraction_prompt(task_text: str) -> str:
    return "\n".join([
        "Extract attraction search parameters from the following task.",
        "",
        f"Task: {task_text.strip()}",
        "",
        "Return strictly valid JSON with this shape:",
        '{"budget": 80.0, "destination": "city name", "traveller_profile": "free text", "day": 1, "previous_activities": "summary or empty string", "orchestrator_hint": "string or null"}',
    ])


def _extract_attraction_params(
    task_text: str,
    model_name: str,
    temperature: float,
) -> AttractionParamsModel:
    user_prompt = _build_param_extraction_prompt(task_text)
    structured_output, _, _ = invoke_structured_model(
        model_name=model_name,
        temperature=temperature,
        system_prompt=_PARAM_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=AttractionParamsModel,
    )
    return structured_output


# ── Archetype / embedding helpers ─────────────────────────────────────────────

def _serialize_profile(profile: dict[str, Any]) -> str:
    interests = ", ".join(profile.get("interests", []))
    return (
        f"{profile['travel_style']}, {profile['party_type']}, {profile['pace']} pace, "
        f"{profile['budget']} budget, interested in {interests}, "
        f"engagement depth: {profile['engagement_depth']}"
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _ensure_embeddings(config: AttractionSearchConfig) -> tuple[list[np.ndarray], list[str]]:
    global _EMBEDDING_CACHE
    if _EMBEDDING_CACHE is not None:
        return _EMBEDDING_CACHE["embeddings"], _EMBEDDING_CACHE["names"]

    client = openai.OpenAI(api_key=config.openai_api_key)
    names: list[str] = []
    texts: list[str] = []
    for archetype in _EXPERIENCE_POOL:
        names.append(archetype["archetype"])
        texts.append(_serialize_profile(archetype["profile"]))

    resp = client.embeddings.create(model=config.embedding_model, input=texts)
    embeddings = [np.array(item.embedding) for item in resp.data]
    _EMBEDDING_CACHE = {"embeddings": embeddings, "names": names}
    return embeddings, names


def _select_archetype(traveller_profile: str, config: AttractionSearchConfig) -> tuple[str, dict[str, Any]]:
    embeddings, names = _ensure_embeddings(config)
    client = openai.OpenAI(api_key=config.openai_api_key)
    resp = client.embeddings.create(model=config.embedding_model, input=[traveller_profile])
    query_emb = np.array(resp.data[0].embedding)

    scores = [_cosine_similarity(query_emb, emb) for emb in embeddings]
    best_idx = int(np.argmax(scores))
    best_name = names[best_idx]
    best_archetype = next(a for a in _EXPERIENCE_POOL if a["archetype"] == best_name)
    return best_name, best_archetype


# ── Experience generation ─────────────────────────────────────────────────────

def _build_generation_system_prompt(archetype: dict[str, Any]) -> str:
    few_shot = json.dumps(archetype["experiences"], indent=2, ensure_ascii=False)
    # Use replace instead of .format() — the prompt template contains JSON schema
    # braces that would be misinterpreted as str.format() placeholders.
    return _GENERATION_SYSTEM_PROMPT.replace("{few_shot_examples}", few_shot)


def _build_generation_user_prompt(params: AttractionParamsModel) -> str:
    lines = [
        f"Destination: {params.destination}",
        f"Day: {params.day}",
        f"Budget: {params.budget:.0f} EUR per activity",
        f"Traveller profile: {params.traveller_profile}",
        f"Previous activities: {params.previous_activities or 'None — this is the first activity'}",
    ]
    if params.orchestrator_hint:
        lines.append(f"Orchestrator hint: {params.orchestrator_hint}")
    lines.append(f"\nGenerate exactly one activity for day {params.day} in {params.destination} as a JSON object.")
    return "\n".join(lines)


def _generate_activity(
    params: AttractionParamsModel,
    archetype: dict[str, Any],
    model_name: str,
    temperature: float,
) -> GeneratedActivityModel:
    system_prompt = _build_generation_system_prompt(archetype)
    user_prompt = _build_generation_user_prompt(params)

    try:
        result, _, _ = invoke_structured_model(
            model_name=model_name,
            temperature=temperature,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=GeneratedActivityModel,
        )
        return result
    except (json.JSONDecodeError, ValidationError):
        retry_prompt = user_prompt + "\n\nThe previous response was not valid JSON. Return ONLY a valid JSON object, no other text."
        result, _, _ = invoke_structured_model(
            model_name=model_name,
            temperature=temperature,
            system_prompt=system_prompt,
            user_prompt=retry_prompt,
            response_model=GeneratedActivityModel,
        )
        return result


# ── SERPAPI helpers ───────────────────────────────────────────────────────────

def _search_candidates(
    keyword: str,
    destination: str,
    config: AttractionSearchConfig,
) -> list[AttractionCandidateModel]:
    if not config.serpapi_api_key:
        return []
    try:
        response = requests.get(
            _SEARCH_URL,
            params={
                "engine": "google_maps",
                "q": f"{keyword} {destination}",
                "type": "search",
                "api_key": config.serpapi_api_key,
            },
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.HTTPError, requests.Timeout, requests.ConnectionError):
        return []

    candidates = []
    for raw in data.get("local_results", [])[: config.max_candidates]:
        gps = raw.get("gps_coordinates")
        candidates.append(
            AttractionCandidateModel(
                title=raw.get("title", ""),
                address=raw.get("address"),
                gps_coordinates={"lat": gps["latitude"], "lng": gps["longitude"]} if gps else None,
                rating=raw.get("rating"),
                reviews=raw.get("reviews"),
                price=raw.get("price"),
                type=raw.get("type"),
                data_id=raw.get("data_id") or raw.get("place_id"),
                hours=raw.get("hours"),
            )
        )
    return candidates


def _fetch_reviews(
    candidate: AttractionCandidateModel,
    config: AttractionSearchConfig,
) -> list[str]:
    if not candidate.data_id:
        return []
    try:
        response = requests.get(
            _SEARCH_URL,
            params={
                "engine": "google_maps_reviews",
                "data_id": candidate.data_id,
                "hl": "en",
                "sort_by": "ratingLow",
                "api_key": config.serpapi_api_key,
            },
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return [r["snippet"] for r in data.get("reviews", [])[:3] if r.get("snippet")]
    except Exception:
        return []


def _find_candidates(
    activity: GeneratedActivityModel,
    destination: str,
    config: AttractionSearchConfig,
) -> list[AttractionCandidateModel]:
    candidates: list[AttractionCandidateModel] = []
    for keyword in activity.search_keywords:
        if len(candidates) >= 2:
            break
        candidates.extend(_search_candidates(keyword, destination, config))

    if not candidates:
        return []

    sorted_candidates = sorted(
        candidates, key=lambda c: c.rating or 0.0, reverse=True
    )
    for i, candidate in enumerate(sorted_candidates[: config.top_review_candidates]):
        snippets = _fetch_reviews(candidate, config)
        sorted_candidates[i] = candidate.model_copy(update={"review_snippets": snippets})

    return sorted_candidates


def _format_candidate_line(i: int, c: AttractionCandidateModel) -> str:
    line = (
        f"[{i}] {c.title} | {c.address or 'address unknown'} | "
        f"Rating: {c.rating or 'N/A'} | Reviews: {c.reviews or 'N/A'} | "
        f"Price: {c.price or 'N/A'} | Type: {c.type or 'N/A'}"
    )
    if c.review_snippets:
        snippets = " / ".join(f"'{s}'" for s in c.review_snippets)
        line += f"\n   Review snippets: {snippets}"
    else:
        line += "\n   Review snippets: (not available)"
    return line


def _select_candidate(
    activity: GeneratedActivityModel,
    candidates: list[AttractionCandidateModel],
    traveller_profile: str,
    model_name: str,
    temperature: float,
) -> tuple[AttractionCandidateModel, str]:
    candidate_block = "\n".join(
        _format_candidate_line(i, c) for i, c in enumerate(candidates)
    )
    user_prompt = (
        f"Activity: {activity.title}\n"
        f"{activity.description}\n\n"
        f"Traveller profile: {traveller_profile}\n\n"
        f"Candidates:\n{candidate_block}\n\n"
        "Selection criteria:\n"
        "- POSITIVE signals in reviews: 'locals', 'regulars', 'community', 'authentic'\n"
        "- NEGATIVE signals: 'tourists', 'overpriced', 'crowded', 'commercial'\n\n"
        'Return JSON only: {"selected_index": int, "selection_reason": "one sentence"}'
    )
    try:
        result, _, _ = invoke_structured_model(
            model_name=model_name,
            temperature=temperature,
            system_prompt=_SELECTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=CandidateSelectionModel,
        )
        idx = max(0, min(result.selected_index, len(candidates) - 1))
        return candidates[idx], result.selection_reason
    except Exception:
        return candidates[0], "Fallback to highest-rated candidate."


# ── Item construction ─────────────────────────────────────────────────────────

def _budget_to_price_symbol(budget: float) -> str:
    if budget < 30:
        return "$"
    elif budget < 80:
        return "$$"
    return "$$$"


def _build_item(
    activity: GeneratedActivityModel,
    selected: AttractionCandidateModel | None,
    selection_reason: str | None,
    destination: str,
    budget: float,
    archetype_name: str,
) -> AttractionItemModel:
    price_symbol = _budget_to_price_symbol(budget)
    if selected is not None:
        return AttractionItemModel(
            day=activity.day,
            time_slot=activity.time_slot,
            title=activity.title,
            description=activity.description,
            local_touchpoint=activity.local_touchpoint,
            estimated_duration_hours=activity.estimated_duration_hours,
            has_specific_location=activity.has_specific_location,
            location_name=selected.title,
            location_address=selected.address,
            coordinates=selected.gps_coordinates,
            place_id=selected.data_id,
            place_rating=selected.rating,
            place_review_count=selected.reviews,
            place_price_level=selected.price,
            place_type=selected.type,
            place_hours=selected.hours,
            selection_reason=selection_reason,
            place_found=True,
            estimated_price_range=selected.price or price_symbol,
            selected_archetype=archetype_name,
            provenance="LLM activity | SERPAPI google_maps",
        )
    else:
        return AttractionItemModel(
            day=activity.day,
            time_slot=activity.time_slot,
            title=activity.title,
            description=activity.description,
            local_touchpoint=activity.local_touchpoint,
            estimated_duration_hours=activity.estimated_duration_hours,
            has_specific_location=activity.has_specific_location,
            location_name=destination,
            place_found=False,
            estimated_price_range=price_symbol,
            selected_archetype=archetype_name,
            provenance="LLM activity | no place found",
        )


def _normalize_error(code: str, message: str) -> AttractionSearchErrorModel:
    valid = {"missing_api_key", "http_error", "timeout_error", "parse_error", "llm_error", "unknown_error"}
    return AttractionSearchErrorModel(
        code=code if code in valid else "unknown_error",  # type: ignore[arg-type]
        message=message,
    )


def _compute_status(item: AttractionItemModel | None, errors: list[AttractionSearchErrorModel]) -> str:
    if item is None:
        return "failed"
    if item.has_specific_location and not item.place_found:
        return "partial"
    if errors:
        return "partial"
    return "success"


# ── Public run function ───────────────────────────────────────────────────────

def run_attraction_search(
    params: AttractionParamsModel,
    model_name: str,
    temperature: float,
    config: AttractionSearchConfig,
    task_ref: str = "scratch",
) -> AttractionArtifactContentModel:
    errors: list[AttractionSearchErrorModel] = []

    if not config.openai_api_key:
        return AttractionArtifactContentModel(
            task_ref=task_ref,
            status="failed",
            provider="openai_embeddings+llm+serpapi_google_maps",
            destination=params.destination,
            budget=params.budget,
            selected_archetype="",
            errors=[_normalize_error("missing_api_key", "OPENAI_API_KEY is not set")],
            config={"openai_api_key_set": False, "serpapi_api_key_set": bool(config.serpapi_api_key)},
        )

    # Step 1: Select archetype
    try:
        archetype_name, archetype = _select_archetype(params.traveller_profile, config)
    except Exception as exc:
        return AttractionArtifactContentModel(
            task_ref=task_ref,
            status="failed",
            provider="openai_embeddings+llm+serpapi_google_maps",
            destination=params.destination,
            budget=params.budget,
            selected_archetype="",
            errors=[_normalize_error("llm_error", f"Archetype embedding failed: {exc}")],
            config={"openai_api_key_set": bool(config.openai_api_key)},
        )

    # Step 2: Generate one activity
    try:
        activity = _generate_activity(params, archetype, model_name, temperature)
    except Exception as exc:
        return AttractionArtifactContentModel(
            task_ref=task_ref,
            status="failed",
            provider="openai_embeddings+llm+serpapi_google_maps",
            destination=params.destination,
            budget=params.budget,
            selected_archetype=archetype_name,
            errors=[_normalize_error("llm_error", f"Activity generation failed: {exc}")],
            config={"openai_api_key_set": bool(config.openai_api_key)},
        )

    # Step 3: Resolve to a place via SERPAPI
    top_candidates: list[AttractionCandidateModel] = []

    if not config.serpapi_api_key:
        errors.append(_normalize_error("missing_api_key", "SERPAPI_API_KEY is not set — place search skipped"))

    if activity.has_specific_location and config.serpapi_api_key:
        try:
            top_candidates = _find_candidates(activity, params.destination, config)
        except Exception as exc:
            errors.append(_normalize_error("http_error", f"Candidate search failed: {exc}"))

    if top_candidates:
        selected, reason = _select_candidate(
            activity,
            top_candidates[: config.top_review_candidates],
            params.traveller_profile,
            model_name,
            temperature,
        )
        item = _build_item(activity, selected, reason, params.destination, params.budget, archetype_name)
    else:
        item = _build_item(activity, None, None, params.destination, params.budget, archetype_name)

    status = _compute_status(item, errors)
    return AttractionArtifactContentModel(
        task_ref=task_ref,
        status=status,  # type: ignore[arg-type]
        provider="openai_embeddings+llm+serpapi_google_maps",
        destination=params.destination,
        budget=params.budget,
        selected_archetype=archetype_name,
        item=item,
        top_candidates=top_candidates[: config.top_review_candidates],
        errors=errors,
        config={
            "openai_api_key_set": bool(config.openai_api_key),
            "serpapi_api_key_set": bool(config.serpapi_api_key),
            "embedding_model": config.embedding_model,
            "answer_model_name": config.answer_model_name,
            "max_candidates": config.max_candidates,
            "timeout_seconds": config.timeout_seconds,
        },
    )


# ── LangGraph node + graph ────────────────────────────────────────────────────

def make_graph():
    def attraction_search(state: AttractionSearchAgentState) -> dict[str, Any]:
        config = load_config_from_env()
        attraction_tasks = [t for t in state.task_list if t.type == "attraction" and t.is_valid]

        if not attraction_tasks:
            existing = state.agent_artifacts.get("attraction_search_agent", [])
            return {"agent_artifacts": {**state.agent_artifacts, "attraction_search_agent": existing}}

        artifacts: list[AgentArtifactModel] = []

        for task in attraction_tasks:
            try:
                params = _extract_attraction_params(
                    task.text, state.model_name, state.temperature
                )
            except Exception as exc:
                err_content = AttractionArtifactContentModel(
                    task_ref=task.name,
                    status="failed",
                    provider="openai_embeddings+llm+serpapi_google_maps",
                    destination="",
                    budget=0.0,
                    selected_archetype="",
                    errors=[_normalize_error("parse_error", f"Parameter extraction failed: {exc}")],
                    config={"openai_api_key_set": bool(config.openai_api_key)},
                )
                artifacts.append(
                    AgentArtifactModel(
                        name=task.name,
                        type="attraction-search-result",
                        content=err_content.model_dump(mode="json"),
                        description=f"Attraction search failed for task '{task.name}': parameter extraction error",
                    )
                )
                continue

            content = run_attraction_search(
                params=params,
                model_name=state.model_name,
                temperature=state.temperature,
                config=config,
                task_ref=task.name,
            )
            place_info = f"place: {content.item.location_name}" if content.item else "no item"
            artifacts.append(
                AgentArtifactModel(
                    name=task.name,
                    type="attraction-search-result",
                    content=content.model_dump(mode="json"),
                    description=(
                        f"{params.destination} day {params.day} — "
                        f"archetype: {content.selected_archetype}, {place_info}, status: {content.status}"
                    ),
                )
            )

        existing = state.agent_artifacts.get("attraction_search_agent", [])
        return {
            "agent_artifacts": {
                **state.agent_artifacts,
                "attraction_search_agent": existing + artifacts,
            },
        }

    graph = StateGraph(AttractionSearchAgentState)
    graph.add_node("attraction_search", attraction_search)
    graph.set_entry_point("attraction_search")
    graph.add_edge("attraction_search", END)
    return graph.compile()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    config = load_config_from_env()
    params = AttractionParamsModel(
        budget=80.0,
        destination="Barcelona",
        traveller_profile="solo digital nomad interested in the local startup scene, wants to blend remote work with exploration of creative and professional communities, slow pace",
        day=1,
        previous_activities="",
        orchestrator_hint=None,
    )
    result = run_attraction_search(
        params=params,
        model_name=config.answer_model_name,
        temperature=0.0,
        config=config,
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
