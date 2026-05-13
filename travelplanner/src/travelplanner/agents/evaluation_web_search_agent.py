"""Per-slot rationale verifier for the LLM-as-a-Judge evaluation pipeline.

This agent verifies the factual rationale of a single Slot from a TravelPlan
against either (a) the slot's own ``links`` (treated as authoritative), or
(b) a Tavily web search constructed from the slot's claims when no link is
provided.

It is a focused twin of ``general_web_search_agent`` — it deliberately reuses
the Tavily helpers from that module so we have a single place that talks to
Tavily, while keeping the evaluator-specific reasoning isolated here.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from langsmith import traceable

from travelplanner.agents.general_web_search_agent import (
    _extract_full_content,
    _search_tavily,
)
from travelplanner.evaluation.systemprompt import (
    EVAL_RATIONALE_SYSTEM_PROMPT,
    _format_slot_block,
    build_rationale_verification_prompt,
)
from travelplanner.schema.judge_artifact import (
    RationaleVerificationModel,
    RationaleVerificationOutputModel,
)
from travelplanner.travelplan.plan import TravelPlan
from travelplanner.travelplan.slot import Slot
from travelplanner.utils.llm import invoke_structured_model


# ─── Constants ────────────────────────────────────────────────────────────────

_EXTRACT_TIMEOUT_SECONDS = 15
_SEARCH_MAX_RESULTS = 5
_SEARCH_TIMEOUT_SECONDS = 20
_FETCH_CONTENT_CHAR_BUDGET = 2000
_MAX_PARALLEL_SLOT_VERIFICATIONS = 6


# ─── Helpers ──────────────────────────────────────────────────────────────────

_NUMERIC_RE = re.compile(r"\d[\d.,]*")


def _build_search_query(slot: Slot) -> str:
    """Build a focused Tavily query from a slot's claims."""
    bits: list[str] = []
    if slot.name:
        bits.append(slot.name.strip())
    if slot.location:
        bits.append(slot.location.strip())
    # Pull any numeric / category cue from notes & description so the query is
    # specific to the claim rather than the venue alone.
    cue_text = " ".join(filter(None, [slot.description or "", slot.notes or ""]))
    numeric_cues = _NUMERIC_RE.findall(cue_text)
    if numeric_cues:
        bits.append(numeric_cues[0])
    if slot.category and slot.category != "other":
        bits.append(slot.category)
    query = " ".join(bits)
    return query.strip() or (slot.name or "travel slot")


def _format_evidence_block_from_extract(extract: dict[str, Any]) -> str:
    url = extract.get("url", "")
    title = extract.get("title", "")
    raw = (extract.get("raw_content", "") or "")[:_FETCH_CONTENT_CHAR_BUDGET]
    lines = [
        f"URL: {url}",
        f"Title: {title or '(unknown)'}",
        "",
        raw if raw else "(no content retrieved)",
    ]
    return "\n".join(lines)


def _format_evidence_block_from_search(
    *, query: str, results: list[dict[str, Any]]
) -> str:
    lines = [
        f"Search query: {query}",
        f"Result count: {len(results)}",
        "",
    ]
    if not results:
        lines.append("(no results)")
        return "\n".join(lines)
    for i, r in enumerate(results[:_SEARCH_MAX_RESULTS], start=1):
        title = r.get("title", "") or "(untitled)"
        url = r.get("url", "") or "(no url)"
        content = (r.get("content") or r.get("snippet") or "").strip()
        if len(content) > 600:
            content = content[:600].rstrip() + "…"
        lines.append(f"[{i}] {title}")
        lines.append(f"    {url}")
        if content:
            lines.append(f"    {content}")
    return "\n".join(lines)


def _gather_evidence(slot: Slot) -> tuple[str, list[str], list[str]]:
    """Return (source_type, source_urls, evidence_blocks) for a slot.

    Tries slot.links first; falls back to a Tavily web search using a
    keyword query derived from the slot's claims.
    """
    if slot.links:
        evidence_blocks: list[str] = []
        source_urls: list[str] = []
        try:
            extracts = _extract_full_content(
                list(slot.links),
                timeout=_EXTRACT_TIMEOUT_SECONDS,
                extract_depth="basic",
            )
        except Exception:
            extracts = []
        for e in extracts:
            url = e.get("url", "")
            if url:
                source_urls.append(url)
            evidence_blocks.append(_format_evidence_block_from_extract(e))
        # If Tavily extract returned nothing, still record the URLs so the
        # judge knows what we tried.
        if not source_urls:
            source_urls = list(slot.links)
        return "link", source_urls, evidence_blocks

    query = _build_search_query(slot)
    try:
        search_result = _search_tavily(
            query,
            max_results=_SEARCH_MAX_RESULTS,
            timeout=_SEARCH_TIMEOUT_SECONDS,
            search_depth="basic",
            include_answer=False,
        )
    except Exception as exc:
        return "web_search", [], [
            _format_evidence_block_from_search(query=query, results=[])
            + f"\n(search failed: {exc})"
        ]
    results = search_result.get("results", []) if search_result.get("ok") else []
    source_urls = [r.get("url", "") for r in results if r.get("url")]
    block = _format_evidence_block_from_search(query=query, results=results)
    return "web_search", source_urls, [block]


@traceable(name="slot_rationale_verification")
def verify_slot(
    *,
    day_index: int,
    slot_position: int,
    slot: Slot,
    model_name: str,
) -> RationaleVerificationModel:
    """Verify a single slot's factual rationale against retrieved evidence."""
    source_type, source_urls, evidence_blocks = _gather_evidence(slot)

    slot_block = _format_slot_block(
        day_index=day_index,
        slot_position=slot_position,
        name=slot.name,
        description=slot.description,
        location=slot.location,
        category=slot.category,
        start_time=slot.start_time.isoformat(),
        end_time=slot.end_time.isoformat(),
        cost=slot.cost,
        links=list(slot.links),
        notes=slot.notes,
    )
    user_prompt = build_rationale_verification_prompt(
        slot_block=slot_block,
        source_type=source_type,
        evidence_blocks=evidence_blocks,
    )

    try:
        output, _, _ = invoke_structured_model(
            model_name=model_name,
            temperature=0.0,
            system_prompt=EVAL_RATIONALE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=RationaleVerificationOutputModel,
        )
        return RationaleVerificationModel(
            day_index=day_index,
            slot_position=slot_position,
            slot_name=slot.name,
            source_type=source_type,
            source_urls=source_urls,
            verdict=output.verdict,
            reasoning=output.reasoning,
            claims_checked=output.claims_checked,
        )
    except Exception as exc:
        return RationaleVerificationModel(
            day_index=day_index,
            slot_position=slot_position,
            slot_name=slot.name,
            source_type=source_type,
            source_urls=source_urls,
            verdict="MISSING_INFO",
            reasoning=f"Rationale verification failed: {exc}",
            claims_checked=[],
        )


def verify_all_slots(
    plan: TravelPlan,
    model_name: str,
    progress_callback: Callable[[RationaleVerificationModel, int, int], None] | None = None,
) -> list[RationaleVerificationModel]:
    """Verify every slot in the plan in parallel and return the results.

    Slots are returned ordered by (day_index, slot_position).

    Args:
        plan: TravelPlan whose slots to verify.
        model_name: Model used by each per-slot verifier.
        progress_callback: Optional ``(rv, done, total)`` callback fired as each
            slot completes. ``done`` is 1-based.
    """
    jobs: list[tuple[int, int, Slot]] = []
    for day in plan.days:
        for pos, slot in enumerate(day.sorted_slots(), start=1):
            jobs.append((day.index, pos, slot))

    if not jobs:
        return []

    total = len(jobs)
    done = 0
    results: list[RationaleVerificationModel | None] = [None] * len(jobs)
    with ThreadPoolExecutor(
        max_workers=min(len(jobs), _MAX_PARALLEL_SLOT_VERIFICATIONS)
    ) as pool:
        future_to_idx = {
            pool.submit(
                verify_slot,
                day_index=day_index,
                slot_position=slot_position,
                slot=slot,
                model_name=model_name,
            ): idx
            for idx, (day_index, slot_position, slot) in enumerate(jobs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            day_index, slot_position, slot = jobs[idx]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = RationaleVerificationModel(
                    day_index=day_index,
                    slot_position=slot_position,
                    slot_name=slot.name,
                    source_type="skipped",
                    source_urls=[],
                    verdict="MISSING_INFO",
                    reasoning=f"Unexpected error during verification: {exc}",
                    claims_checked=[],
                )
            done += 1
            if progress_callback is not None:
                try:
                    progress_callback(results[idx], done, total)
                except Exception:
                    pass
    return [r for r in results if r is not None]
