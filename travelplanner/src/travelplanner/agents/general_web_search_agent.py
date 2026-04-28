from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from tavily import TavilyClient

from travelplanner.config import get_setting
from travelplanner.schema.general_web_search_artifact import (
    GeneralWebSearchArtifactContentModel,
    GeneralWebSearchErrorModel,
    GeneralWebSearchProofPointModel,
    GeneralWebSearchResultModel,
    GeneralWebSearchSourceModel,
)
from travelplanner.schema.system_state import (
    AgentArtifactModel,
    MessageHistoryModel,
    TaskModel,
)
from travelplanner.utils.llm import make_chat_model


SearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]

_VALID_SEARCH_DEPTHS: set[str] = {"basic", "advanced", "fast", "ultra-fast"}

_DEFAULT_MAX_RESULTS = 10
_DEFAULT_TIMEOUT_SECONDS = 30
_DEFAULT_EXTRACT_TIMEOUT = 15
_DEFAULT_ANSWER_MODEL = "openrouter:minimax/minimax-m2.5"
_DEFAULT_MAX_RETRIES = 1
_DEFAULT_MAX_SEARCHES = 3
_DEFAULT_EXTRACT_MAX_SOURCES = 3

# Prompt constants - these define agent behavior, not config tunables
_ANSWER_SYSTEM_PROMPT = """You are a detail-obsessed research analyst for a TRAVEL PLANNER agent.

MISSION: Transform every location, attraction, and service into an EXECUTION-READY recommendation. A name alone is worthless — every entity must be unpacked to its actionable specifics.

MANDATORY FIELDS (include all available for each entity):
- NAME (as given in source)
- ADDRESS or intersection (street number, neighborhood, city)
- TRANSIT: station name + line number (e.g., "Barcelona Sants, R2 Nord / R12")
- HOURS: specific open/close times, not "open daily"
- PRICE: exact admission/entry fee in local currency, not "affordable" or "moderate"
- PHONE/URL for official confirmation
- Noteworthy details: accessibility, photography rules, age minimums, etc.

SOURCING HIERARCHY (use in this order):
1. wikidata, wikipedia, government sites, official tourism boards
2. authoritative blog posts
3. SKIP: forum questions, review Q&A, Reddit threads, Quora

HANDLING MISSING DATA:
- If a detail is unavailable, write "— unavailable —" for that field
- Do NOT skip the field entirely
- If ALL fields are unavailable for an entity, still output it with "— no verifiable details —" and source URL

OUTPUT FORMAT (strict per entity):
```
[ENTITY] {name}
[LOCATION] {address or — unavailable —}
[TRANSIT] {station + line or — unavailable —}
[HOURS] {specific times or — unavailable —}
[PRICE] {exact amount currency or — unavailable —}
[DETAILS] {actionable specifics}
[SOURCE] {type} | {url}
```

FACT vs OPINION:
- FACTS: pool exists, closed Mondays, ¥500 entry, station is 400m away
- OPINIONS: beautiful, great for families, worth visiting, delicious food

COPY RESTRICTION: Never copy more than 3 consecutive words from any source.

QUALITY GATE: If you cannot fill at least 3 of the 5 mandatory fields (LOCATION, TRANSIT, HOURS, PRICE, DETAILS), mark entity clearly insufficient for planning."""

_ANSWER_INSTRUCTION = """Answer the user's query using the search results provided.

FORMAT: For each entity (place, attraction, route, etc.) you recommend, output a DENSE INFORMATION BLOCK:

**{Entity Name}**
- Location/Transit: [Specific directions: nearest station, line, duration, cost, walking minutes from station]
- Address: [Full street address if available]
- Hours/Schedule: [Specific hours, best time to visit, seasonal variations]
- Price: [Specific cost, currency, what's included vs. extra]
- Why It Fits: [How it addresses the user's specific request — be concrete, not generic]
- Confidence: [HIGH/MEDIUM/LOW — based on evidence quality and recency]

Return 3–6 entities maximum. Each entity block should be self-contained and specific enough that someone could visit without additional searching.

Do NOT output:
- Generic "options" or lists of alternatives without specifics
- Place names without supporting details
- Broad categories without naming exact venues

For LOGISTICS queries (directions, schedules, costs): prioritize transit details and precise timing.
For HISTORY/CULTURE queries: prioritize significance and what makes it worth visiting.
For ENTITY queries (specific places): prioritize address, hours, and practical visitor info.

Include confidence caveats inline using [Confidence: LOW — source is 2+ years old] or similar."""


@dataclass(frozen=True)
class GeneralWebSearchConfig:
    provider: str = "tavily"
    max_results: int = _DEFAULT_MAX_RESULTS
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
    max_retries: int = _DEFAULT_MAX_RETRIES
    max_searches: int = _DEFAULT_MAX_SEARCHES
    search_depth: SearchDepth = "basic"
    include_answer: bool = True
    answer_model_name: str | None = _DEFAULT_ANSWER_MODEL
    answer_temperature: float = 0.0
    answer_system_prompt: str = _ANSWER_SYSTEM_PROMPT
    answer_instruction: str = _ANSWER_INSTRUCTION
    extract_max_sources: int = _DEFAULT_EXTRACT_MAX_SOURCES
    extract_timeout: int = _DEFAULT_EXTRACT_TIMEOUT
    extract_quality_threshold: float = (
        0.4  # min quality score to use extracted content over snippet
    )


LOW_SIGNAL_SNIPPET_TOKENS: tuple[str, ...] = (
    "budget backpacking",
    "how to plan a trip",
    "what to pack",
    "subscribe",
    "cookie",
    "terms of service",
)

# Question detection patterns
_QUESTION_PREFIXES = (
    r"^(who|what|where|when|why|how|which|whose|whom)\s",
    r"^(can|could|should|would|will|is|are|was|were|do|does|did|have|has|had)\s",
)
_QUESTION_SUBSTRINGS = (
    "i think ",
    "could i ",
    "should i ",
    "is it ",
    "are there ",
    "can i ",
    "how do i ",
    "what is the best ",
    "any recommendation",
    "??",
)


def _is_question_snippet(snippet: str) -> bool:
    """Return True if snippet looks like a forum/review question rather than an answer."""
    stripped = snippet.strip()
    if not stripped or len(stripped) < 15:
        return True
    if stripped.endswith("?"):
        return True
    lower = stripped.lower()
    for prefix in _QUESTION_PREFIXES:
        if re.match(prefix, lower):
            return True
    for substr in _QUESTION_SUBSTRINGS:
        if substr in lower:
            return True
    return False


# Domain quality tiers
_TIER1_DOMAINS = frozenset(
    [
        "wikidata.org",
        "wikipedia.org",
        "wikivoyage.org",
        "openstreetmap.org",
        "wikimedia.org",
        "wikitravel.org",
        "lonelyplanet.com",
        "roughguides.com",
        "frommers.com",
        "fodors.com",
        "travelandleisure.com",
        "nationalgeographic.com",
        "bbc.com",
        "theguardian.com",
        "nytimes.com",
        "reuters.com",
        "apnews.com",
    ]
)
_TIER1_SUFFIXES = frozenset([".gov", ".edu", ".gov.", ".edu."])
_TIER2_DOMAINS = frozenset(
    [
        "roughguides.com",
        "wikivoyage.org",
        "frommers.com",
        "fodors.com",
        "travelandleisure.com",
        "cntraveler.com",
        "timeout.com",
        "afAR.com",
        "tripadvisor.com",
        "booking.com",
        "expedia.com",
        "agoda.com",
        "hotels.com",
    ]
)
_TIER4_DOMAINS = frozenset(
    [
        "reddit.com",
        "facebook.com",
        "quora.com",
        "twitter.com",
        "instagram.com",
        "youtube.com",
        "tiktok.com",
        "pinterest.com",
        "amazon.com",
        "ebay.com",
    ]
)

# Domains that return structured nav content (ToC, H1 lists) when extracted.
# These sites have good search relevance but poor extracted content quality.
# Use search snippet as evidence instead.
_EXTRACTION_SKIP_DOMAINS: frozenset[str] = frozenset(
    {
        "magical-trip.com",
        "getyourguide.com",
        "byfood.com",
        "bestejapan.com",
        "agoda.com",
        "tripadvisor.com",
        "viator.com",
        "airbnb.com",
        "booking.com",
        "hotels.com",
        "expedia.com",
        "klook.com",
        "yelp.com",
        "reddit.com",
        "facebook.com",
        "youtube.com",
        "instagram.com",
    }
)


def _get_domain_tier(url: str) -> int:
    """Return domain quality tier (1=best, 4=block). Lower = better."""
    if not url:
        return 3
    lower = url.lower()
    # Extract domain
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc.lower()
    except Exception:
        domain = lower

    # Strip www. prefix
    if domain.startswith("www."):
        domain = domain[4:]

    # Check exact matches first
    if any(d in domain for d in _TIER1_DOMAINS):
        return 1
    if any(d in domain for d in _TIER4_DOMAINS):
        return 4
    # Check suffixes
    for suffix in (".gov", ".edu"):
        if domain.endswith(suffix) or suffix in domain:
            return 1
    # Tier 2 known domains
    if any(d in domain for d in _TIER2_DOMAINS):
        return 2
    # Default tier 3 (unknown)
    return 3


def _get_skip_extraction(url: str) -> bool:
    """Return True if extraction should be skipped for this URL."""
    if not url:
        return True
    url_lower = url.lower()
    return any(domain in url_lower for domain in _EXTRACTION_SKIP_DOMAINS)


def _is_substantive_evidence(evidence: str) -> bool:
    """Return False if evidence is clearly non-substantive navigation/breadcrumb text."""
    if not evidence or len(evidence.strip()) < 30:
        return False
    ev = evidence.strip()

    # H1/H2 heading detection with zero-width unicode handling
    # Strip zero-width chars, join multiple spaces, then check
    heading_pattern = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]+", "", ev)
    heading_pattern = re.sub(
        r"(#+)[\s\u200b\u200c\u200d\u2060\ufeff]+", r"\1 ", heading_pattern
    )
    heading_pattern = re.sub(r"\s+", " ", heading_pattern).strip()
    if heading_pattern.startswith("# ") or heading_pattern.startswith("## "):
        return False

    # Pipe ToC detection at any position
    if "|" in ev and ev.count("|") >= 2:
        return False

    # Breadcrumb at START of evidence (case-insensitive, regardless of total length)
    ev_lower = ev.lower()
    if ev_lower.startswith("you are here:"):
        return False
    if ev_lower.startswith("home / "):
        return False
    # General breadcrumb: starts with path-like segment followed by " / "
    # e.g., "Spain / France / Italy" at the beginning
    if re.match(r"^[a-z][a-z\s]*/ /", ev_lower[:50]) is not None:
        return False

    # Middle-dot enumeration (tour recommendation list pattern)
    if ev.count("\u00b7") >= 2:
        return False

    if ev.startswith("http://") or ev.startswith("https://"):
        return False
    return True


def _is_high_signal_source(source: GeneralWebSearchSourceModel) -> bool:
    snippet = (source.snippet or "").strip().lower()
    if len(snippet) < 40:
        return False
    return not any(token in snippet for token in LOW_SIGNAL_SNIPPET_TOKENS)


def _score_extracted_content(raw_content: str) -> float:
    """Score extracted content quality 0.0-1.0. Higher = more likely to be substantive body text."""
    if not raw_content or len(raw_content) < 200:
        return 0.0

    content_lower = raw_content.lower()

    # Penalize navigation/structure indicators
    # Nav patterns found in boilerplate/nav text from tour sites, blogs, etc.
    nav_indicators = [
        "home / travel",
        "home / food",
        "home / hotels",
        "tour overview",
        "breadcrumb",
        "cookie",
        "subscribe",
        "| table of contents",
        "navigation menu",
        "back to top",
        "advertisement",
        "you are here:",
        # Tour/activity site structured nav (pipe-separated ToC lines)
        "| tour overview",
        "| features",
        "| itinerary",
        "| ",
    ]
    nav_count = sum(1 for ind in nav_indicators if ind in content_lower)
    # Nav penalty is STRONGER than substantive bonus: 0.20 per nav indicator
    # (tour-site nav text often contains "experience", "enjoy", "savor" which are
    # substantive-sounding but appear in nav/ToC structure — nav must dominate)
    nav_penalty = min(0.75, nav_count * 0.20)

    # Reward substantive indicators — real body content from articles
    substantive_indicators = [
        "is located",
        "is situated",
        "approximately",
        "takes about",
        "costs around",
        "opens at",
        "closes at",
        "walk from",
        "take the",
        "metro to",
        "station to",
        "address:",
        "phone:",
        "hours:",
        "price:",
        "built in",
        "established",
        "founded",
    ]
    subst_count = sum(1 for ind in substantive_indicators if ind in content_lower)
    subst_bonus = min(0.3, subst_count * 0.06)

    # Length bonus (longer content more likely substantive body, not nav snippet)
    length_ratio = min(1.0, len(raw_content) / 2000)

    score = 0.5 + subst_bonus - nav_penalty + (length_ratio * 0.1)
    return max(0.0, min(1.0, score))


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _parse_int(value: str | None, *, default: int, minimum: int) -> int:
    if value is None or value.strip() == "":
        return default
    parsed = int(value.strip())
    return max(parsed, minimum)


def load_config_from_env() -> GeneralWebSearchConfig:
    cfg_prefix = "agents.general_web_search"

    answer_model_name_raw = os.getenv(
        "TRAVELPLANNER_GENERAL_WEB_SEARCH_ANSWER_MODEL",
        get_setting(f"{cfg_prefix}.answer_model_name", _DEFAULT_ANSWER_MODEL),
    )
    if isinstance(answer_model_name_raw, str):
        answer_model_name = answer_model_name_raw.strip()
    else:
        answer_model_name = str(answer_model_name_raw).strip()
    if answer_model_name.lower() in {"", "none", "off", "false"}:
        answer_model_name = ""

    _search_depth_env = os.getenv(
        "TRAVELPLANNER_GENERAL_WEB_SEARCH_DEPTH",
        str(get_setting(f"{cfg_prefix}.search_depth", "basic")),
    ).strip()
    _search_depth = cast(
        SearchDepth,
        _search_depth_env if _search_depth_env in _VALID_SEARCH_DEPTHS else "basic",
    )

    return GeneralWebSearchConfig(
        provider=str(get_setting(f"{cfg_prefix}.provider", "tavily")),
        max_results=_parse_int(
            os.getenv("TRAVELPLANNER_GENERAL_WEB_SEARCH_MAX_RESULTS"),
            default=int(get_setting(f"{cfg_prefix}.max_results", _DEFAULT_MAX_RESULTS)),
            minimum=1,
        ),
        timeout_seconds=_parse_int(
            os.getenv("TRAVELPLANNER_GENERAL_WEB_SEARCH_TIMEOUT_SECONDS"),
            default=int(
                get_setting(f"{cfg_prefix}.timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
            ),
            minimum=5,
        ),
        max_retries=_parse_int(
            os.getenv("TRAVELPLANNER_GENERAL_WEB_SEARCH_MAX_RETRIES"),
            default=int(get_setting(f"{cfg_prefix}.max_retries", _DEFAULT_MAX_RETRIES)),
            minimum=1,
        ),
        max_searches=_parse_int(
            os.getenv("TRAVELPLANNER_GENERAL_WEB_SEARCH_MAX_SEARCHES"),
            default=int(
                get_setting(f"{cfg_prefix}.max_searches", _DEFAULT_MAX_SEARCHES)
            ),
            minimum=1,
        ),
        search_depth=_search_depth,
        include_answer=_parse_bool(
            os.getenv("TRAVELPLANNER_GENERAL_WEB_SEARCH_INCLUDE_ANSWER"),
            default=bool(get_setting(f"{cfg_prefix}.include_answer", True)),
        ),
        answer_model_name=answer_model_name or None,
        answer_temperature=float(
            os.getenv(
                "TRAVELPLANNER_GENERAL_WEB_SEARCH_ANSWER_TEMPERATURE",
                str(get_setting(f"{cfg_prefix}.answer_temperature", "0.0")),
            )
        ),
        extract_max_sources=_parse_int(
            os.getenv("TRAVELPLANNER_GENERAL_WEB_SEARCH_EXTRACT_MAX_SOURCES"),
            default=int(
                get_setting(
                    f"{cfg_prefix}.extract_max_sources", _DEFAULT_EXTRACT_MAX_SOURCES
                )
            ),
            minimum=1,
        ),
        extract_timeout=_parse_int(
            os.getenv("TRAVELPLANNER_GENERAL_WEB_SEARCH_EXTRACT_TIMEOUT"),
            default=int(
                get_setting(f"{cfg_prefix}.extract_timeout", _DEFAULT_EXTRACT_TIMEOUT)
            ),
            minimum=5,
        ),
    )


class GeneralWebSearchAgentState(BaseModel):
    query: str
    task_list: list[TaskModel] = Field(default_factory=list)
    agent_artifacts: dict[str, list[AgentArtifactModel]] = Field(default_factory=dict)
    message_history: MessageHistoryModel | None = None


def _build_message_history(
    messages: list[dict[str, str]],
    model_ref: str,
) -> MessageHistoryModel:
    return MessageHistoryModel(
        user_agent="general_web_search_agent",
        model=model_ref,
        agent_ref="travelplanner.agents.general_web_search_agent",
        messages=messages,
    )


def _extract_search_tasks(task_list: list[TaskModel]) -> list[TaskModel]:
    return [
        task
        for task in task_list
        if task.is_valid and task.type == "general-web-search"
    ]


def _search_tavily(
    query: str,
    *,
    max_results: int,
    timeout: int,
    search_depth: SearchDepth,
    include_answer: bool,
) -> dict[str, Any]:
    try:
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return {
                "ok": False,
                "query": query,
                "error": "Missing TAVILY_API_KEY environment variable.",
                "results": [],
            }

        client = TavilyClient(api_key=api_key)
        raw_response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
        )

        results = raw_response.get("results", [])

        return {
            "ok": True,
            "query": query,
            "answer": raw_response.get("answer"),
            "results": results,
            "raw_response": raw_response,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "query": query,
            "error": str(exc),
            "results": [],
        }


ExtractDepth = Literal["basic", "advanced"]


def _extract_full_content(
    urls: list[str],
    *,
    timeout: int,
    extract_depth: ExtractDepth = "basic",
) -> list[dict[str, Any]]:
    """Extract full page content from URLs using Tavily extract API.

    Returns list of {url, raw_content, title, quality_score} dicts.
    quality_score is 0.0-1.0 (higher = more likely substantive body text).
    Falls back to empty raw_content on failure — caller should use snippet as fallback.
    """
    if not urls:
        return []
    try:
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return []
        client = TavilyClient(api_key=api_key)
        response = client.extract(
            urls=urls,
            extract_depth=extract_depth,
            timeout=timeout,
        )
        results = []
        for item in response.get("results", []):
            raw_content = item.get("raw_content", "")
            results.append(
                {
                    "url": item.get("url", ""),
                    "raw_content": raw_content,
                    "title": item.get("title", ""),
                    "quality_score": _score_extracted_content(raw_content),
                }
            )
        return results
    except Exception:  # noqa: BLE001
        return []


def _build_answer_prompt(query: str, result: dict[str, Any], instruction: str) -> str:
    answer_input = {
        "query": query,
        "answer": result.get("answer"),
        "results": result.get("results", []),
    }
    return "\n".join(
        [
            instruction,
            "",
            "Search payload JSON:",
            json.dumps(answer_input, ensure_ascii=True, indent=2),
            "",
            "Return plain text answer with max 8 bullet points and include confidence caveats.",
        ]
    )


def _synthesize_answer_with_model(
    *,
    system_prompt: str,
    instruction: str,
    model_name: str,
    temperature: float,
    query: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    try:
        client = make_chat_model(model_name=model_name, temperature=temperature)
        response = client.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=_build_answer_prompt(query, result, instruction)),
            ]
        )
        return {
            "ok": True,
            "model": model_name,
            "text": str(response.content).strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "model": model_name,
            "error": str(exc),
        }


def _normalize_sources(result: dict[str, Any]) -> list[GeneralWebSearchSourceModel]:
    normalized: list[GeneralWebSearchSourceModel] = []
    seen: set[str] = set()
    for entry in result.get("results", []):
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("url") or entry.get("title") or "").strip().lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        normalized.append(
            GeneralWebSearchSourceModel(
                title=str(entry.get("title"))
                if entry.get("title") is not None
                else None,
                url=str(entry.get("url")) if entry.get("url") is not None else None,
                snippet=str(entry.get("content") or entry.get("snippet") or "") or None,
                score=(lambda s: float(s) if isinstance(s, (int, float)) else None)(
                    entry.get("score")
                ),
            )
        )
    return normalized


def _score_and_filter_sources(
    sources: list[GeneralWebSearchSourceModel],
) -> list[tuple[GeneralWebSearchSourceModel, float]]:
    """Score and filter sources by composite quality score.

    Composite = domain_tier_weighted * tavily_score
    where domain_tier_weighted = (5 - tier) so tier1=4, tier2=3, tier3=2, tier4=0.

    Also filters out:
    - Question snippets
    - Tier 4 domains entirely
    - Low signal sources
    """
    scored: list[tuple[GeneralWebSearchSourceModel, float]] = []

    for source in sources:
        url = source.url or ""
        snippet = source.snippet or ""

        # Skip tier 4 (reddit, facebook, etc.)
        tier = _get_domain_tier(url)
        if tier == 4:
            continue

        # Skip question snippets
        if _is_question_snippet(snippet):
            continue

        # Skip low signal
        if not _is_high_signal_source(source):
            continue

        # Composite score = tier-weighted * tavily_score
        tier_weight = 5 - tier  # tier1->4, tier2->3, tier3->2
        tavily_score = source.score if source.score is not None else 0.3
        composite = tier_weight * tavily_score

        scored.append((source, composite))

    # Sort by composite score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _build_proof_points_with_model(
    *,
    system_prompt: str,
    model_name: str,
    temperature: float,
    query: str,
    sources: list[GeneralWebSearchSourceModel],
    extracted_contents: dict[str, str],
) -> list[GeneralWebSearchProofPointModel]:
    if not sources:
        return []

    # Good claims (claim != evidence copy)
    GOOD_CLAIMS = [
        "The Shinkansen from Tokyo to Osaka takes 2.5 hours and costs around ¥14,000.",
        "N Osaka metro Line 1 runs from Namba to Umeda every 5 minutes until midnight.",
        "La Sagrada Familia tickets must be booked online at least 3 days in advance.",
    ]
    # Bad claims (claim = evidence copy)
    BAD_CLAIMS = [
        '"La Sagrada Familia is a large church built by Antoni Gaudi" — copied verbatim',
        '"The hotel is beautiful and the staff is friendly" — subjective opinion, not fact',
        '"Multiple food options are available nearby" — too vague, no specifics',
    ]

    sources_input = []
    for i, src in enumerate(sources):
        snippet = (src.snippet or "")[:200].strip()
        url = src.url or "unknown"
        content = extracted_contents.get(url, snippet)
        quality = _score_extracted_content(content) if content else 0.0
        word_count = len(content.split()) if content else 0
        sources_input.append(
            {
                "index": i,
                "title": src.title or "unknown",
                "url": url,
                "evidence": content[:1500].strip() if content else snippet,
                "tier": _get_domain_tier(url),
                "quality": quality,
                "word_count": word_count,
            }
        )

    # Filter out non-substantive evidence before sending to LLM
    filtered_sources = [
        {**src, "evidence": src["evidence"]}
        for src in sources_input
        if _is_substantive_evidence(src.get("evidence", ""))
    ]
    if not filtered_sources:
        return []
    sources_for_model = filtered_sources

    user_prompt = f"""You are a research analyst extracting key findings from web search results.

## Query
{query}

## IMPORTANT - HALLUCINATION PREVENTION
- ONLY claim a specific beach name, restaurant name, station name, or price IF that exact information appears in the evidence
- If evidence says "golden sandy beaches" but does NOT name "Bogatell Beach", you CANNOT claim "Bogatell Beach provides..."
- If evidence only gives a neighborhood name like "Poblenou area" and does not name a specific restaurant, you CANNOT claim the specific restaurant name
- Generic descriptive words like "beautiful", "clean", "popular" can be reused from evidence. Specific proper nouns (names) CANNOT be invented

## Task
For each source below, produce a brief "claim" that:
1. Answers or addresses the query
2. Is phrased DIFFERENTLY from the source text — REWRITE, don't copy
3. Is a useful, SPECIFIC finding (transit line name, price, hours, address, etc.)
4. Is grounded in what the source actually says
5. Is a FACT not an opinion

Also provide a confidence score (0.0-1.0) and one-line usefulness note.

## Anti-copy rules (CRITICAL)
- Do NOT copy phrases longer than 3 words from any source
- REWRITE everything in your own words
- If you must reference specific data (e.g. a price), cite it but restate it differently
- Phrases like "La Sagrada Familia is a large church" are COPY — say instead "The basilica by Gaudi dominates the Eixample skyline"

## Source type classification
For each source identify TYPE: authoritative (gov/edu/wikipedia), blog (travel site), forum (reddit/quora), review (tripadvisor booking), question (Q&A).
Only use authoritative and blog types as primary evidence. Skip forum/review/question types unless they contain unique factual information not available elsewhere.

## Question exclusion
If a source is clearly a forum question ("how do I get to X?", "should I visit Y?") or a review question, SKIP it.
Questions are not evidence — they represent uncertainty, not facts.

## Evidence quality check (MANDATORY before producing claims)
Skip any source where:
- word_count < 20 (not enough substantive content)
- quality < 0.4 (navigation/boilerplate likely — e.g. "Tour Overview", "Home / Travel", table of contents)

If a source has insufficient substantive content, EXCLUDE it from the JSON output entirely.

## Few-shot examples

GOOD claims (specific, rewritten, factual):
{GOOD_CLAIMS[0]}
{GOOD_CLAIMS[1]}
{GOOD_CLAIMS[2]}

BAD claims (copied, vague, opinion, or question):
{BAD_CLAIMS[0]}
{BAD_CLAIMS[1]}
{BAD_CLAIMS[2]}

## Sources JSON
{json.dumps(sources_for_model, ensure_ascii=True, indent=2)}

## Output format
Return a JSON array of objects with keys: index, claim, confidence, usefulness_note.
Return ONLY the JSON array, no markdown, no code fences, no preamble."""

    try:
        client = make_chat_model(model_name=model_name, temperature=temperature)
        response = client.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        raw = str(response.content).strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        parsed = json.loads(raw)
        proof_points = []
        for item in parsed:
            idx = item.get("index")
            src = (
                sources[idx]
                if idx is not None and 0 <= idx < len(sources)
                else sources[0]
            )
            evidence = (src.snippet or "")[:200].strip()
            proof_points.append(
                GeneralWebSearchProofPointModel(
                    claim=str(item.get("claim", "")).strip(),
                    evidence=evidence,
                    confidence=float(item.get("confidence"))
                    if item.get("confidence") is not None
                    else None,
                    source_url=src.url,
                )
            )
        return proof_points
    except Exception as exc:  # noqa: BLE001
        return []


def _normalize_error(error_message: str) -> GeneralWebSearchErrorModel:
    lowered = error_message.lower()
    if "missing tavily_api_key" in lowered:
        code = "missing_api_key"
    elif "httperror" in lowered:
        code = "http_error"
    elif "urlerror" in lowered or "timeout" in lowered:
        code = "timeout_error"
    elif "json" in lowered:
        code = "parse_error"
    else:
        code = "unknown_error"
    return GeneralWebSearchErrorModel(code=code, message=error_message)


def _compute_status(
    *, result_ok: bool, synthesis_ok: bool
) -> Literal["success", "partial", "failed", "skipped"]:
    if not result_ok:
        return "failed"
    if synthesis_ok:
        return "success"
    return "partial"


def _build_answer_with_proof(
    *,
    result: dict[str, Any],
    sources: list[GeneralWebSearchSourceModel],
    synthesis_ok: bool,
    answer_text: str | None,
    config: GeneralWebSearchConfig,
    query: str,
    extracted_contents: dict[str, str],
) -> tuple[str, list[GeneralWebSearchProofPointModel]]:
    final_answer = ""
    if synthesis_ok and answer_text:
        final_answer = answer_text.strip()
    elif isinstance(result.get("answer"), str):
        final_answer = result["answer"].strip()
    else:
        final_answer = (
            "No reliable answer could be synthesized from retrieved evidence."
        )

    # Filter and score sources by quality
    scored_sources = _score_and_filter_sources(sources)
    top_sources = [s for s, _ in scored_sources[: config.extract_max_sources * 2]]

    proof_points: list[GeneralWebSearchProofPointModel] = []
    if config.answer_model_name and config.answer_system_prompt:
        proof_points = _build_proof_points_with_model(
            system_prompt=config.answer_system_prompt,
            model_name=config.answer_model_name,
            temperature=config.answer_temperature,
            query=query,
            sources=top_sources[:6],
            extracted_contents=extracted_contents,
        )

    if not proof_points:
        for source, _ in scored_sources[:6]:
            evidence = (source.snippet or "").strip()
            if not evidence:
                continue
            if not _is_substantive_evidence(evidence):
                continue
            clean_evidence = evidence[:200].strip()
            first_sentence = clean_evidence
            for se in (". ", ".", "! ", "!", "? ", "?"):
                if se in clean_evidence:
                    idx = clean_evidence.index(se)
                    first_sentence = clean_evidence[: idx + len(se.rstrip())].strip()
                    break
            for prefix in ("the ", "a ", "an "):
                if first_sentence.lower().startswith(prefix):
                    first_sentence = first_sentence[len(prefix) :]
                    break
            claim = (
                first_sentence[:120] if len(first_sentence) > 120 else first_sentence
            )
            if len(claim) == len(clean_evidence) and len(clean_evidence) > 120:
                claim = clean_evidence[:117].rsplit(" ", 1)[0] + "…"
            proof_points.append(
                GeneralWebSearchProofPointModel(
                    claim=claim,
                    evidence=clean_evidence,
                    confidence=None,
                    source_url=source.url,
                )
            )

    if not proof_points and sources:
        fallback = sources[0]
        clean_evidence = (fallback.snippet or "")[
            :200
        ].strip() or "Source evidence unavailable."
        proof_points.append(
            GeneralWebSearchProofPointModel(
                claim=final_answer[:120]
                if final_answer
                else "No explicit claim available.",
                evidence=clean_evidence,
                confidence=None,
                source_url=fallback.url,
            )
        )
    return final_answer, proof_points


@dataclass
class GapAnalysisResult:
    """Result of gap analysis for a single missing detail."""

    gap_type: str
    description: str
    suggested_query: str


def _analyze_gaps_and_suggest_queries(
    query: str,
    aggregated_results: list[dict[str, Any]],
    existing_answer: str | None,
    model_name: str | None,
    temperature: float = 0.0,
) -> list[GapAnalysisResult]:
    """Analyze search results to identify missing details and suggest follow-up queries.

    Uses LLM to identify what's MISSING after initial search. Returns up to 2 gap
    objects with gap_type, description, and suggested_query. Falls back to empty
    list on error (caller falls back to static behavior).

    Args:
        query: Original user query
        aggregated_results: List of Tavily search result dicts
        existing_answer: Current answer text if any (may be None)
        model_name: Model name for gap analysis LLM call
        temperature: Temperature for LLM call

    Returns:
        List of GapAnalysisResult objects (max 2), empty list on error
    """
    if not model_name:
        return []

    try:
        result_texts: list[str] = []
        for r in aggregated_results[:5]:
            title = r.get("title", "")
            snippet = r.get("content") or r.get("snippet", "")
            url = r.get("url", "")
            if title or snippet:
                result_texts.append(
                    f"Title: {title}\nURL: {url}\nContent: {snippet[:300]}"
                )

        if not result_texts:
            return []

        results_summary = "\n\n---\n\n".join(result_texts)

        analysis_prompt = f"""You are a gap analyst for a universal research assistant that handles ANY question type — travel, biography, history, science, current events, logistics, or any topic.

ORIGINAL QUERY: {query}

CURRENT ANSWER (if any):
{existing_answer or "(no answer yet)"}

SEARCH RESULTS SO FAR:
{results_summary}

Your job: Identify gaps that can be RESOLVED by another targeted search. Focus on gaps where:
1. A specific follow-up query would likely find the missing info
2. The missing info is CRITICAL for answering the query

IGNORE gaps that are inherent data limitations (small businesses without web presence, opinion-based queries, etc.)

For each FILLABLE gap:
1. GAP TYPE: missing_context, missing_verification, missing_access, missing_cost, missing_timing, or missing_source
2. DESCRIPTION: what's missing
3. SUGGESTED_QUERY: specific query (max 15 words) that would fill this gap

DO NOT return gaps for:
- missing_location/address for small local businesses without verified addresses (restaurants, cafes, shops)
- missing_concrete_details for hidden gems or local favorites
- gaps that require visiting the physical location to verify

CRITICAL: Only suggest a query if you believe a search with different keywords/entity names WOULD find the missing info. If the info simply isn't online, don't suggest a fruitless search.

Rules:
- Return max 2 gaps, prioritize CRITICAL gaps
- Do NOT suggest queries that are just the original query repeated
- If results already have sufficient detail for a good answer, return empty list

Gap type guidance:
- missing_context: "biography of X", "history of Y" — search differently
- missing_verification: "confirm Z fact" — search authoritative source
- missing_access: "how to get to X", "transit to Y" — search transit routes
- missing_cost: "price of X", "admission Y" — search official pricing
- missing_timing: "hours of X", "when does Y open" — search official hours
- missing_source: "source for X claim" — search authoritative source

Output as JSON list:
[{{"gap_type": "...", "description": "...", "suggested_query": "..."}}]
"""

        model = make_chat_model(
            model_name=model_name,
            temperature=temperature,
        )
        response = model.invoke([HumanMessage(content=analysis_prompt)])
        content = str(response.content).strip()

        # Try to extract JSON from response
        json_start = content.find("[")
        json_end = content.rfind("]") + 1
        if json_start == -1 or json_end == 0:
            return []

        json_str = content[json_start:json_end]
        parsed = json.loads(json_str)

        gaps: list[GapAnalysisResult] = []
        for item in parsed[:2]:
            if (
                isinstance(item, dict)
                and item.get("gap_type")
                and item.get("suggested_query")
            ):
                gaps.append(
                    GapAnalysisResult(
                        gap_type=str(item["gap_type"]),
                        description=str(item.get("description", "")),
                        suggested_query=str(item["suggested_query"]),
                    )
                )
        return gaps

    except Exception:  # noqa: BLE001
        # Fall back to static behavior on any error
        return []


def make_graph():
    config = load_config_from_env()
    model_ref = (
        f"tavily+{config.answer_model_name}" if config.answer_model_name else "tavily"
    )

    def search_node(state: GeneralWebSearchAgentState) -> dict[str, Any]:
        search_tasks = _extract_search_tasks(state.task_list)
        base_artifacts = dict(state.agent_artifacts)
        existing = list(base_artifacts.get("general_web_search_agent", []))
        messages: list[dict[str, str]] = []

        if not search_tasks:
            messages.append(
                {
                    "role": "assistant",
                    "content": "No valid general-web-search tasks found. Skipping execution.",
                }
            )
            return {
                "agent_artifacts": base_artifacts,
                "message_history": _build_message_history(messages, model_ref),
            }

        for task in search_tasks:
            aggregated_results: list[dict[str, Any]] = []
            last_result: dict[str, Any] = {
                "ok": False,
                "query": task.text,
                "results": [],
            }
            searches_done = 0
            attempt = 0
            search_outcomes: list[dict[str, Any]] = []
            best_score_seen: float = 0.0

            for _ in range(config.max_retries):
                attempt += 1
                last_result = _search_tavily(
                    task.text,
                    max_results=config.max_results,
                    timeout=config.timeout_seconds,
                    search_depth=config.search_depth,
                    include_answer=config.include_answer,
                )
                if last_result.get("ok"):
                    aggregated_results.extend(last_result.get("results", []))
                    break
            searches_done += 1
            for r in aggregated_results:
                score = r.get("score", 0) if isinstance(r, dict) else 0
                if score > best_score_seen:
                    best_score_seen = score
            search_outcomes.append(
                {
                    "query": last_result.get("query", task.text),
                    "ok": bool(last_result.get("ok")),
                    "attempt": attempt,
                    "error": None
                    if last_result.get("ok")
                    else str(last_result.get("error", "")),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"search=1 query='{task.text}' ok={bool(last_result.get('ok'))}"
                    ),
                }
            )

            if best_score_seen >= 0.5 and len(aggregated_results) >= 2:
                gaps = []
            elif searches_done < config.max_searches:
                gaps = _analyze_gaps_and_suggest_queries(
                    query=task.text,
                    aggregated_results=aggregated_results,
                    existing_answer=None,
                    model_name=config.answer_model_name,
                    temperature=config.answer_temperature,
                )
                if gaps:
                    for gap in gaps[:1]:
                        if searches_done >= config.max_searches:
                            break
                        refined_query = gap.suggested_query
                        attempt = 0
                        for _ in range(config.max_retries):
                            attempt += 1
                            last_result = _search_tavily(
                                refined_query,
                                max_results=config.max_results,
                                timeout=config.timeout_seconds,
                                search_depth=config.search_depth,
                                include_answer=config.include_answer,
                            )
                            if last_result.get("ok"):
                                aggregated_results.extend(
                                    last_result.get("results", [])
                                )
                                break
                        searches_done += 1
                        for r in last_result.get("results", []):
                            score = r.get("score", 0) if isinstance(r, dict) else 0
                            if score > best_score_seen:
                                best_score_seen = score
                        search_outcomes.append(
                            {
                                "query": last_result.get("query", refined_query),
                                "ok": bool(last_result.get("ok")),
                                "attempt": attempt,
                                "error": None
                                if last_result.get("ok")
                                else str(last_result.get("error", "")),
                            }
                        )
                        messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    f"search={searches_done} gap_query='{refined_query}' "
                                    f"gap_type={gap.gap_type} ok={bool(last_result.get('ok'))}"
                                ),
                            }
                        )
                else:
                    refined_query = (
                        f"{task.text} site:wikidata.org OR site:openstreetmap.org"
                    )
                    attempt = 0
                    for _ in range(config.max_retries):
                        attempt += 1
                        last_result = _search_tavily(
                            refined_query,
                            max_results=config.max_results,
                            timeout=config.timeout_seconds,
                            search_depth=config.search_depth,
                            include_answer=config.include_answer,
                        )
                        if last_result.get("ok"):
                            aggregated_results.extend(last_result.get("results", []))
                            break
                    searches_done += 1
                    for r in last_result.get("results", []):
                        score = r.get("score", 0) if isinstance(r, dict) else 0
                        if score > best_score_seen:
                            best_score_seen = score
                    search_outcomes.append(
                        {
                            "query": last_result.get("query", refined_query),
                            "ok": bool(last_result.get("ok")),
                            "attempt": attempt,
                            "error": None
                            if last_result.get("ok")
                            else str(last_result.get("error", "")),
                        }
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"search={searches_done} query='{refined_query}' "
                                f"ok={bool(last_result.get('ok'))}"
                            ),
                        }
                    )

                if best_score_seen >= 0.5 and len(aggregated_results) >= 2:
                    pass
                elif searches_done < config.max_searches:
                    if gaps and len(gaps) > 1:
                        gap = gaps[1]
                        alt_query = gap.suggested_query
                    elif gaps:
                        alt_query = f"latest news events {task.text}"
                    else:
                        alt_query = f"latest news events {task.text}"
                    attempt = 0
                    for _ in range(config.max_retries):
                        attempt += 1
                        last_result = _search_tavily(
                            alt_query,
                            max_results=config.max_results,
                            timeout=config.timeout_seconds,
                            search_depth=config.search_depth,
                            include_answer=config.include_answer,
                        )
                        if last_result.get("ok"):
                            aggregated_results.extend(last_result.get("results", []))
                            break
                    searches_done += 1
                    for r in last_result.get("results", []):
                        score = r.get("score", 0) if isinstance(r, dict) else 0
                        if score > best_score_seen:
                            best_score_seen = score
                    search_outcomes.append(
                        {
                            "query": last_result.get("query", alt_query),
                            "ok": bool(last_result.get("ok")),
                            "attempt": attempt,
                            "error": None
                            if last_result.get("ok")
                            else str(last_result.get("error", "")),
                        }
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"search={searches_done} query='{alt_query}' "
                                f"ok={bool(last_result.get('ok'))}"
                            ),
                        }
                    )

            search_result = {
                "ok": any(outcome.get("ok") for outcome in search_outcomes),
                "query": task.text,
                "answer": last_result.get("answer"),
                "results": aggregated_results,
            }

            # Normalize sources and extract full content from top sources
            normalized_sources = _normalize_sources(search_result)
            scored_sources = _score_and_filter_sources(normalized_sources)

            # Extract full content from top N sources by composite score
            # Filter out tour/activity sites that return nav content when extracted
            extract_urls = [
                s.url
                for s, _ in scored_sources[: config.extract_max_sources]
                if s.url and not _get_skip_extraction(s.url)
            ]
            extracted_raw: list[dict[str, Any]] = []
            if extract_urls:
                extracted_raw = _extract_full_content(
                    extract_urls,
                    timeout=config.extract_timeout,
                    extract_depth="basic",
                )

            # Build url -> raw_content map for proof point building
            # Only use extracted content if quality >= threshold, else fall back to snippet
            extracted_contents: dict[str, str] = {}
            for item in extracted_raw:
                url = item.get("url", "")
                raw_content = item.get("raw_content", "")
                quality = item.get("quality_score", 0.0)
                if not url:
                    continue
                if quality >= config.extract_quality_threshold and raw_content:
                    extracted_contents[url] = raw_content

            # For URLs not in extracted_contents, fall back to snippet
            for s, _ in scored_sources[: config.extract_max_sources]:
                if s.url and s.url not in extracted_contents:
                    snippet = s.snippet or ""
                    if snippet:
                        extracted_contents[s.url] = snippet[:500]

            synthesis_ok: bool = False
            answer_text: str | None = None
            model_name: str | None = None
            errors: list[GeneralWebSearchErrorModel] = []
            if not search_result.get("ok"):
                errors.append(
                    _normalize_error(
                        str(last_result.get("error", "no successful search"))
                    )
                )
            elif config.answer_model_name:
                answer_dict = _synthesize_answer_with_model(
                    system_prompt=config.answer_system_prompt,
                    instruction=config.answer_instruction,
                    model_name=config.answer_model_name,
                    temperature=config.answer_temperature,
                    query=task.text,
                    result=search_result,
                )
                synthesis_ok = bool(answer_dict.get("ok", False))
                answer_text = answer_dict.get("text")
                model_name = answer_dict.get("model")
                if not synthesis_ok:
                    errors.append(
                        GeneralWebSearchErrorModel(
                            code="answer_error",
                            message=answer_dict.get("error")
                            or "answer synthesis failed",
                        )
                    )

            final_answer, proof_points = _build_answer_with_proof(
                result=search_result,
                sources=normalized_sources,
                synthesis_ok=synthesis_ok,
                answer_text=answer_text,
                config=config,
                query=task.text,
                extracted_contents=extracted_contents,
            )

            content_model = GeneralWebSearchArtifactContentModel(
                task_ref=task.name,
                query=task.text,
                provider="tavily",
                status=_compute_status(
                    result_ok=bool(search_result.get("ok")), synthesis_ok=synthesis_ok
                ),
                attempt=attempt,
                result=GeneralWebSearchResultModel.model_validate(search_result),
                answer=final_answer,
                model=model_name,
                sources=normalized_sources,
                proof_points=proof_points,
                errors=errors,
                config={
                    "provider": config.provider,
                    "max_results": config.max_results,
                    "timeout_seconds": config.timeout_seconds,
                    "max_retries": config.max_retries,
                    "max_searches": config.max_searches,
                    "search_outcomes": search_outcomes,
                    "search_depth": config.search_depth,
                    "include_answer": config.include_answer,
                    "answer_model_name": config.answer_model_name,
                    "extract_max_sources": config.extract_max_sources,
                    "extract_timeout": config.extract_timeout,
                },
            )
            artifact = AgentArtifactModel(
                name=task.name,
                type="general-web-search-result",
                content=content_model.model_dump(mode="json"),
                description=f"General web search result for task '{task.name}'",
            )
            existing.append(artifact)
            messages.append({"role": "user", "content": task.text})
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Task '{task.name}' -> {content_model.status} "
                        f"(searches={searches_done}, errors={len(errors)})"
                    ),
                }
            )

        base_artifacts["general_web_search_agent"] = existing
        return {
            "agent_artifacts": base_artifacts,
            "message_history": _build_message_history(messages, model_ref),
        }

    graph = StateGraph(GeneralWebSearchAgentState)
    graph.add_node("general_web_search_agent", search_node)
    graph.set_entry_point("general_web_search_agent")
    graph.add_edge("general_web_search_agent", END)
    return graph.compile()
