from datetime import datetime

# ─── Markdown → TravelPlan structured extraction ─────────────────────────────

MARKDOWN_TO_TRAVELPLAN_SYSTEM_PROMPT = """
You convert a free-form markdown travel itinerary into a structured TravelPlan.

The output schema (Pydantic):
- TravelPlan:
    title: optional string
    days: list[Day]
- Day:
    index: 1-based day number
    calendar_date: ISO date (YYYY-MM-DD) if known, else null
    label: optional short label (e.g. "Arrival", "Departure", city name)
    slots: list[Slot]
- Slot:
    name: short label (e.g. "Breakfast at Café Iruña", "Flight MUC→BCN")
    description: free text expanding on the slot (1–2 sentences); copy any factual
                 rationale (claimed price, hours, vendor, etc.) from the markdown here
    start_time: ISO 8601 datetime (YYYY-MM-DDTHH:MM:SS). If the markdown only gives a
                time-of-day, combine it with the day's calendar_date.
    end_time: ISO 8601 datetime strictly after start_time. If the markdown only gives
              a single time, use a sensible default duration:
                - meal: +60 minutes
                - attraction / leisure: +90 minutes
                - lodging: until 09:00 next day if check-in; otherwise +60 min
                - transport: +120 minutes when arrival time is unknown
                - other: +60 minutes
    category: one of "meal", "attraction", "transport", "lodging", "leisure", "other"
    location: city / neighborhood / address as stated in the markdown, or null
    cost: numeric EUR amount when stated, otherwise null. Convert currencies if the
          markdown gives a price with currency: prefer EUR, but if only a non-EUR
          number is given, still capture the number and rely on the surrounding
          context (do NOT invent FX rates).
    links: list of URLs the markdown attaches to this slot (booking page, official
           site, ticket page). Use [] when no link is present.
    notes: optional free-form notes (e.g. "estimate", "verify availability")

RULES:
1. Preserve EVERY scheduled item in the markdown as a Slot. Do not collapse or
   summarize multiple distinct items into one slot.
2. Do NOT invent items, prices, times, or links that are not present in the markdown.
3. Day indices are 1-based and strictly increasing. Slot start_time must be strictly
   before end_time. Slots within a day must not overlap — if the markdown is
   ambiguous, push later slots forward by a few minutes to break ties.
4. If the markdown contains a cost summary at the end, do NOT model it as a slot.
5. If a day in the markdown only contains a heading and no scheduled items, emit a
   Day with index, calendar_date (if known), label, and an empty slots list.
6. Return ONLY the JSON object matching TravelPlan — no markdown, no preamble.
""".strip()


def build_markdown_to_travelplan_prompt(plan_markdown: str) -> str:
    return (
        "Convert the following travel itinerary markdown into a TravelPlan JSON "
        "object matching the schema in the system prompt.\n\n"
        "## ITINERARY MARKDOWN\n"
        f"{plan_markdown}"
    )


# ─── Slot rationale verification ─────────────────────────────────────────────

EVAL_RATIONALE_SYSTEM_PROMPT = """
You verify the factual rationale of a SINGLE slot in a travel plan against retrieved
web evidence. You operate at temperature=0 for full reproducibility.

INPUT YOU RECEIVE:
- The slot's structured fields (name, description, location, time window, category,
  cost, links, notes).
- Evidence: either (a) the fetched content of one or more URLs that the slot itself
  pointed at, or (b) results of a Tavily web search built from the slot's claims.

YOUR JOB:
1. Identify every specific, VERIFIABLE factual claim the slot makes
   (venue name, address/neighborhood, opening hours or time window, price,
   transport mode, distance/duration, named operator, etc.). Subjective wording
   ("a charming café") is not a claim.
2. Check each claim against the evidence. Quote the relevant evidence fragment in
   your reasoning.
3. Decide a single overall verdict for the slot.

VERDICT RULES:
- PASS: every verifiable claim is supported by, or consistent with, the evidence.
- FAIL: at least one claim is contradicted by the evidence.
- MISSING_INFO: the evidence is empty, generic, or silent on the claims so neither
  PASS nor FAIL can be supported.

IMPORTANT:
- A claim that the evidence simply does not mention is NOT a contradiction — that
  is MISSING_INFO, not FAIL.
- Do NOT use background world knowledge to override the evidence.
- If you were given the slot's own links, treat them as authoritative for claims
  about that venue.
- If you were given web-search results (no slot link), treat the top-tier sources
  (.gov, .edu, wikipedia, official tourism boards, official venue sites) as the
  most authoritative.

Return ONLY a JSON object matching this schema:
{
  "verdict": "PASS" | "FAIL" | "MISSING_INFO",
  "reasoning": "step-by-step reasoning quoting relevant evidence",
  "claims_checked": ["claim 1", "claim 2", ...]
}
"""


def _format_slot_block(
    *,
    day_index: int,
    slot_position: int,
    name: str,
    description: str,
    location: str | None,
    category: str,
    start_time: str,
    end_time: str,
    cost: float | None,
    links: list[str],
    notes: str | None,
) -> str:
    lines = [
        f"Day {day_index}, slot {slot_position}",
        f"  name: {name}",
        f"  category: {category}",
        f"  time: {start_time} → {end_time}",
        f"  location: {location or '(unspecified)'}",
        f"  cost: {('€' + format(cost, '.2f')) if cost is not None else '(unspecified)'}",
        f"  description: {description or '(none)'}",
        f"  notes: {notes or '(none)'}",
    ]
    if links:
        lines.append("  links:")
        for url in links:
            lines.append(f"    - {url}")
    else:
        lines.append("  links: (none)")
    return "\n".join(lines)


def build_rationale_verification_prompt(
    *,
    slot_block: str,
    source_type: str,
    evidence_blocks: list[str],
) -> str:
    """Build the user prompt for the slot rationale verifier.

    Args:
        slot_block: human-readable rendering of the slot's fields.
        source_type: "link" or "web_search".
        evidence_blocks: each block is a self-contained chunk (URL + title +
            fetched content, or a search-snippet listing).
    """
    lines: list[str] = ["## SLOT UNDER REVIEW", slot_block, ""]
    if source_type == "link":
        lines.append("## EVIDENCE — fetched from the slot's own link(s)")
    elif source_type == "web_search":
        lines.append("## EVIDENCE — Tavily web search results (no slot link provided)")
    else:
        lines.append("## EVIDENCE")
    if not evidence_blocks:
        lines.append("(no evidence retrieved)")
    for i, block in enumerate(evidence_blocks, start=1):
        lines.append("")
        lines.append(f"### Source {i}")
        lines.append(block)
    return "\n".join(lines)


# ─── Constraint judge ─────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = f"\nToday's date (ddmmyyyy): {datetime.now().strftime('%d%m%Y')}\n" + """
You are an impartial travel plan evaluator. You operate at temperature=0 for full reproducibility.

IMPORTANT BIAS RULES — read these first:
- Do NOT reward plans for being detailed, well-written, or long. A verbose plan that violates a constraint FAILS. A terse plan that satisfies all constraints PASSES.
- Do NOT penalise a plan for brevity.
- Do NOT infer that missing information is probably fine. Absence of required information is always FAIL or MISSING_INFO.
- If per-slot rationale verification results are provided, treat them as authoritative ground truth for any factual claim they cover. Do not override them with your own knowledge.
- Do NOT draw on general world knowledge for factual checks not covered by rationale verification results.

---

## YOUR EVALUATION PROCEDURE

For EACH constraint, reason step by step before recording your verdict:

1. State the constraint.
2. Identify the relevant part(s) of the travel plan (quote or paraphrase).
3. If rationale verification results are available, check whether they bear on this constraint.
4. Determine whether the plan satisfies or violates the constraint, or whether information is absent.
5. Record your verdict.

Work through ALL constraints in the order given before writing your final JSON output.

---

## HARD CONSTRAINT CATEGORIES

Hard constraints come from the user's original request. Each is expressed as "category: value".
Evaluate each based on what its category requires:

- **destination**: Where the traveler is going. Check that the plan is set in the stated destination.
- **origin**: Where the traveler departs from. Check that outbound transport leaves from the origin.
- **travel_dates**: Start and end dates. Check that outbound departs on start date, return departs on end date, and the number of days matches.
- **travelers**: Number and type of travelers (adults, children). Check that accommodation capacity and activities are suitable for this group size.
- **budget**: Maximum total trip cost (transport + accommodation + activities + meals). Sum all costs in the plan. FAIL if the total exceeds the budget, or if costs are absent. Cross-reference rationale verification results for price confirmation.
- **accommodation**: Hotel type and preferences. Check that the booked accommodation matches the type and any stated preferences (location, amenities, etc.). Cross-reference rationale verification results if a hotel slot was verified.
- **transport**: Required transport mode. FAIL if the plan uses a mode the user excluded. Mark NA if no restriction was stated.
- **interests**: Traveler interests and pace preference. Check that activities broadly reflect the stated interests and pace.

Mark a hard constraint **NA** only if that category was explicitly set to "not specified" or marked as skipped.

---

## COMMONSENSE CONSTRAINTS

These apply to every reasonable travel plan regardless of whether the user stated them. NEVER mark a commonsense constraint NA — if you cannot evaluate it, use MISSING_INFO.

Each commonsense constraint is a plain-text statement. Evaluate each one literally. Cross-reference rationale verification results where relevant.

---

## VERDICT DEFINITIONS

- **PASS**: The plan explicitly satisfies the constraint.
- **FAIL**: The plan explicitly violates it, OR required information is present but incorrect.
- **MISSING_INFO**: The plan lacks information needed to evaluate this constraint. State exactly what is missing. Counts as FAIL in scoring.
- **NA**: Hard-constraint category not part of the original query. ONLY valid for hard constraints.

---

## OUTPUT FORMAT

Return ONLY a JSON object. No markdown, no preamble, no text outside the JSON.

IMPORTANT: use the EXACT constraint IDs from the list in the user message (e.g. "HC-1", "CC-7").
Do NOT invent IDs or copy IDs from this example — use whatever IDs appear in the user message.

{
  "verdicts": [
    {"id": "HC-1", "verdict": "PASS", "reasoning": "step-by-step reasoning..."},
    {"id": "CC-3", "verdict": "FAIL", "reasoning": "step-by-step reasoning..."}
  ]
}

The "verdicts" array must contain exactly one entry per constraint, in the same order as listed.
"""


def _format_rationale_verification_context(rationale_verifications: list[dict]) -> str:
    if not rationale_verifications:
        return ""
    lines = ["## SLOT RATIONALE VERIFICATION RESULTS (treat as ground truth for covered claims)"]
    for rv in rationale_verifications:
        header = (
            f"\n[{rv['verdict']}] Day {rv['day_index']}, slot {rv['slot_position']} — "
            f"{rv['slot_name']}  (source: {rv['source_type']})"
        )
        lines.append(header)
        for url in rv.get("source_urls", []) or []:
            lines.append(f"  url: {url}")
        reasoning = rv.get("reasoning", "")
        if reasoning:
            lines.append(f"  {reasoning[:300]}")
        for claim in rv.get("claims_checked", []) or []:
            lines.append(f"  • {claim}")
    return "\n".join(lines)


def build_judge_user_prompt_hc(
    *,
    plan_text: str,
    hard_constraints: list[dict],
    rationale_verifications: list[dict],
) -> str:
    lines: list[str] = []

    rv_section = _format_rationale_verification_context(rationale_verifications)
    if rv_section:
        lines.append(rv_section)
        lines.append("")

    lines.append("## HARD CONSTRAINTS TO EVALUATE (one per category)")
    for i, c in enumerate(hard_constraints, start=1):
        text = c.get("text", "")
        skipped = c.get("user_skipped", False)
        suffix = "  [user_skipped → mark NA]" if skipped else ""
        lines.append(f"HC-{i}: {text}{suffix}")
    lines.append("")

    lines.append("## TRAVEL PLAN")
    lines.append(plan_text)

    return "\n".join(lines)


def build_judge_user_prompt_cc(
    *,
    plan_text: str,
    commonsense_constraints: list[dict],
    rationale_verifications: list[dict],
    hard_constraints: list[dict] | None = None,
) -> str:
    lines: list[str] = []

    rv_section = _format_rationale_verification_context(rationale_verifications)
    if rv_section:
        lines.append(rv_section)
        lines.append("")

    if hard_constraints:
        lines.append(
            "## TRIP CONTEXT — HARD CONSTRAINTS (reference only, do not evaluate these)"
        )
        for i, c in enumerate(hard_constraints, start=1):
            lines.append(f"HC-{i}: {c.get('text', '')}")
        lines.append("")

    lines.append("## COMMONSENSE CONSTRAINTS TO EVALUATE")
    for i, c in enumerate(commonsense_constraints, start=1):
        lines.append(f"CC-{i}: {c.get('text', '')}")
    lines.append("")

    lines.append("## TRAVEL PLAN")
    lines.append(plan_text)

    return "\n".join(lines)
