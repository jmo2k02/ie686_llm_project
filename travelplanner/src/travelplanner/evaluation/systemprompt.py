# ─── URL verification ─────────────────────────────────────────────────────────

URL_VERIFICATION_SYSTEM_PROMPT = """
You are verifying factual claims in a travel plan against a real URL that was linked in the plan.

Your task:
1. Read the plan excerpt that mentions the URL.
2. Read the fetched content from the URL.
3. Identify every specific, verifiable factual claim the plan makes about this URL
   (e.g. prices, dates, names, opening hours, addresses, availability).
4. For each claim, check whether the fetched content confirms, contradicts, or is silent on it.

VERDICT RULES:
- PASS: all identifiable claims match the fetched content (or the content is consistent with them).
- FAIL: at least one claim clearly contradicts the fetched content.
- MISSING_INFO: the fetched content was empty, too generic, or could not confirm/deny any claim.

Do not penalise the plan for claims the URL does not address — only FAIL if there is an
explicit contradiction. If the URL simply does not mention a price, that is MISSING_INFO,
not FAIL.

Return ONLY a JSON object — no markdown, no preamble:
{
  "url": "...",
  "verdict": "PASS",
  "reasoning": "step-by-step reasoning...",
  "claims_checked": ["claim 1 text", "claim 2 text"]
}
"""


def build_url_verification_prompt(
    *,
    url: str,
    fetched_title: str,
    fetched_content: str,
    plan_excerpt: str,
) -> str:
    lines = [
        f"URL: {url}",
        f"Page title: {fetched_title or '(unknown)'}",
        "",
        "## FETCHED CONTENT (first 2000 characters)",
        fetched_content[:2000] if fetched_content else "(no content retrieved)",
        "",
        "## PLAN EXCERPT MENTIONING THIS URL",
        plan_excerpt or "(not found in plan)",
    ]
    return "\n".join(lines)


# ─── Constraint judge ─────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """
You are an impartial travel plan evaluator. You operate at temperature=0 for full reproducibility.

IMPORTANT BIAS RULES — read these first:
- Do NOT reward plans for being detailed, well-written, or long. A verbose plan that violates a constraint FAILS. A terse plan that satisfies all constraints PASSES.
- Do NOT penalise a plan for brevity.
- Do NOT infer that missing information is probably fine. Absence of required information is always FAIL or MISSING_INFO.
- If URL verification results are provided, treat them as authoritative ground truth for any factual claim they cover. Do not override them with your own knowledge.
- Do NOT draw on general world knowledge for factual checks not covered by URL verification results.

---

## YOUR EVALUATION PROCEDURE

For EACH constraint, reason step by step before recording your verdict:

1. State the constraint.
2. Identify the relevant part(s) of the travel plan (quote or paraphrase).
3. If URL verification results are available, check whether they bear on this constraint.
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
- **budget**: Maximum total trip cost (transport + accommodation + activities + meals). Sum all costs in the plan. FAIL if the total exceeds the budget, or if costs are absent. Cross-reference URL verification results for price confirmation.
- **accommodation**: Hotel type and preferences. Check that the booked accommodation matches the type and any stated preferences (location, amenities, etc.). Cross-reference URL verification results if a hotel URL was verified.
- **transport**: Required transport mode. FAIL if the plan uses a mode the user excluded. Mark NA if no restriction was stated.
- **interests**: Traveler interests and pace preference. Check that activities broadly reflect the stated interests and pace.

Mark a hard constraint **NA** only if that category was explicitly set to "not specified" or marked as skipped.

---

## COMMONSENSE CONSTRAINTS

These apply to every reasonable travel plan regardless of whether the user stated them. NEVER mark a commonsense constraint NA — if you cannot evaluate it, use MISSING_INFO.

Each commonsense constraint is a plain-text statement. Evaluate each one literally. Cross-reference URL verification results where relevant.

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


def _format_url_verification_context(url_verifications: list[dict]) -> str:
    if not url_verifications:
        return ""
    lines = ["## URL VERIFICATION RESULTS (treat as ground truth for covered claims)"]
    for uv in url_verifications:
        lines.append(
            f"\n[{uv['verdict']}] {uv['url']}"
        )
        if uv.get("fetched_title"):
            lines.append(f"  Page: {uv['fetched_title']}")
        lines.append(f"  {uv['reasoning'][:300]}")
        for claim in uv.get("claims_checked", []):
            lines.append(f"  • {claim}")
    return "\n".join(lines)


def build_judge_user_prompt_hc(
    *,
    plan_text: str,
    hard_constraints: list[dict],
    url_verifications: list[dict],
) -> str:
    lines: list[str] = []

    uv_section = _format_url_verification_context(url_verifications)
    if uv_section:
        lines.append(uv_section)
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
    url_verifications: list[dict],
    hard_constraints: list[dict] | None = None,
) -> str:
    lines: list[str] = []

    uv_section = _format_url_verification_context(url_verifications)
    if uv_section:
        lines.append(uv_section)
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
