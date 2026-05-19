# Error Analysis Key Takeaways

## Overall Conclusion

The Travel Agent is good at building structured travel plans, but it needs stronger evidence gating and repair control.

Without those controls, it can either over-edit plans that are already close to valid or submit plausible plans that fail external verification.

## Main Takeaways

| Takeaway | Meaning |
|---|---|
| Travel Agent usually satisfies hard constraints | It often gets destination, dates, travelers, transport, and budget broadly right. |
| Failures mostly come from repair and evidence, not search alone | Search often finds useful information, but weak evidence or missing verification can survive into the final plan. |
| Internal validator and external scorecard are misaligned | The validator checks whether the itinerary is reasonable for the user; the scorecard checks every generic benchmark field and evidence requirement. |
| Good final scores can hide unstable execution | `query_8` scored 90.5%, but used 18,145,946 tokens, 536 tool calls, and rebuilt the plan 5 times before first validation. |
| Low final scores can happen without loops | `query_16` had only 65 tool calls and 0 loop indicators, but failed because key fields and evidence were not explicit enough. |
| `init_plan` is dangerous during repair | In `query_8`, repeated `init_plan` caused `add_day x60`: five full 12-day rebuilds. |
| Evidence links are a major bottleneck | Google Maps, Tripadvisor, Rome2Rio, Turbopass, and incomplete official pages often failed rationale verification. |
| Rationale grounding is often weaker than hard-constraint performance | Many plans pass hard constraints but lose points because links do not verify exact prices, hours, routes, or slot claims. (one click away problems) |
| Simple trips perform best | `query_7` did well because it was short, explicit, business-focused, and easy to verify. |
| The system needs controlled repair, not just more search | The next improvement should be a scoped repair layer: diagnose, patch locally, verify locally, then validate. |

## Representative Cases

| Query | What It Shows | Key Evidence |
|---|---|---|
| `query_7` Frankfurt to New York | Efficient high-quality success | 95.0% scorecard, 100.0% hard pass, 94.4% rationale pass, 971,916 tokens, 56 tool calls |
| `query_8` Tokyo group trip | Good score but unstable execution | 90.5% scorecard, 18,145,946 tokens, 536 tool calls, 5 full pre-validator rebuilds |
| `query_16` Rome already at destination | Smooth trace but weak final evaluation | 44.4% scorecard, 40.0% hard pass, 65 tool calls, 0 loop indicators |
| `query_11` Cape Town | Long-plan evidence failure | 76.2% scorecard, 32.7% rationale pass, 62 missing rationale checks |
| `query_9` Maldives | High mutation without strong score | 76.2% scorecard, 216 mutation calls, 10,242,724 tokens |
| `query_21` Ibiza | Mixed-group tradeoff and weak evidence | 81.0% scorecard, 87.5% hard pass, 39.2% rationale pass |

## What Query 8 Revealed

`query_8` is the clearest trace-instability example.

The final answer scored reasonably well, but the process was expensive and unstable:

| Metric | Value |
|---|---:|
| Scorecard pass rate | 90.5% |
| Hard-constraint pass rate | 100.0% |
| Rationale pass rate | 60.4% |
| Real tool calls | 536 |
| TravelPlan mutation calls | 436 |
| Total tokens | 18,145,946 |
| Validation attempts | 3 |

The biggest issue was repeated full-plan reconstruction:

| Behavior | Meaning |
|---|---|
| `init_plan x5` | The agent reset the TravelPlan five times before first validation. |
| `add_day x60` | The agent rebuilt the same 12-day trip skeleton five times. |
| Broad mutation after validator feedback | Repairs touched many days rather than only affected slots. |

This shows that a high final score does not necessarily mean the graph worked efficiently.

## What Query 16 Revealed

`query_16` is the clearest final-answer failure example.

The internal validator passed the plan, but the external scorecard failed it heavily.

| Failure Area | What Happened |
|---|---|
| Date coverage | The user said July 5 to July 8, but the final plan covered July 5 to July 7. |
| Budget | Slot costs were below EUR 400, but day totals were missing and budget proof was weak. |
| Interests | The scorecard expected Colosseum and Trastevere, but the plan focused on other Rome history and hidden-gem activities. |
| Accommodation | The user said no hotel was needed, but generic scorecard checks still expected accommodation handling. |
| Transport | The user was already in Rome, but generic scorecard checks still expected outbound and return transport. |
| Evidence | 10 rationale checks were missing because links did not expose enough retrievable evidence. |

This shows that the internal validator and external scorecard are not evaluating the same thing.

## What Query 7 Revealed

`query_7` is the clean positive contrast case.

It worked well because it was short, explicit, and easy to verify:

| Strength | Evidence |
|---|---|
| Clear hard constraints | Frankfurt origin, New York destination, business hotel, flights, and dates all passed. |
| Simple trip structure | 3 days, one hotel, outbound and return flights, client dinners. |
| Strong evidence grounding | 17 of 18 rationale checks passed. |
| Low execution cost | 56 tool calls and 971,916 tokens. |
| No rebuild churn | Only one `init_plan`, three `add_day` calls, and one validator attempt. |

The only rationale issue was STK Midtown: the official website links did not expose enough fetched content to verify exact address, hours, pricing, or private dining details.

## Recommended System Improvements

| Improvement | Why It Matters |
|---|---|
| Lock `init_plan` after first successful `add_day` | Prevents repeated full-plan resets like `query_8`. |
| Add a repair planner before mutation | Converts validator feedback into a concrete checklist before editing. |
| Patch only affected slots | Avoids broad rewrites across unrelated days. |
| Add a mutation budget | Stops repeated `delete_slot`, `insert_slot`, and `add_slot` churn. |
| Add evidence gates | Prevents unsupported links from being copied into final slots. |
| Run local checks after each patch | Cost, route timing, overlap, and opening-hour checks should happen before validator reruns. |
| Align validator with scorecard expectations | The internal validator should explicitly check benchmark-style fields such as date coverage, transport, accommodation, and evidence retrievability. |

## How The Ideal Repair Flow Works

The ideal repair flow is designed to prevent the two main failure types found in the analysis:

1. Full-plan rebuild churn, as seen in `query_8`.
2. Plausible but under-verified final answers, as seen in `query_16`.

The repair flow is:

**diagnose -> scope -> patch locally -> verify locally -> validate -> escalate only if necessary**

Raw diagram file: `error_analysis/ideal_repair_pattern.mmd`

### Step-by-Step Pattern

| Step | What Happens | Why It Helps |
|---|---|---|
| 1. Detect issue | A validator, local check, or self-check finds a problem. | Starts repair from a concrete failure instead of vague dissatisfaction. |
| 2. Classify issue type | The system labels the issue as budget, route, evidence, timing, date coverage, opening hours, accommodation, transport, or missing requirement. | Different issues require different tools; this prevents random broad edits. |
| 3. Identify affected slots and days | The system maps the issue to exact days and slots. | Keeps the repair local. A flight problem should not rewrite unrelated meals. |
| 4. Create repair checklist | The validator feedback becomes a short actionable list. | Converts broad feedback into specific edits. |
| 5. Choose smallest safe patch | The agent selects the least disruptive edit set. | Prevents full-plan resets when a local change is enough. |
| 6. Search only missing evidence | The agent searches only for facts needed by the patch. | Reduces token usage and avoids distracting new information. |
| 7. Patch affected slots | The agent uses `delete_slot`, `insert_slot`, or `add_slot` only on affected slots. | Keeps itinerary structure stable. |
| 8. Run local checks | Cost, route timing, overlaps, dates, opening hours, and evidence retrievability are checked before validator rerun. | Catches obvious repair mistakes early. |
| 9. Submit to validator | Only after local checks pass, the plan goes back to the validator. | Validator is used as a final gate, not as the only debugging tool. |
| 10. Escalate if needed | If local patches fail repeatedly, rebuild one affected day or section. | Escalation is controlled and scoped. |
| 11. Full rebuild last | `init_plan` is only allowed if the plan is structurally unrecoverable. | Prevents `query_8` style `init_plan x5` and `add_day x60`. |

### How It Would Fix Query 8

`query_8` repeatedly rebuilt the whole Tokyo itinerary after discovering budget and flight issues.

The ideal flow would behave differently:

| Observed Query 8 Behavior | Ideal Repair Behavior |
|---|---|
| Agent called `init_plan` five times before first validator. | `init_plan` is locked after the first successful `add_day`. |
| Agent rebuilt all 12 days five times. | Agent patches only flights, lodging, food budget, and affected nightlife slots. |
| `add_day x60` appeared before first validation. | `add_day x12` appears once; later changes use slot-level patches. |
| Budget issue caused global reconstruction. | Budget issue triggers `cost_summary`, cheaper flight/lodging search, and targeted replacement. |
| Return-flight issue caused broad day edits. | Return-flight issue patches only departure/return transport slots and connected airport buffers. |
| Validator feedback led to 60 mutations across 9 days. | Repair checklist limits edits to affected days unless local checks fail. |

The key rule is:

**Do not reset the whole plan when the problem is local.**

### How It Would Help Query 16

`query_16` did not fail because of loop churn. It failed because the final answer did not explicitly satisfy the external scorecard and evidence checks.

The ideal repair flow would help by adding local verification before final validation:

| Query 16 Failure | Ideal Repair Check |
|---|---|
| Date span looked incomplete to the scorecard. | Date coverage check confirms whether July 8 must appear explicitly. |
| User said no hotel needed, but scorecard expected accommodation handling. | Missing-field check adds an explicit note: hotel already arranged, no accommodation needed. |
| User was already in Rome, but scorecard expected outbound/return transport. | Transport check adds explicit note: no outbound/return transport required because user is already at destination. |
| Budget proof was weak. | Cost check computes and displays total trip cost under EUR 400. |
| Evidence links were not retrievable. | Evidence gate rejects Google Maps or pages that do not expose price, hours, or location details. |
| Opening hours had missing information. | Opening-hours check forces better sources or marks uncertainty clearly before final plan. |

For `query_16`, the ideal repair layer would not rebuild the itinerary. It would add missing explicit fields and replace weak evidence links.

### Important Guardrails

| Guardrail | Rule |
|---|---|
| `init_plan` lock | Disable `init_plan` after the first successful `add_day`, unless explicit escalation is approved. |
| Patch budget | Stop after repeated edits to the same day or slot position. |
| Locality rule | Only edit slots directly connected to the issue. |
| Evidence gate | Do not keep a slot if the source cannot verify the claim. |
| Cost gate | Run `cost_summary` after budget-related edits. |
| Route gate | Run route checks after transport changes. |
| Date coverage gate | Confirm the final plan explicitly covers the required dates or explains why not. |
| Scorecard-alignment gate | Check generic benchmark expectations before final submission. |

### Why This Pattern Is Better

| Current Failure Mode | Ideal Repair Benefit |
|---|---|
| Agent over-edits when uncertain. | Repairs are scoped to concrete issues. |
| Agent rebuilds entire plans. | Full rebuilds require escalation. |
| Weak links survive into final answer. | Evidence gate blocks unverifiable claims. |
| Validator misses scorecard-specific fields. | Local scorecard-alignment checks catch missing explicit fields. |
| Token usage explodes. | Narrow searches and local patches reduce repeated work. |

The ideal repair pattern turns repair from open-ended mutation into controlled debugging.

## Best One-Sentence Summary

The Travel Agent can produce strong structured itineraries, but the remaining errors come from uncontrolled repair behavior and insufficient evidence verification rather than from a simple lack of search ability.
