# Query 8 Add Day Rebuild Analysis

## Question

Why did `query_8` have `add_day x60` even though the Tokyo itinerary only covers a 12-day trip?

## Short Answer

`add_day x60` happened because the execution agent rebuilt the whole itinerary five times before the first validator ran.

Each rebuild followed the same pattern:

1. Call `init_plan`, which resets the stored TravelPlan to `0 day(s)`.
2. Recreate the 12-day trip skeleton with `add_day`.
3. Continue editing or searching.
4. Later decide to restart again.

So the count is:

| Calculation | Result |
|---|---:|
| Days per Tokyo plan skeleton | 12 |
| Full pre-validator rebuilds | 5 |
| `add_day` calls | 12 x 5 = 60 |

## Evidence Timeline

| Phase | Time | Tool behavior | Meaning |
|---|---:|---|---|
| Initial plan reset | 13:24:42 | `init_plan` | First empty plan created |
| Rebuild 1 | 13:25:49 | `add_day x12` | First 12-day Tokyo skeleton |
| Plan reset | 13:28:12 | `init_plan` | Existing plan reset to 0 days |
| Rebuild 2 | 13:28:16 | `add_day x12` | Same trip rebuilt |
| Plan reset | 13:28:46 | `init_plan` | Existing plan reset again |
| Rebuild 3 | 13:28:48-13:29:10 | `add_day x12` | Same trip rebuilt in date order |
| Plan reset | 13:37:49 | `init_plan` | Plan reset after more search and edits |
| Rebuild 4 | 13:37:52-13:38:25 | `add_day x12` | Budget-focused version rebuilt |
| Plan reset | 13:41:25 | `init_plan` | Final pre-validator reset |
| Rebuild 5 | 13:41:31 | `add_day x12` | Strict-budget version rebuilt |
| First validation | 13:51:11 | validator attempt 1 failed | Validator ran only after all five rebuilds |

## Rebuild Batches

| Batch | Start Time | End Time | `add_day` Calls | Notes |
|---|---:|---:|---:|---|
| 1 | 13:25:49 | 13:25:49 | 12 | Initial itinerary skeleton; some days added out of chronological order |
| 2 | 13:28:16 | 13:28:16 | 12 | Rebuilt after second `init_plan`; output order was inconsistent for some days |
| 3 | 13:28:48 | 13:29:10 | 12 | Rebuilt sequentially from Sep 20 to Oct 1 |
| 4 | 13:37:52 | 13:38:25 | 12 | Rebuilt with more budget-oriented labels and fallback wording |
| 5 | 13:41:31 | 13:41:31 | 12 | Final pre-validator rebuild titled `Tokyo strict-budget trip` |

## Why The Agent Likely Did This

The trace shows the agent intentionally restarted the plan after discovering that flight cost consumed most of the budget.

The clearest trace message before the final reset says:

> The live flight result makes EUR 9000 a hard cap: flights alone are EUR 7064 for six, leaving EUR 1936 for all lodging, food, local transport, nightlife, and activities. I am rebuilding the stored plan as a strict budget version rather than leaving an over-budget itinerary.

This suggests the agent treated the existing plan as globally invalid and chose a full rebuild instead of localized edits.

## Root Cause Interpretation

| Cause | Evidence | Effect |
|---|---|---|
| `init_plan` remained available during execution | Five `init_plan` calls before first validator | Agent could reset the plan at any time |
| No guard against reinitializing non-empty plans | Each `init_plan` output says plan has `0 day(s)` | Existing itinerary work was discarded |
| Budget pressure from flight search | Flight cost was EUR 7064 for six | Agent decided the whole plan needed restructuring |
| No local repair policy before validation | Rebuild happened before validator attempt 1 | Agent self-corrected by global restart |
| Long itinerary with many dependent slots | 12-day trip, 6 travelers, strict EUR 9000 budget | Restarting looked simpler than patching every affected slot |

## Presentation Takeaway

`add_day x60` is not 60 unique travel days and not just duplicated CSV rows. It is real execution churn:

| What It Looks Like | What It Actually Means |
|---|---|
| 60 day additions | 5 complete 12-day skeleton rebuilds |
| A huge itinerary | Same itinerary repeatedly reset and recreated |
| Validator repair loop | Mostly pre-validator self-repair |
| Search failure | Budget-driven plan reconstruction after flight evidence |

## System Implication

The Travel Agent needs stronger repair control around state-reset tools:

| Improvement | Why It Helps |
|---|---|
| Disable `init_plan` after first successful day creation | Prevents accidental full resets |
| Require explicit reason before `init_plan` on non-empty plans | Makes global rebuilds auditable |
| Prefer patch tools over reset tools during repair | Reduces `add_day` and `add_slot` churn |
| Add a mutation budget | Stops repeated rebuilds before validation |
| Run early budget checks before full itinerary construction | Avoids building an over-budget plan that later gets discarded |
