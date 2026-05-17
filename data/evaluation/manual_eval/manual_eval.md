# Manual Evaluation — Constraint Verdicts

Queries: 1 (query_id=1), 5 (query_id=7), 16 (query_id=16) | Agents: TP = TravelPlanner, BL = Baseline

## Query 1  (query_id_used = 1)

| Constraint | Description | TP Verdict | BL Verdict | Diff-from-Judge (CC) | Notes |
|---|---|---|---|---|---|
| CC-1 | Car/road trip not feasible for island/overseas destinations — flight o… | PASS | PASS | YES |  |
| CC-2 | No key information missing (e.g. accommodation gap) | PASS | PASS |  |  |
| CC-3 | No conflicting transport modes | PASS | PASS |  |  |
| CC-4 | Both outbound and return transport present | PASS | FAIL |  |  |
| CC-5 | Sufficient connection time (≥60 min domestic, ≥90 min international) | NAN | PASS |  |  |
| CC-6 | Accommodation planned for every night | PASS | PASS |  |  |
| CC-7 | Hotel check-in scheduled after arrival transport | PASS | PASS |  |  |
| CC-8 | Restaurants open on planned day and time | PASS | PASS |  |  |
| CC-9 | No repeated restaurant choices | FAIL | PASS |  |  |
| CC-10 | Activities geographically sensible per day | PASS | PASS |  |  |
| CC-11 | Attractions open on planned day and time | PASS | PASS |  |  |
| CC-12 | No repeated attraction choices | PASS | PASS |  |  |
| CC-13 | Realistic travel time between consecutive activities | PASS | PASS |  |  |
| HC-1 | origin | PASS | PASS |  |  |
| HC-2 | destination | PASS | PASS |  |  |
| HC-3 | travel_dates | PASS | PASS |  |  |
| HC-4 | travelers | PASS | PASS |  |  |
| HC-5 | budget | PASS | FAIL |  |  |
| HC-6 | accommodation | PASS | PASS |  |  |
| HC-7 | interests | PASS | PASS |  |  |

## Query 5  (query_id_used = 7)

| Constraint | Description | TP Verdict | BL Verdict | Diff-from-Judge (CC) | Notes |
|---|---|---|---|---|---|
| CC-1 | Car/road trip not feasible for island/overseas destinations — flight o… | PASS | FAIL |  |  |
| CC-2 | No key information missing (e.g. accommodation gap) | PASS | FAIL |  |  |
| CC-3 | No conflicting transport modes | PASS | PASS |  |  |
| CC-4 | Both outbound and return transport present | PASS | PASS |  |  |
| CC-5 | Sufficient connection time (≥60 min domestic, ≥90 min international) | NAN | FAIL |  |  |
| CC-6 | Accommodation planned for every night | PASS | PASS |  |  |
| CC-7 | Hotel check-in scheduled after arrival transport | PASS | PASS |  |  |
| CC-8 | Restaurants open on planned day and time | PASS | FAIL |  |  |
| CC-9 | No repeated restaurant choices | PASS | FAIL |  |  |
| CC-10 | Activities geographically sensible per day | PASS | PASS |  |  |
| CC-11 | Attractions open on planned day and time | PASS | FAIL |  |  |
| CC-12 | No repeated attraction choices | PASS | FAIL |  |  |
| CC-13 | Realistic travel time between consecutive activities | PASS | FAIL |  |  |
| HC-1 | origin | PASS | PASS |  |  |
| HC-2 | destination | PASS | PASS |  |  |
| HC-3 | travel_dates | PASS | PASS |  |  |
| HC-4 | travelers | PASS | PASS |  |  |
| HC-5 | budget | PASS | PASS |  |  |
| HC-6 | accommodation | PASS | PASS |  |  |
| HC-7 | interests | PASS | PASS |  |  |

## Query 16  (query_id_used = 16)

| Constraint | Description | TP Verdict | BL Verdict | Diff-from-Judge (CC) | Notes |
|---|---|---|---|---|---|
| CC-1 | Car/road trip not feasible for island/overseas destinations — flight o… | PASS | FAIL |  |  |
| CC-2 | No key information missing (e.g. accommodation gap) | FAIL | FAIL |  |  |
| CC-3 | No conflicting transport modes | PASS | PASS |  |  |
| CC-4 | Both outbound and return transport present | FAIL | FAIL |  |  |
| CC-5 | Sufficient connection time (≥60 min domestic, ≥90 min international) | MISSING INFO | MISSING INFO |  |  |
| CC-6 | Accommodation planned for every night | FAIL | FAIL |  |  |
| CC-7 | Hotel check-in scheduled after arrival transport | MISSING INFO | MISSING INFO |  |  |
| CC-8 | Restaurants open on planned day and time | PASS | PASS |  |  |
| CC-9 | No repeated restaurant choices | FAIL | PASS |  |  |
| CC-10 | Activities geographically sensible per day | PASS | PASS |  |  |
| CC-11 | Attractions open on planned day and time | MISSING INFO | PASS |  |  |
| CC-12 | No repeated attraction choices | PASS | PASS |  |  |
| CC-13 | Realistic travel time between consecutive activities | PASS | PASS |  |  |
| HC-1 | origin | PASS | PASS |  |  |
| HC-2 | destination | PASS | PASS |  |  |
| HC-3 | travel_dates | FAIL | FAIL |  | no flight back |
| HC-4 | travelers | PASS | PASS |  |  |
| HC-5 | budget | PASS | PASS |  |  |
| HC-6 | accommodation | PASS | PASS |  |  |
| HC-7 | interests | PASS | PASS |  | checked to little HC? |


---

# Manual Evaluation — Slot Spot-Check Verdicts

Format: PASS / MISSING INFO / FAIL / - (not applicable)

## Query 1 — Spot-Check

| Day-Slot | TP Verdict | TP Diff-Judge | BL Verdict | BL Diff-Judge |
|---|---|---|---|---|
| 11 | day_slot
11    MISSING INFO
11    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
11    NO
11    NO
Name: diff_from_llm_judge, dtype: object | MISSING INFO | YES |
| 12 | day_slot
12    MISSING INFO
12    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
12    NO
12    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 13 | day_slot
13    PASS
13    PASS
Name: manual_verdict, dtype: object | day_slot
13    YES
13    YES
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 14 | day_slot
14    PASS
14    PASS
Name: manual_verdict, dtype: object | day_slot
14    YES
14    YES
Name: diff_from_llm_judge, dtype: object | FAIL | YES |
| 15 | day_slot
15    MISSING INFO
15    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
15    NO
15    NO
Name: diff_from_llm_judge, dtype: object | MISSING INFO | YES |
| 16 | day_slot
16    PASS
16    PASS
Name: manual_verdict, dtype: object | day_slot
16    YES
16    YES
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 17 | day_slot
17    PASS
17    PASS
Name: manual_verdict, dtype: object | day_slot
17    NO
17    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 21 | day_slot
21    PASS
21    PASS
Name: manual_verdict, dtype: object | day_slot
21    NO
21    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 22 | day_slot
22    MISSING INFO
22    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
22    YES
22    YES
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 23 | day_slot
23    MISSING INFO
23    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
23    NO
23    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 24 | day_slot
24    PASS
24    PASS
Name: manual_verdict, dtype: object | day_slot
24    NO
24    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 25 | day_slot
25    PASS
25    PASS
Name: manual_verdict, dtype: object | day_slot
25    NO
25    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 26 | day_slot
26    PASS
26    PASS
Name: manual_verdict, dtype: object | day_slot
26    NO
26    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 31 | day_slot
31    PASS
31    PASS
Name: manual_verdict, dtype: object | day_slot
31    NO
31    NO
Name: diff_from_llm_judge, dtype: object | MISSING INFO | NO |
| 32 | day_slot
32    PASS
32    PASS
Name: manual_verdict, dtype: object | day_slot
32    YES
32    YES
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 33 | day_slot
33    PASS
33    PASS
Name: manual_verdict, dtype: object | day_slot
33    YES
33    YES
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 34 | day_slot
34    MISSING INFO
34    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
34    NO
34    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 35 | day_slot
35    MISSING INFO
35    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
35    NO
35    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 36 | day_slot
36    PASS
36    PASS
Name: manual_verdict, dtype: object | day_slot
36    NO
36    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 41 | day_slot
41    PASS
41    PASS
Name: manual_verdict, dtype: object | day_slot
41    NO
41    NO
Name: diff_from_llm_judge, dtype: object | PASS | No |
| 42 | day_slot
42    MISSING INFO
42    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
42    YES
42    YES
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 43 | day_slot
43    MISSING INFO
43    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
43    NO
43    NO
Name: diff_from_llm_judge, dtype: object | MISSING INFO | NO |
| 44 | day_slot
44    PASS
44    PASS
Name: manual_verdict, dtype: object | day_slot
44    NO
44    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 45 | day_slot
45    MISSING INFO
45    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
45    NO
45    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 46 | day_slot
46    PASS
46    PASS
Name: manual_verdict, dtype: object | day_slot
46    NO
46    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 51 | day_slot
51    MISSING INFO
51    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
51    NO
51    NO
Name: diff_from_llm_judge, dtype: object | MISSING INFO | NO |
| 52 | day_slot
52    PASS
52    PASS
Name: manual_verdict, dtype: object | day_slot
52    NO
52    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 53 | day_slot
53    PASS
53    PASS
Name: manual_verdict, dtype: object | day_slot
53    NO
53    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 54 | day_slot
54    MISSING INFO
54    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
54    NO
54    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 55 | day_slot
55    MISSING INFO
55    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
55    NO
55    NO
Name: diff_from_llm_judge, dtype: object | FAIL | Yes |
| 27 | - | - | PASS | NO |
| 47 | - | - | PASS | NO |

## Query 5 — Spot-Check

| Day-Slot | TP Verdict | TP Diff-Judge | BL Verdict | BL Diff-Judge |
|---|---|---|---|---|
| 11 | day_slot
11    PASS
11    PASS
Name: manual_verdict, dtype: object | day_slot
11    NO
11    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 12 | day_slot
12    PASS
12    PASS
Name: manual_verdict, dtype: object | day_slot
12    NO
12    NO
Name: diff_from_llm_judge, dtype: object | MISSING INFO | NO |
| 13 | day_slot
13    PASS
13    PASS
Name: manual_verdict, dtype: object | day_slot
13    NO
13    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 14 | day_slot
14    PASS
14    PASS
Name: manual_verdict, dtype: object | day_slot
14    YES
14    YES
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 21 | day_slot
21    PASS
21    PASS
Name: manual_verdict, dtype: object | day_slot
21    NO
21    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 22 | day_slot
22    PASS
22    PASS
Name: manual_verdict, dtype: object | day_slot
22    NO
22    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 23 | day_slot
23    PASS
23    PASS
Name: manual_verdict, dtype: object | day_slot
23    NO
23    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 24 | day_slot
24    PASS
24    PASS
Name: manual_verdict, dtype: object | day_slot
24    NO
24    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 25 | day_slot
25    PASS
25    PASS
Name: manual_verdict, dtype: object | day_slot
25    NO
25    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 26 | day_slot
26    PASS
26    PASS
Name: manual_verdict, dtype: object | day_slot
26    NO
26    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 27 | day_slot
27    PASS
27    PASS
Name: manual_verdict, dtype: object | day_slot
27    NO
27    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 31 | day_slot
31    PASS
31    PASS
Name: manual_verdict, dtype: object | day_slot
31    NO
31    NO
Name: diff_from_llm_judge, dtype: object | FAIL | YES |
| 32 | day_slot
32    PASS
32    PASS
Name: manual_verdict, dtype: object | day_slot
32    NO
32    NO
Name: diff_from_llm_judge, dtype: object | FAIL | NO |
| 33 | day_slot
33    PASS
33    PASS
Name: manual_verdict, dtype: object | day_slot
33    NO
33    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 34 | day_slot
34    PASS
34    PASS
Name: manual_verdict, dtype: object | day_slot
34    NO
34    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 35 | day_slot
35    PASS
35    PASS
Name: manual_verdict, dtype: object | day_slot
35    NO
35    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 36 | day_slot
36    PASS
36    PASS
Name: manual_verdict, dtype: object | day_slot
36    NO
36    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 37 | day_slot
37    PASS
37    PASS
Name: manual_verdict, dtype: object | day_slot
37    NO
37    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 15 | - | nan | PASS | NO |
| 16 | - | nan | PASS | NO |
| 17 | - | nan | PASS | NO |
| 18 | - | nan | PASS | NO |
| 19 | - | nan | PASS | NO |
| 28 | - | . | PASS | NO |
| 41 | - | nan | FAIL | Yes |
| 42 | - | nan | PASS | NO |
| 43 | - | nan | PASS | NO |
| 44 | - | nan | PASS | NO |
| 45 | - | nan | PASS | NO |
| 46 | - | nan | PASS | NO |
| 47 | - | nan | PASS | NO |

## Query 16 — Spot-Check

| Day-Slot | TP Verdict | TP Diff-Judge | BL Verdict | BL Diff-Judge |
|---|---|---|---|---|
| 11 | day_slot
11    PASS
11    PASS
Name: manual_verdict, dtype: object | day_slot
11    NO
11    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 12 | day_slot
12    MISSING INFO
12    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
12    NO
12    NO
Name: diff_from_llm_judge, dtype: object | FAIL | YES |
| 13 | day_slot
13    PASS
13    PASS
Name: manual_verdict, dtype: object | day_slot
13    NO
13    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 14 | day_slot
14    MISSING INFO
14    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
14    YES
14    YES
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 15 | day_slot
15    PASS
15    PASS
Name: manual_verdict, dtype: object | day_slot
15    NO
15    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 16 | day_slot
16    MISSING INFO
16    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
16    NO
16    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 17 | day_slot
17    PASS
17    PASS
Name: manual_verdict, dtype: object | day_slot
17    NO
17    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 18 | day_slot
18    PASS
18    PASS
Name: manual_verdict, dtype: object | day_slot
18    YES
18    YES
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 21 | day_slot
21    PASS
21    PASS
Name: manual_verdict, dtype: object | day_slot
21    NO
21    NO
Name: diff_from_llm_judge, dtype: object | PASS | Yes |
| 22 | day_slot
22    PASS
22    PASS
Name: manual_verdict, dtype: object | day_slot
22    YES
22    YES
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 23 | day_slot
23    MISSING INFO
23    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
23    NO
23    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 24 | day_slot
24    MISSING INFO
24    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
24    YES
24    YES
Name: diff_from_llm_judge, dtype: object | PASS | YES |
| 25 | day_slot
25    MISSING INFO
25    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
25    NO
25    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 26 | day_slot
26    MISSING INFO
26    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
26    NO
26    NO
Name: diff_from_llm_judge, dtype: object | MISSING INFO | NO |
| 27 | day_slot
27    PASS
27    PASS
Name: manual_verdict, dtype: object | day_slot
27    NO
27    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 28 | day_slot
28    PASS
28    PASS
Name: manual_verdict, dtype: object | day_slot
28    NO
28    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 31 | day_slot
31    MISSING INFO
31    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
31    NO
31    NO
Name: diff_from_llm_judge, dtype: object | FAIL | YES |
| 32 | day_slot
32    PASS
32    PASS
Name: manual_verdict, dtype: object | day_slot
32    YES
32    YES
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 33 | day_slot
33    PASS
33    PASS
Name: manual_verdict, dtype: object | day_slot
33    YES
33    YES
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 34 | day_slot
34    MISSING INFO
34    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
34    YES
34    YES
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 35 | day_slot
35    PASS
35    PASS
Name: manual_verdict, dtype: object | day_slot
35    NO
35    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 36 | day_slot
36    MISSING INFO
36    MISSING INFO
Name: manual_verdict, dtype: object | day_slot
36    NO
36    NO
Name: diff_from_llm_judge, dtype: object | MISSING INFO | NO |
| 37 | day_slot
37    PASS
37    PASS
Name: manual_verdict, dtype: object | day_slot
37    NO
37    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 38 | day_slot
38    PASS
38    PASS
Name: manual_verdict, dtype: object | day_slot
38    NO
38    NO
Name: diff_from_llm_judge, dtype: object | PASS | NO |
| 19 | - | nan | PASS | NO |
| 29 | - | nan | PASS | NO |
| 39 | - | nan | PASS | NO |
| 310 | - | nan | PASS | NO |
