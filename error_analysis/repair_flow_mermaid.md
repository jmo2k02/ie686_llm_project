# Repair Flow Visualizations

These Mermaid diagrams use conservative syntax for slide tools that are picky about Mermaid parsing. Copy the whole fenced block, starting at `flowchart LR`.

## Marrakech: Broad Repair After One Validator Failure

```mermaid
flowchart LR
V["Validator: senior-friendly trip failed; layover too tiring; peak-heat walking; verify return price"]
SF["search_flights x4"]
SH["search_hotels x3"]
DEL["delete_slot x32"]
INS["insert_slot x27"]
ADD["add_slot x1"]
COST["cost_summary x7"]
DAYS["8 mutated days: Days 1-8"]
OUT["Next validator: PASS; cost 3179 to 2994 EUR"]
V --> SF
V --> SH
SF --> DEL
SH --> INS
DEL --> DAYS
INS --> DAYS
ADD --> DAYS
DAYS --> COST
COST --> OUT
```

## Tokyo Attempt 2: Repair Churn With Successful Outcome

```mermaid
flowchart LR
V["Validator: 4-leg return excessive; food budget unrealistic; nightlife missing; Ghibli booking missing"]
FLIGHT["search_flights x10"]
WEB["search_web x8"]
HOTEL["search_hotels x1"]
DEL["delete_slot x30"]
ADD["add_slot x27"]
INS["insert_slot x3"]
DAYS["9 mutated days: Days 3-11"]
REPEAT["12 repeated day-position groups; max repeat 7x"]
OUT["Next validator: PASS; cost stayed 8619 EUR"]
V --> FLIGHT
V --> WEB
V --> HOTEL
FLIGHT --> DEL
WEB --> ADD
HOTEL --> ADD
DEL --> DAYS
ADD --> DAYS
INS --> DAYS
DAYS --> REPEAT
REPEAT --> OUT
```

## Tokyo Attempt 1: Cost Fixed, But Validation Still Failed

```mermaid
flowchart LR
V["Validator: over budget; impossible return connection; flight cost and routing confused"]
SF["search_flights x3"]
SH["search_hotels x3"]
SW["search_web x1"]
DEL["delete_slot x14"]
INS["insert_slot x9"]
ADD["add_slot x3"]
COST["cost_summary x4"]
DAYS["8 mutated days: 1, 3, 6, 7, 9, 10, 11, 12"]
OUT["Next validator: FAIL; cost 9089 to 8999 EUR"]
V --> SF
V --> SH
V --> SW
SF --> DEL
SH --> ADD
SW --> INS
DEL --> DAYS
INS --> DAYS
ADD --> DAYS
DAYS --> COST
COST --> OUT
```

## Amsterdam Attempt 2: Focused Repair Contrast

```mermaid
flowchart LR
V["Validator: contradictory luggage plan; Day 4 not linear; repeated Hearth restaurant"]
SR["search_restaurants x4"]
SW["search_web x3"]
DEL["delete_slot x3"]
ADD["add_slot x1"]
COST["cost_summary x1"]
DAYS["2 mutated days: Days 3-4"]
OUT["Next validator: PASS; cost stayed 1399 EUR"]
V --> SR
V --> SW
SR --> DEL
SW --> ADD
DEL --> DAYS
ADD --> DAYS
DAYS --> COST
COST --> OUT
```

## General Repair Pattern

```mermaid
flowchart LR
A["Validator rejects itinerary"]
B["Execution agent reads current plan: view_plan"]
C["Search and check tools: flights, hotels, web, route timing"]
D["Mutation burst: delete_slot, insert_slot, add_slot"]
E["Read-only checks: cost_summary, view_plan"]
F["Next validator attempt"]
G["Final plan accepted"]
A --> B
B --> C
C --> D
D --> E
E --> F
F -->|pass| G
F -->|fail| A
E --> C
E --> D
```
