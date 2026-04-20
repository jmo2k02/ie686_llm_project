# Multi-Agent Travel Planning System - Architecture

## System Architecture Overview

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    %% ============================================
    %% NODE TYPE DEFINITIONS
    %% ============================================
    classDef userNode fill:#f8fafc,stroke:#334155,stroke-width:2px,color:#1e293b
    classDef constraintNode fill:#f0f9ff,stroke:#0369a1,stroke-width:2px,color:#0c4a6e
    classDef plannerNode fill:#faf5ff,stroke:#7c3aed,stroke-width:2px,color:#4c1d95
    classDef reviewerNode fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#854d0e
    classDef executorNode fill:#f0fdf4,stroke:#15803d,stroke-width:2px,color:#166534
    classDef searchNode fill:#fff7ed,stroke:#ea580c,stroke-width:1px,color:#9a3412
    classDef artifactNode fill:#f1f5f9,stroke:#64748b,stroke-width:1px,color:#334155
    classDef outputNode fill:#ecfdf5,stroke:#059669,stroke-width:3px,color:#064e3b

    %% ============================================
    %% PHASE 0: CONSTRAINT GATHERING
    %% ============================================
    subgraph PHASE0["PHASE 0: Constraint Gathering"]
        direction LR
        U0["User"]:::userNode
        CI0["Constraint Iteration Agent"]:::constraintNode
        RV0["Reviewer (Constraints)"]:::reviewerNode

        U0 -->|"Initial Prompt<br/>e.g. Barcelona trip"| CI0
        CI0 -->|"Iterate until<br/>constraints clear"| U0
        CI0 -->|"Submit for validation"| RV0
        RV0 -.->|"Revision needed"| CI0
        RV0 -->|"Validated constraints"| ART0
    end

    %% ============================================
    %% ARTIFACT: CONSTRAINTS
    %% ============================================
    ART0{{"Constraints Artifact<br/>- Location: Barcelona<br/>- Time: Jun 15-22<br/>- Budget: 1500 EUR<br/>- Transport: Flight<br/>- Type: Business"}}:::artifactNode

    %% ============================================
    %% PHASE 1: PLANNING
    %% ============================================
    subgraph PHASE1["PHASE 1: Planning"]
        direction LR
        PL["Planner Agent<br/>Task Decomposition"]:::plannerNode
        RV1["Reviewer (Plan Validation)"]:::reviewerNode

        ART0 -->|"Constraints + Chat History"| PL
        PL -->|"Draft Plan Task List"| RV1
        RV1 -.->|"Invalid - retry plan"| PL
        RV1 -->|"Validated Plan<br/>Approved Tasks"| ART1
    end

    %% ============================================
    %% ARTIFACT: PLAN
    %% ============================================
    ART1{{"Plan Artifact<br/>- Search flights (BCN)<br/>- Find hotel (Eixample)<br/>- Find restaurants<br/>- Find attractions<br/>- Route optimization"}}:::artifactNode

    %% ============================================
    %% PHASE 2: EXECUTION → RETAIN ARTIFACTS → ITINERARY → REVIEW
    %% ============================================
    subgraph PHASE2["PHASE 2: Execution, artifact retention & final review"]
        direction TB

        EX["Execution Agent<br/>Orchestrator + decisions"]:::executorNode

        subgraph SPAWN["Search agents (spawned on demand from the plan)"]
            direction LR
            FS["Flight Search"]:::searchNode
            HS["Hotel Search"]:::searchNode
            RS["Restaurant Search"]:::searchNode
            AS["Attraction Search"]:::searchNode
            RC["Route Check"]:::searchNode
        end

        subgraph TYPED_ART["Typed artifacts (one canonical schema per agent)"]
            direction LR
            ARTF{{"Flight options<br/>artifact"}}:::artifactNode
            ARTH{{"Hotel shortlist<br/>artifact"}}:::artifactNode
            ARTR{{"Restaurant picks<br/>artifact"}}:::artifactNode
            ARTA{{"Attraction set<br/>artifact"}}:::artifactNode
            ARTC{{"Route / timing<br/>artifact"}}:::artifactNode
        end

        STORE{{"Retained typed artifacts<br/>(in-memory in minimal setups;<br/>durable store when retries / APIs / audit matter)"}}:::artifactNode

        DRAFT{{"Itinerary artifact (draft)<br/>Merged schedule, primary picks,<br/>alternatives, feasibility notes"}}:::artifactNode

        RV2["Final Reviewer<br/>Itinerary validation"]:::reviewerNode

        ART2{{"Itinerary artifact (validated)<br/>Ready for export / user handoff"}}:::artifactNode
    end

    %% ============================================
    %% OUTPUT
    %% ============================================
    OUTPUT["Validated Travel Itinerary<br/>+ Alternatives + Markdown Report"]:::outputNode

    %% ============================================
    %% CONNECTIONS
    %% ============================================
    ART1 -->|"Approved plan"| EX

    EX -.->|"spawn task(s)<br/>as needed"| FS
    EX -.->|"spawn task(s)<br/>as needed"| HS
    EX -.->|"spawn task(s)<br/>as needed"| RS
    EX -.->|"spawn task(s)<br/>as needed"| AS
    EX -.->|"spawn task(s)<br/>as needed"| RC

    FS --> ARTF
    HS --> ARTH
    RS --> ARTR
    AS --> ARTA
    RC --> ARTC

    ARTF -->|"return artifact"| EX
    ARTH -->|"return artifact"| EX
    ARTR -->|"return artifact"| EX
    ARTA -->|"return artifact"| EX
    ARTC -->|"return artifact"| EX

    EX -->|"retain / optionally<br/>persist typed artifacts"| STORE
    EX -->|"compose draft itinerary<br/>from retained artifacts"| DRAFT
    DRAFT -->|"submit for checks"| RV2
    RV2 -.->|"fails review —<br/>retry with fixes"| EX
    RV2 -->|"passes review"| ART2
    ART2 --> OUTPUT
```

---

## Phase 0: Constraint Gathering (Iteration Loop)

### Purpose
Before deep research begins, the **Constraint-Iteration Agent** works with the user to gather and clarify all requirements. This prevents expensive iterations during the research phase.

### How It Works

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart LR
    classDef userNode fill:#f8fafc,stroke:#334155,stroke-width:2px,color:#1e293b
    classDef constraintNode fill:#f0f9ff,stroke:#0369a1,stroke-width:2px,color:#0c4a6e
    classDef reviewerNode fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#854d0e

    U["User"]:::userNode
    CI["Constraint Agent"]:::constraintNode
    RV["Reviewer"]:::reviewerNode

    U -->|"I want to go to<br/>Barcelona"| CI
    CI -->|"Identify missing<br/>constraints"| U
    U -->|"June 15-22,<br/>budget 1500 EUR"| CI
    CI -->|"Flight preference?"| U
    U -->|"Flight preferred,<br/>flexible times"| CI
    CI -->|"All constraints identified"| RV
```

### Constraint Types

| Category | Examples | Required? |
|----------|----------|-----------|
| **Location** | Destination city, neighborhoods, landmarks | ✅ Yes |
| **Time** | Start date, end date, preferred days/times | ✅ Yes |
| **Budget** | Max cost, currency, spending priorities | Optional |
| **Transport** | Flight, train, car, bus preferences | Optional |
| **Purpose** | Business, vacation, honeymoon, family | Optional |
| **Preferences** | Cuisine types, activity levels, mobility needs | Optional |

### Output Artifact

```json
{
  "location": {
    "destination": "Barcelona",
    "areas": ["Eixample", "Gothic Quarter"]
  },
  "time": {
    "start": "2024-06-15",
    "end": "2024-06-22",
    "flexible": false
  },
  "budget": {
    "max": 1500,
    "currency": "EUR",
    "priority": "accommodation"
  },
  "transport": {
    "type": "flight",
    "departure": "Munich",
    "flexible_times": true
  },
  "purpose": "business_conference",
  "preferences": {
    "cuisine": ["Spanish", "Catalan"],
    "activity_level": "moderate"
  }
}
```

---

## Phase 1: Planning & Validation

### Purpose
The **Planner Agent** decomposes the validated constraints into executable tasks. The **Reviewer Agent** validates the plan to ensure all requirements are covered.

### Task Types

| Task Type | Agent | Output |
|------------|-------|--------|
| `flight_search` | Flight Search Agent | Top 3 options with prices/times |
| `hotel_search` | Hotel Search Agent | Top 3 options with location/amenities |
| `restaurant_search` | Restaurant Agent | Recommendations by cuisine/price |
| `attraction_search` | Attraction Agent | Must-see + hidden gems |
| `route_optimization` | Route Agent | Optimized daily schedule |

### Plan Artifact Example

```json
{
  "plan_id": "plan_2024_001",
  "constraints_ref": "constraints_2024_001",
  "tasks": [
    {
      "id": 1,
      "type": "flight_search",
      "priority": 1,
      "params": {
        "from": "Munich",
        "to": "BCN",
        "date": "2024-06-15",
        "return": "2024-06-22"
      }
    },
    {
      "id": 2,
      "type": "hotel_search",
      "priority": 2,
      "params": {
        "location": "Eixample",
        "dates": "2024-06-15 to 2024-06-22",
        "budget_max": 800
      },
      "depends_on": [1]
    },
    {
      "id": 3,
      "type": "attraction_search",
      "priority": 3,
      "params": {
        "interests": ["architecture", "art", "history"]
      }
    }
  ],
  "review_status": "approved"
}
```

---

## Phase 2: Execution & Final Review

### Execution Agent Role

The **Execution Agent** is the single orchestrator for Phase 2. It reads the approved plan, **spawns only the search agents that are needed** for each task (some may run in parallel when dependencies allow). **Every search agent returns a single, typed artifact** (its own schema and id); execution never treats Phase 2 output as an anonymous blob.

It **retains each typed artifact** at least until the draft itinerary is built (typically in the orchestrator’s memory). **Durable persistence** (versioned records, provenance, ids for partial reruns) is optional for a small demo, but valuable when the final reviewer may loop (so you do not re-hit every search API), when calls are paid or rate-limited, or when you need an audit trail. From whatever is retained it **materializes a draft itinerary artifact**—primary choices, backups, day-by-day timing, and cost notes. **Only that draft itinerary** is handed to the **Final Reviewer**; on failure, the reviewer sends feedback back to execution, which may spawn additional searches or adjust composition before a new revision.

1. **Spawn-on-demand** — instantiate search agents per plan task, not a fixed static pool every time  
2. **Emit typed artifacts** — each agent produces one canonical artifact (flight options, hotel shortlist, …) with normalized fields and source metadata  
3. **Retain artifacts** — keep each typed document until composition (and longer if using a durable store for retries, cost control, or provenance)  
4. **Compose itinerary artifact** — merge retained typed artifacts into one draft document the reviewer can judge  
5. **Decide under conflict** — price vs. time and similar tradeoffs inform picks and alternatives  
6. **Iterate with reviewer** — draft → review → refine until pass or escalation

### Typed artifacts (Phase 2)

Each spawned search agent **must** hand back an artifact document (JSON or equivalent), not only log text or chat. Names are stable so composition (and optional durable storage) can reference them.

| Agent | Artifact | Typical payload |
|-------|----------|-----------------|
| Flight Search | **Flight options artifact** | Ranked itineraries, carriers, price, duration, booking links |
| Hotel Search | **Hotel shortlist artifact** | Ranked properties, nightly rate, area, amenities |
| Restaurant Search | **Restaurant picks artifact** | Ranked venues, cuisine, price band, hours |
| Attraction Search | **Attraction set artifact** | Ranked sights, categories, estimated visit time |
| Route Check | **Route / timing artifact** | Legs, travel times, feasibility flags between pinned locations |

### Decision Logic

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TD
    classDef decisionNode fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#854d0e
    classDef agentNode fill:#f0fdf4,stroke:#15803d,stroke-width:2px,color:#166534
    classDef outputNode fill:#ecfdf5,stroke:#059669,stroke-width:2px,color:#064e3b

    DECISION{"Compare Options"}:::decisionNode
    OPTA["Option A<br/>Cheaper: 150 EUR<br/>Longer: 5h"]:::agentNode
    OPTB["Option B<br/>Expensive: 280 EUR<br/>Faster: 2.5h"]:::agentNode
    OPTC["Option C<br/>Medium: 200 EUR<br/>Balanced: 3.5h"]:::agentNode

    DECISION -->|"Budget Priority"| OPTA
    DECISION -->|"Time Priority"| OPTB
    DECISION -->|"Balance Priority"| OPTC

    OPTA -->|"Selected<br/>Backup: B"| RESULT["Primary +<br/>Alternatives Stored"]:::outputNode
    OPTB -->|"Selected<br/>Backup: C"| RESULT
    OPTC -->|"Selected<br/>Backup: A"| RESULT
```

### Search Agents Detail

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    classDef searchNode fill:#fff7ed,stroke:#ea580c,stroke-width:1px,color:#9a3412
    classDef executorNode fill:#f0fdf4,stroke:#15803d,stroke-width:2px,color:#166534
    classDef artifactNode fill:#f1f5f9,stroke:#64748b,stroke-width:1px,color:#334155
    classDef reviewerNode fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#854d0e

    EX["Execution Agent"]:::executorNode

    subgraph FLIGHT["Flight Search Agent (spawned)"]
        F1["Search API<br/>Google Flights"]
        F2["Parse Results"]
        F3["Rank by<br/>Price / Duration / Stops"]
    end

    subgraph HOTEL["Hotel Search Agent (spawned)"]
        H1["Search API<br/>TripAdvisor"]
        H2["Filter by<br/>Location and Budget"]
        H3["Rank by<br/>Rating / Price / Reviews"]
    end

    subgraph RESTAURANT["Restaurant Agent (spawned)"]
        R1["Search API<br/>TripAdvisor / OpenStreetMap"]
        R2["Filter by<br/>Cuisine and Price Range"]
        R3["Rank by<br/>Rating / Popularity"]
    end

    subgraph ATTRACTION["Attraction Agent (spawned)"]
        A1["Search API<br/>Tavily / OpenStreetMap"]
        A2["Filter by<br/>Category and Location"]
        A3["Rank by<br/>Rating / Accessibility"]
    end

    subgraph ROUTE["Route Check Agent (spawned)"]
        RT1["Build legs<br/>between stops"]
        RT2["Duration +<br/>buffer estimates"]
        RT3["Feasibility +<br/>conflict flags"]
    end

    ARTF{{"Flight options artifact"}}:::artifactNode
    ARTH{{"Hotel shortlist artifact"}}:::artifactNode
    ARTR{{"Restaurant picks artifact"}}:::artifactNode
    ARTA{{"Attraction set artifact"}}:::artifactNode
    ARTC{{"Route / timing artifact"}}:::artifactNode

    STORE{{"Retained typed artifacts<br/>(optional durable store)"}}:::artifactNode
    DRAFT{{"Itinerary artifact (draft)"}}:::artifactNode
    RV["Final Reviewer"]:::reviewerNode

    EX -.->|"execute_task"| FLIGHT
    EX -.->|"execute_task"| HOTEL
    EX -.->|"execute_task"| RESTAURANT
    EX -.->|"execute_task"| ATTRACTION
    EX -.->|"execute_task"| ROUTE

    FLIGHT --> ARTF
    HOTEL --> ARTH
    RESTAURANT --> ARTR
    ATTRACTION --> ARTA
    ROUTE --> ARTC

    ARTF --> EX
    ARTH --> EX
    ARTR --> EX
    ARTA --> EX
    ARTC --> EX

    EX -->|"retain / optionally persist"| STORE
    EX -->|"compose from typed artifacts"| DRAFT
    DRAFT -->|"itinerary review"| RV
```

---

## Final Output Structure

The validated itinerary is delivered as:

### 1. Calendar/Timetable (Primary Output)

```markdown
## Travel Itinerary: Barcelona Business Trip

### Day 1 - June 15 (Monday)
| Time | Activity | Details | Booking |
|------|----------|---------|---------|
| 09:00 | Flight MUC to BCN | LH1134, 2h 15m | Confirmed |
| 12:30 | Check-in Hotel | Hotel Arts, Room 412 | Confirmed |
| 14:00 | Conference Check-in | CCIB, Registration | Required |
| 18:00 | Dinner | Can Culleretes (traditional) | Reservation #1234 |

### Day 2 - June 16 (Tuesday)
| Time | Activity | Details | Booking |
|------|----------|---------|---------|
| 08:00 | Breakfast | Hotel restaurant | Included |
| 09:00 | Conference Day 1 | CCIB Main Hall | Full day |
| 12:30 | Lunch | Taller de Tapas (nearby) | Walk-in |
| 19:00 | Evening | Beach walk, Barceloneta | Free time |

**... continues for duration of trip ...**
```

### 2. Alternatives (Backup Options)

```markdown
## Alternative Options

### Flight Alternatives
| Option | Price | Duration | Times |
|--------|-------|----------|-------|
| Selected | 180 EUR | 2h 15m | 09:00-11:15 |
| Alt 1 | 145 EUR | 4h 30m | 06:30+stop |
| Alt 2 | 220 EUR | 2h 05m | 19:45-21:50 |

### Hotel Alternatives
| Option | Price/night | Rating | Notes |
|--------|-------------|--------|-------|
| Selected | 120 EUR | 4 stars | Arts district |
| Alt 1 | 95 EUR | 3 stars | Gothic Quarter |
| Alt 2 | 150 EUR | 5 stars | Beachfront |
```

### 3. Markdown Report

Full formatted document with:
- Executive summary
- Daily breakdown
- Cost summary
- Important contacts
- Packing recommendations

---

## Agent Communication Protocol

### Artifact Flow

```mermaid
%%{init: {'theme': 'base'}}%%
sequenceDiagram
    participant U as User
    participant CI as Constraint Agent
    participant RV as Constraint Reviewer
    participant PL as Planner Agent
    participant EX as Execution Agent
    participant SA as Search Agents
    participant FR as Final Reviewer

    U->>CI: Initial prompt
    CI->>U: Clarifying questions
    U->>CI: Answers
    CI->>RV: Submit constraints
    RV-->>CI: Revision needed
    CI->>U: Follow-up questions
    U->>CI: More details
    CI->>RV: Resubmit
    RV->>PL: Validated constraints
    
    PL->>RV: Draft plan
    RV-->>PL: Invalid plan
    PL->>RV: Revised plan
    RV->>EX: Approved plan
    
    loop Per plan task (parallel where allowed)
        EX->>SA: Spawn agent / execute_task
        SA-->>EX: Typed agent artifact (e.g. hotel_shortlist)
        EX->>EX: Retain artifact (optionally write durable snapshot)
    end
    EX->>EX: Compose itinerary artifact (draft) from store
    EX->>FR: Submit draft itinerary
    alt Review fails
        FR-->>EX: Feedback + required fixes
        EX->>SA: Retry / spawn follow-up tasks
    else Review passes
        FR->>U: Final validated itinerary
    end
```

### Message Types

| From → To | Message Type | Content |
|------------|--------------|---------|
| User → Constraint | `prompt` | Initial travel request |
| Constraint → User | `clarification` | Questions to user |
| Constraint → Reviewer | `submit` | Final constraints |
| Reviewer → Constraint | `revision` | Changes needed |
| Planner → Reviewer | `plan_draft` | Proposed task list |
| Reviewer → Planner | `plan_feedback` | Approval or changes |
| Execution → Search | `execute_task` | Task with parameters |
| Flight Search → Execution | `flight_options_artifact` | Canonical flight-options document |
| Hotel Search → Execution | `hotel_shortlist_artifact` | Canonical hotel shortlist document |
| Restaurant Search → Execution | `restaurant_picks_artifact` | Canonical restaurant picks document |
| Attraction Search → Execution | `attraction_set_artifact` | Canonical attraction set document |
| Route Check → Execution | `route_timing_artifact` | Legs, times, feasibility flags |
| Execution → Persistence | `artifact_persist` | Optional: versioned snapshots when using a durable store |
| Execution → Reviewer | `itinerary` | Draft itinerary artifact (composed from store) |
| Reviewer → Execution | `review_result` | Validation status |

---

## Review Loop Mechanism

### Retry Logic

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TD
    classDef retryNode fill:#fef2f2,stroke:#dc2626,stroke-width:2px,color:#991b1b
    classDef successNode fill:#ecfdf5,stroke:#059669,stroke-width:2px,color:#064e3b

    START["Start Review"]:::retryNode
    CHECK{"Passes All<br/>Checks?"}:::retryNode
    FAIL["Failed Check<br/>Identify Issues"]:::retryNode
    RETRY["Retry Phase<br/>with Fixes"]:::retryNode
    MAX["Max retries reached?"]:::retryNode
    ESCALATE["Escalate to<br/>User Decision"]:::retryNode
    SUCCESS["All Checks Passed"]:::successNode

    START --> CHECK
    CHECK -->|"Pass"| SUCCESS
    CHECK -->|"Fail"| FAIL
    FAIL --> RETRY
    RETRY --> CHECK
    RETRY -->|"3 attempts"| MAX
    MAX -->|"Exceeded"| ESCALATE
    MAX -->|"Under limit"| CHECK
```

### Review Criteria

| Check | What It Validates |
|-------|-------------------|
| **Constraint Coverage** | All user constraints addressed |
| **Temporal Validity** | Times don't conflict, realistic travel |
| **Budget Compliance** | Total cost ≤ user budget |
| **Completeness** | All days have activities |
| **Feasibility** | Travel times between locations are possible |
| **Alternatives** | Backup options available for key decisions |

---

## System Constants

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Max Retries** | 3 | Prevent infinite loops |
| **Search Results** | Top 3 per category | Balance quality vs. cognitive load |
| **Alternative Options** | 2 per primary | Give user choice without overwhelming |
| **Reviewer Model** | Higher capability | Better validation quality |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Constraint-Iteration before Planner** | Front-loading simplifies later phases; iteration in research mode is complex |
| **Execution Agent makes decisions** | Needs judgment when options conflict (e.g., cheaper flight = longer time) |
| **General Web Search for edge cases** | Handles queries that don't fit specialized agents |
| **Timetable as primary output** | iCal format enables calendar integration |
| **Search agents spawn on demand; one typed artifact each** | Clear contracts per agent (schemas, ids); composition consumes structured documents, not chat. Durable persistence is recommended when reviewer loops or external APIs make re-fetching expensive |
| **3 max retries** | Prevent infinite loops |
| **Always provide alternatives** | User autonomy, handle preferences not explicitly stated |

---

## Legend

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart LR
    classDef userNode fill:#f8fafc,stroke:#334155,stroke-width:2px,color:#1e293b
    classDef constraintNode fill:#f0f9ff,stroke:#0369a1,stroke-width:2px,color:#0c4a6e
    classDef plannerNode fill:#faf5ff,stroke:#7c3aed,stroke-width:2px,color:#4c1d95
    classDef reviewerNode fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#854d0e
    classDef executorNode fill:#f0fdf4,stroke:#15803d,stroke-width:2px,color:#166534
    classDef searchNode fill:#fff7ed,stroke:#ea580c,stroke-width:1px,color:#9a3412
    classDef artifactNode fill:#f1f5f9,stroke:#64748b,stroke-width:1px,color:#334155
    classDef outputNode fill:#ecfdf5,stroke:#059669,stroke-width:3px,color:#064e3b

    U["User"]:::userNode
    CI["Constraint Agent"]:::constraintNode
    PL["Planner Agent"]:::plannerNode
    RV["Reviewer Agent"]:::reviewerNode
    EX["Execution Agent"]:::executorNode
    SA["Search Agent"]:::searchNode
    AR["Artifact"]:::artifactNode
    OUT["Output"]:::outputNode

    U ~~~ CI ~~~ PL ~~~ RV ~~~ EX ~~~ SA ~~~ AR ~~~ OUT
```

**Node Types:**
- **User Node** (gray) - Human actor
- **Constraint Node** (blue) - Constraint gathering phase
- **Planner Node** (purple) - Task decomposition
- **Reviewer Node** (amber) - Validation/approval
- **Executor Node** (green) - Orchestration/execution
- **Search Node** (orange) - Specialized search agents
- **Artifact Node** (slate) - Data/artifacts between phases
- **Output Node** (emerald) - Final validated output

