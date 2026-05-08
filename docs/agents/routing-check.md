# Routing check: addresses → clusters + travel-time graph

**Goal:** You pass a JSON list of **addresses** (optional labels). The executor returns a **`place_distance_graph`** artifact: **walk clusters**, **hub-to-hub legs** (bicycle, transit, drive — never walking between hubs), and **directed stop→stop edges** so you can look up **distance and duration from A to B**.

**Who decides walk pockets:** The **planner** picks `cluster_context` on the `routing-check` task (`dense_urban` | `mixed` | `sparse`). The places file stays **addresses only**; clustering thresholds follow that preset. Execution is **deterministic** (code + Google APIs), not an LLM.

**LangGraph:** **`travelplanner.integrations.routing_check_agent`** chooses between **`single_od`** and **`place_graph_file`** (with a structured LLM decision step when ambiguous), executes via the same deterministic path as **`execute_routing_check_task`**, then validates artifacts. For **many addresses** plus **LLM-selected** `cluster_context` when omitted (via an inner cluster-context step inside that graph), **`travelplanner.agents.routing_agent`** exposes **`build_routing_graph`** for the places-only clustering flow. Scripts can also invoke **`execute_routing_check_task`** directly with **`kind`**: **`single_od`** or **`place_graph_file`** JSON.

---

## 1. Places file (input)

Minimal shape: **`stops`** with **`address`** per row. Omit `cluster_context` here when the planner sets it on the task.

```json
{
  "stops": [
    { "address": "Carrer de la Guardia, 14, 08002 Barcelona, Spain", "label": "Hotel" },
    { "address": "Palau Nacional, Parc de Montjuïc, 08038 Barcelona, Spain", "label": "MNAC" }
  ]
}
```

- Optional per stop: `label` / `name` (display). If you omit `id`, stops become `stop_0`, `stop_1`, … after geocoding.
- Optional in file: `"cluster_context": "dense_urban" | "mixed" | "sparse"` — defaults to `mixed` if missing or invalid. **Task-level `cluster_context` overrides the file.**

---

## 2. Task the planner emits

```json
{
  "kind": "place_graph_file",
  "places_json_path": "examples/routing_check/inputs/input_barcelona_mixed.json",
  "cluster_context": "dense_urban"
}
```

Executor resolves `places_json_path` relative to `places_json_base_dir` (usually the `travelplanner` working directory). Pick **`dense_urban`** for tight city cores, **`sparse`** when stops are spread out, **`mixed`** otherwise.

---

## 3. What you get (artifact)

`AgentArtifactModel` with `type: "place_distance_graph"` and `content` matching **`PlaceDistanceGraphModel`** (`schema_version` e.g. `1.5`). Use these top-level keys:

| Key | Use |
|-----|-----|
| **`places`** | Resolved coordinates, `cluster_id`, `cluster_hub_id` per stop. |
| **`clusters`** | Members per walk pocket and each pocket’s **hub** (medoid stop id). |
| **`hub_hub_legs`** | Every ordered hub pair × **BICYCLE**, **TRANSIT** (if enabled for the build), **DRIVE** — `distance_meters`, `duration_seconds`, `duration_minutes`. |
| **`edges`** | Every ordered stop pair A≠B: default total duration/distance; `quality` is `haversine_walk` (same cluster) or `hub_chain` (walk to hub → one matrix mode → walk from hub). See `primary_hub_travel_mode` and `legs`. |
| **`inter_cluster_transit`** | Optional richer transit options between hub **addresses** when the build adds them. |
| **`geojson`** | Points (and hub legs) for map viewers. |

**A→B lookup:** Find the `edges` row with `from_place_id` / `to_place_id`, or for mode-specific hub legs use `PlaceDistanceGraphModel.hub_hub_leg(from_hub_id, to_hub_id, "BICYCLE" | "TRANSIT" | "DRIVE")` on parsed content.

---

## 4. Trimmed output example

After geocoding and a build, `content` looks conceptually like this (values illustrative):

```json
{
  "schema_version": "1.5",
  "places": [
    {
      "id": "stop_0",
      "name": "Hotel",
      "latitude": 41.3825,
      "longitude": 2.1771,
      "cluster_id": 0,
      "cluster_hub_id": "stop_0"
    }
  ],
  "clusters": [
    {
      "cluster_id": 0,
      "member_place_ids": ["stop_0"],
      "hub_place_id": "stop_0",
      "centroid_latitude": 41.3825,
      "centroid_longitude": 2.1771
    }
  ],
  "hub_hub_legs": [
    {
      "from_hub_id": "stop_0",
      "to_hub_id": "stop_1",
      "travel_mode": "BICYCLE",
      "matrix_policy_band": "bicycle",
      "distance_meters": 1200,
      "duration_seconds": 720.0,
      "haversine_meters": 1400.0
    }
  ],
  "edges": [
    {
      "from_place_id": "stop_0",
      "to_place_id": "stop_1",
      "distance_meters": 1200.0,
      "duration_seconds": 720.0,
      "travel_mode_effective": "COMPOSED",
      "quality": "hub_chain",
      "primary_hub_travel_mode": "BICYCLE",
      "legs": []
    }
  ],
  "stats": {
    "place_count": 2,
    "cluster_count": 2,
    "edges_stored": 2
  }
}
```

`edges[].legs` is populated in real output (walk stubs + matrix leg); the snippet stays short for readability.

---

## 5. Behaviour in short

1. **Geocode** each address → pin.  
2. **Walk clusters:** pins closer than the effective walk link (preset + adaptive cap) chain into one pocket; **union–find** merges chains.  
3. **Hub** = medoid per cluster.  
4. **Between hubs:** Route Matrix fills **bicycle / transit / drive** (no walk).  
5. **Stop→stop:** same pocket → walk approximation; different pockets → **hub_chain** using a **primary** mode from Haversine bands on hub separation (`hub_pair_bicycle_max_m`, `hub_pair_transit_max_m` in build config — rarely set by hand; presets tune the walk link and related behaviour).

**Limits:** Cost grows with hub count (full mode matrix per ordered pair). Same-pocket long walks use approximations, not pedestrian routing. Traffic and transit times vary with Google.

---

## 6. Code entry points

- **`execute_routing_check_task`** — run `routing-check` tasks (single OD or place graph).  
- **`build_place_distance_graph`** — graph construction.  
- **`parse_routing_check_task_text`** / **`ROUTING_PLANNER_INSTRUCTION_BLOCK`** in `routing_contracts` — task JSON contract for the planner.

---

## 7. Execution Agent usage: actually querying the graph

Per `docs/workflow.md`, the **Execution Agent** must be able to *use* the Route Check artifact during itinerary composition (not just store it).

### 7.0 Orchestrator / main agent: build once, query with tools

For **tool-calling** orchestrators (separate from the `routing-check` task executor), the split is:

1. **Build connections** — call **`build_place_graph_with_routing_agent`** in `travelplanner.integrations.routing_agent_tools` (wraps **`run_routing_graph_result`** / the LangGraph routing agent: optional LLM cluster preset, then Google graph build). Do this **once** per segment that needs a full graph.
2. **Keep** the returned **`graph`** dict (`place_distance_graph` content) in session / tool context.
3. **Read on demand** — call **`distance_between_places`**, **`closest_places_to_target`**, or `travelplanner.routing_lookup` helpers on that dict. These are **read-only** (no Google calls).

OpenAI-style tool definitions for the three-step loop live in **`ORCHESTRATOR_ROUTING_TOOL_SCHEMAS`**; a wider set (adds single-leg `route_one_leg` and deterministic **`build_distance_graph_from_stops`**) is **`ROUTING_TOOL_SCHEMAS`**.

The classic workflow path remains:

1. **Spawn Route Check** (from a `routing-check` task) → receive a typed artifact: `type: "place_distance_graph"`.
2. **Retain the artifact** in `agent_artifacts` (in-memory is fine; durable persistence optional).
3. **Query it deterministically** for questions like:
   - “distance from **X** to **Y**”
   - “closest of {**A,B,C**} to **Target**”

### 7.1 Deterministic name→id resolution (no LLM)

Use `travelplanner.routing_lookup` helpers. They do not call Google APIs; they only read the precomputed graph.

```python
from travelplanner.routing_lookup import resolve_place_id

graph = artifact.content  # dict (PlaceDistanceGraphModel JSON)
hotel = resolve_place_id(graph, "Hotel")            # fuzzy match
museum = resolve_place_id(graph, "Rijksmuseum")     # exact match
print(hotel.matched.place_id, museum.matched.place_id)
```

If a label is ambiguous (two near-equal matches), `resolve_place_id` raises `PlaceResolutionError` with candidate suggestions so the Execution Agent can ask the user to clarify (or pick a deterministic tie-break policy).

### 7.2 Distance/time between named places

```python
from travelplanner.routing_lookup import distance_between

res = distance_between(graph, "Hotel", "Rijksmuseum", preference="duration")
opt = res.option
print(opt.travel_mode, opt.duration_minutes, opt.distance_km, res.explanation)
```

Notes:
- The returned option is based on the graph’s stored policy (walk within clusters; hub-chain across clusters).
- `res.explanation` reflects whether the edge is `haversine_walk` vs `hub_chain` (and the primary hub mode when composed).

### 7.3 Closest-of queries

```python
from travelplanner.routing_lookup import closest_to

out = closest_to(graph, target_name="Hotel", candidate_names=["A", "B", "C"])
winner = out.winner
print("closest:", winner.candidate.matched.name, winner.option.duration_minutes)
```

### 7.4 If you need IDs only (no fuzzy names)

If you already have `stop_0` / `stop_1` ids, use the lower-level API:
- `load_routing_lookup(graph).get(from_id, to_id)` → all mode options
- `best_mode(graph, from_id, to_id)` → one chosen option
