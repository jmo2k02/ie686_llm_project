# Routing Example

Simple address-first routing: give a list of locations → get clusters + connection time lookup.

## Input Format

```json
{
  "description": "Optional description",
  "stops": [
    {"address": "Dam 1, Amsterdam", "name": "Start"},
    {"address": "Museumplein 6, Amsterdam", "name": "End"}
  ]
}
```

## Usage

```python
from travelplanner.agents.routing_agent import run_routing_agent

stops = [
    {"address": "Dam 1, 1012 JS Amsterdam", "name": "Dam Square"},
    {"address": "Museumplein 6, 1071 DJ Amsterdam", "name": "Rijksmuseum"},
]
artifact = run_routing_agent(stops, api_key="YOUR_API_KEY")
print(artifact.description)
# Output: "4 places, 2 clusters, 12 directed edges"
```

## Output Structure

The artifact content is a `PlaceDistanceGraphModel` with:
- `places`: List of place nodes with lat/lng and cluster assignment
- `clusters`: List of clusters with hub medoid and member place IDs
- `hub_hub_legs`: Hub-to-hub distance matrix for bike/transit/drive
- `edges`: Directed pairwise edges with duration estimates (A→B lookups)

## Cluster Presets

The LLM automatically picks the best cluster preset:
- `dense_urban`: City center, tight walking clusters
- `sparse`: Road trip, looser clustering
- `mixed`: Default, balanced