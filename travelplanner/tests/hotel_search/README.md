# Hotel Search Agent Tests

Two-stage test organization:

- `test_unit/`: unit tests for parsing, filtering, ranking (no API calls)
- `test_integration/`: integration tests with LiteAPI (live API calls)

Run all tests from `travelplanner/`:

```bash
uv run python -m unittest discover -s tests/hotel_search -p "test_*.py" -v
```

## Test files

**Unit tests** (test_unit/):
- `test_artifact_schema.py` — artifact schema validation
- `test_parsing_functions.py` — date/location parsing logic
- `test_amenity_matching.py` — fuzzy amenity matching logic
- `test_filtering_ranking.py` — hotel filtering and ranking

**Integration tests** (test_integration/):
- `test_liteapi_backend.py` — LiteAPI hotel search and details
- `test_graph_execution.py` — full graph execution with real data
- `test_amenity_enrichment.py` — hotel details enrichment workflow
