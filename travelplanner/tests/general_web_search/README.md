# General Web Search Agent Tests

Two-stage test organization:

- `test_unit/`: config, schema, and unit tests (no graph execution)
- `test_integration/`: graph-based tests (API backends, workflow execution)

Run all tests from `travelplanner/`:

```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

## Test files

**Unit tests** (test_unit/):
- `test_artifact_schema.py` — artifact schema validation
- `test_config.py` — configuration defaults and env overrides
- `test_search_agent_registry.py` — agent registration
- `test_settings_loader.py` — settings loading and merging
- `test_task_filtering.py` — task selection behavior

**Integration tests** (test_integration/):
- `test_tavily_backend.py` — Tavily API backend
- `test_retry_and_status.py` — retry logic and status handling
- `test_iterative_search_quality_gating.py` — iterative search with quality gating
- `test_graph_openrouter_summary.py` — OpenRouter summary in graph
- `test_workflow_execution_metadata.py` — workflow metadata recording