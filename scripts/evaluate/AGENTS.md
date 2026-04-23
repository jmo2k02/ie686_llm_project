# scripts/evaluate/

**Parent:** `../../AGENTS.md`

---

## OVERVIEW

Benchmark evaluation pipeline for LangGraph-based TravelPlanner agents. Loads dataset from HuggingFace (`osunlp/TravelPlanner`), runs agents against it, persists results as JSON.

---

## STRUCTURE

```
evaluate/
├── __init__.py
├── evaluate.py       # EMPTY (0 lines) — do NOT use as reference
├── generate.py       # Core: generation_loop(), instantiate_graph(), load_split()
├── pipeline.py       # CLI entry: build_arg_parser(), main()
├── process.py        # Post-processing (not examined)
└── schema.py         # Empty (3 lines, only imports BaseModel)
```

---

## AGENT INSTRUCTIONS

### How to Run Evaluations

**ALWAYS** run from the `scripts/` directory:

```bash
cd scripts/
uv run python evaluate/pipeline.py [args]
```

### Required Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `--model_name` | OpenAI model ID | Model to evaluate (required) |
| `--set_type` | `train`, `validation`, `test` | Dataset split |
| `--output_dir` | path | Directory for JSON results |
| `--mode` | `two-stage`, `sole-planning` | Evaluation mode |
| `--strategy` | string | Strategy tag (default: `direct`) |

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `instantiate_graph()` | `generate.py:105` | Loads graph from import string |
| `load_split()` | `generate.py:113` | Loads HuggingFace dataset |
| `generation_loop()` | `generate.py:159` | Runs benchmark loop |

---

## OUTPUT FORMAT

Results saved as `generated_plan_*.json` files with key structure:
```
{model_name}{suffix}_{mode}_results
```

Example: `gpt-4o-mini_direct_sole-planning_results`

---

## ANTI-PATTERNS

- **`evaluate.py` is EMPTY** — do not use as reference or example
- `generation_loop()` overwrites only the **last entry** if file exists (line 152)

---

## EXAMPLE COMMANDS

```bash
# Standard evaluation
uv run python evaluate/pipeline.py --model_name gpt-4o-mini --set_type validation --output_dir ./outputs

# With example cap (for testing)
uv run python evaluate/pipeline.py --model_name gpt-4o-mini --max_examples 10

# Custom graph specification
uv run python evaluate/pipeline.py --graph_spec /path/to/module.py:make_graph
```

---

## GOTCHAS

- `evaluate.py` is 0 lines — incomplete, do not reference
- `process.py` and `schema.py` not examined — may contain relevant code