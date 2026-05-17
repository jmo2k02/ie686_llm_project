# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-23
**Commit:** d20c2ac
**Branch:** main

---

## OVERVIEW

Automated Travel and Event Planner using LLM agents (LangGraph). Multi-package Python repo with `travelplanner` (core package) and `scripts/` (evaluation tools). Research project for benchmarking LLM agents against the TravelPlanner dataset.

---

## STRUCTURE

```
ie686_llm_project/
├── travelplanner/           # Core package (src layout)
│   └── src/travelplanner/
│       ├── agents/          # Empty submodule stub
│       ├── pipeline/        # Empty submodule stub
│       ├── schema/          # Empty submodule stub
│       ├── __init__.py      # Package init with stub main()
│       └── main.py          # Intended entry point (stub docstring only)
├── scripts/                 # Evaluation/utility tools
│   ├── evaluate/            # Benchmark evaluation pipeline
│   ├── example_agents/      # Example LangGraph agent implementations
│   ├── hello_world.py       # Test scaffold
│   └── pyproject.toml       # References travelplanner from git URL
├── data/                    # Training data (tp_train/, samples)
├── docs/                    # Architecture docs + course materials
│   ├── workflow.md          # 616-line system design spec
│   └── lecture/             # University course (IE685/IE686)
└── sandbox/                 # Experimental code (non-standard)
```

---

## AGENT INSTRUCTIONS

### How to Navigate This Project

| If you need... | Go to... |
|----------------|----------|
| Project overview & structure | `./AGENTS.md` (this file) |
| Lecture materials & course content | `docs/lecture/AGENTS.md` |
| Evaluation pipeline details | `scripts/evaluate/AGENTS.md` |
| Architecture & agent design rules | `docs/workflow.md` |
| DeepAgent lab agent skills | `docs/lecture/deep_agent/deepagents_lab/skills/*/SKILL.md` |

### Agent Skill Definitions (DeepAgent Lab)

When working on student proposals, read the skill files:
- **Research**: `docs/lecture/deep_agent/deepagents_lab/skills/research/benchmark-research/SKILL.md`
- **Writing**: `docs/lecture/deep_agent/deepagents_lab/skills/main/proposal-writer/SKILL.md`
- **Review**: `docs/lecture/deep_agent/deepagents_lab/skills/review/proposal-critic/SKILL.md`

### Script Execution

**ALWAYS** run scripts from the `scripts/` directory:

```bash
cd scripts/
uv run python <script> [OPTIONS]
```

---

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Run LangGraph agent | `scripts/example_agents/langgraph_minimal.py` | Full CLI with dataset loading |
| Evaluate agents | `scripts/evaluate/pipeline.py` | Benchmark runner with --mode flags |
| Architecture docs | `docs/workflow.md` | 616-line system design spec |
| Course materials | `docs/lecture/` | University lecture notes (IE685/IE686) |
| DeepAgent lab | `docs/lecture/deep_agent/deepagents_lab/` | Student project workspace |

---

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `make_graph()` | Function | `scripts/example_agents/langgraph_minimal.py:80` | Builds LangGraph StateGraph |
| `generation_loop()` | Function | `scripts/evaluate/generate.py:159` | Benchmark evaluation loop |
| `instantiate_graph()` | Function | `scripts/evaluate/generate.py:105` | Dynamic graph loading via import string |
| `load_split()` | Function | `scripts/evaluate/generate.py:113` | Loads HuggingFace TravelPlanner dataset |
| `TravelPlanner` | Dataset | `osunlp/TravelPlanner` | External benchmark dataset |

---

## CONVENTIONS

- **Python version**: 3.12 (travelplanner), 3.9+ (scripts)
- **Package manager**: `uv` (Astral) — NOT pip/poetry
- **Multi-package repo**: `travelplanner/` and `scripts/` are separate packages
- **External dep**: `scripts/pyproject.toml` references travelplanner via git URL, not local path

---

## ANTI-PATTERNS

**From docs/workflow.md — Design Constraints:**

| Forbidden | Location | Rule |
|-----------|----------|------|
| Anonymous output blobs | workflow.md:266 | Agents MUST return typed JSON artifacts |
| Fixed static agent pools | workflow.md:270 | Spawn agents on-demand per task |
| Plain text/chat returns | workflow.md:279 | Return artifact documents, not logs |
| Planner before constraints | workflow.md:571 | Constraint gathering is mandatory first |
| >3 retries | workflow.md:560 | Max 3 retries to prevent infinite loops |
| Single-option without alternatives | workflow.md:577 | Always provide backup options |

**Critical**: Search agents must emit typed artifacts with schema+id. Treat output as anonymous blob = forbidden.

---

## UNIQUE STYLES

- **Benchmark-driven**: No pytest/unit tests — uses train/validation/test data splits
- **Dynamic graph loading**: `instantiate_graph("/path/to/module.py:function_name")` pattern
- **No CI/CD**: Manual execution only, no GitHub Actions/Makefile/Docker
- **Sparse codebase**: Empty submodule stubs in `travelplanner/src/travelplanner/`

---

## COMMANDS

```bash
# Run LangGraph agent
cd scripts/
uv run python example_agents/langgraph_minimal.py --model_name gpt-4o-mini --set_type validation

# Evaluate agents
uv run python evaluate/pipeline.py --model_name gpt-4o-mini --set_type train --output_dir /tmp/travelplanner --mode sole-planning --strategy direct

# Install travelplanner package
cd travelplanner/
uv pip install -e .
```

---

## GOTCHAS

- `travelplanner/src/travelplanner/main.py` is a STUB (3 lines, docstring only)
- `__init__.py:1` contains stub `main()` printing "Hello from travelplanner!" (not a real entry point)
- `scripts/evaluate/evaluate.py` is empty (0 lines) — incomplete
- `scripts/pyproject.toml` has git URL dependency on travelplanner package

---

## MAINTENANCE

**AGENTS.md files must be kept in sync with code changes.** With each PR:
1. Review all AGENTS.md files affected by your changes
2. Update any stale facts, line numbers, or descriptions
3. Add new conventions/anti-patterns discovered during implementation
4. Remove references to code that no longer exists

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **ie686_llm_project** (7399 symbols, 11738 relationships, 281 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/ie686_llm_project/context` | Codebase overview, check index freshness |
| `gitnexus://repo/ie686_llm_project/clusters` | All functional areas |
| `gitnexus://repo/ie686_llm_project/processes` | All execution flows |
| `gitnexus://repo/ie686_llm_project/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
