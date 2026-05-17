"""Layer 1 process-error analysis from local LangSmith trace files.

Reads exported JSON traces from::

    data/travelplans/LangSmithTraces/baseline/
    data/travelplans/LangSmithTraces/travel_agent/

No LangSmith API key required.

Usage::

    python -m travelplanner.evaluation.error_analysis.error_analysis_llm
    python -m travelplanner.evaluation.error_analysis.error_analysis_llm \\
        --traces-dir data/travelplans/LangSmithTraces \\
        --output data/evaluation/error_report.json

Or from Python::

    from travelplanner.evaluation.error_analysis.error_analysis_llm import run_analysis
    report = run_analysis("data/travelplans/LangSmithTraces")
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from travelplanner.evaluation.error_analysis.error_classes import (
    AnalysisConfig,
    ErrorClass,
    ErrorReport,
    RunClassification,
    ToolCallRecord,
)

# ── Optional: sentence-transformers for REPEATED_QUERY ────────────────────

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_MODEL: Any = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    _ST_MODEL = None

# ── Constants ──────────────────────────────────────────────────────────────

SOURCES = ("baseline", "travel_agent")

_LOOP_KW = frozenset({
    "recursion limit", "max iterations",
    "graphrecursionerror", "recursionerror",
})


# ── Trace loading ──────────────────────────────────────────────────────────

def _load_trace(path: Path) -> dict | None:
    """Return parsed trace dict, or None for stub/unreadable files."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("outputs"):
        return None
    return data


# ── Tool call extraction ───────────────────────────────────────────────────

def _calls_from_messages(
    messages: list[dict],
    agent_name: str | None,
    start_pos: int,
) -> list[ToolCallRecord]:
    records: list[ToolCallRecord] = []
    pos = start_pos
    for msg in messages:
        if msg.get("type") != "ai":
            continue
        for tc in msg.get("tool_calls", []):
            records.append(ToolCallRecord(
                tool_name=tc.get("name", ""),
                inputs_json=json.dumps(tc.get("args", {}), sort_keys=True),
                position=pos,
                error=None,
                agent_name=agent_name,
            ))
            pos += 1
        for tc in msg.get("invalid_tool_calls", []):
            records.append(ToolCallRecord(
                tool_name=tc.get("name", "unknown"),
                inputs_json=json.dumps(tc.get("args", {}), sort_keys=True),
                position=pos,
                error="invalid_tool_call",
                agent_name=agent_name,
            ))
            pos += 1
    return records


def extract_tool_calls(trace: dict, source: str) -> list[ToolCallRecord]:
    """Return all tool calls in execution order, tagged with agent_name."""
    outputs = trace.get("outputs") or {}
    records: list[ToolCallRecord] = []

    if source == "baseline":
        records = _calls_from_messages(outputs.get("messages", []), None, 0)
    else:
        pos = 0
        for agent_name, history in outputs.get("message_histories", {}).items():
            new = _calls_from_messages(history.get("messages", []), agent_name, pos)
            records.extend(new)
            pos += len(new)

    return records


# ── Similarity ─────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def _similarity(a: str, b: str) -> float:
    if _ST_MODEL is not None:
        vecs = _ST_MODEL.encode([a, b], normalize_embeddings=True)
        return float(np.dot(vecs[0], vecs[1]))
    return _jaccard(a, b)


# ── Detectors ──────────────────────────────────────────────────────────────

def detect_repeated_tool_call(tool_calls: list[ToolCallRecord]) -> tuple[bool, dict]:
    """Consecutive identical (tool_name, args) pair — exact string match."""
    valid = [tc for tc in tool_calls if tc.error is None]
    for i in range(len(valid) - 1):
        a, b = valid[i], valid[i + 1]
        if a.tool_name == b.tool_name and a.inputs_json == b.inputs_json:
            return True, {"position": a.position, "tool_name": a.tool_name}
    return False, {}


def detect_repeated_query(
    tool_calls: list[ToolCallRecord],
    threshold: float = 0.85,
) -> tuple[bool, dict]:
    """Semantically similar (but not identical) calls to the same tool."""
    valid = [tc for tc in tool_calls if tc.error is None]
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a, b = valid[i], valid[j]
            if a.tool_name != b.tool_name or a.inputs_json == b.inputs_json:
                continue
            sim = _similarity(a.inputs_json, b.inputs_json)
            if sim >= threshold:
                return True, {
                    "positions": [a.position, b.position],
                    "similarity": round(sim, 3),
                    "tool_name": a.tool_name,
                    "method": "sentence_transformers" if _ST_MODEL else "jaccard",
                }
    return False, {}


def detect_dead_loop(
    trace: dict,
    tool_calls: list[ToolCallRecord],
    source: str,
    min_repetitions: int = 3,
) -> tuple[bool, dict]:
    """Top-level error keyword, excessive validation retries, or repeating tool sequence."""
    error = (trace.get("error") or "").lower()
    if any(kw in error for kw in _LOOP_KW):
        return True, {"trigger": "error_field", "error": error[:200]}

    outputs = trace.get("outputs") or {}
    if source == "travel_agent":
        attempts = outputs.get("validation_attempts", 1)
        if attempts >= min_repetitions:
            return True, {"trigger": "validation_loop", "attempts": attempts}

    seq = [(tc.tool_name, tc.inputs_json) for tc in tool_calls if tc.error is None]
    n = len(seq)
    for window in range(2, n // 2 + 1):
        for start in range(n - window * min_repetitions + 1):
            pattern = seq[start: start + window]
            reps, pos = 1, start + window
            while pos + window <= n and seq[pos: pos + window] == pattern:
                reps += 1
                pos += window
            if reps >= min_repetitions:
                return True, {"trigger": "tool_loop", "pattern": [p[0] for p in pattern], "repetitions": reps}
    return False, {}


def detect_cascading_error(
    trace: dict,
    tool_calls: list[ToolCallRecord],
) -> tuple[bool, dict]:
    """Invalid tool calls, multiple failed tool messages, or top-level run error."""
    invalid = sum(1 for tc in tool_calls if tc.error == "invalid_tool_call")
    if invalid > 0:
        return True, {"invalid_tool_calls": invalid}

    outputs = trace.get("outputs") or {}
    failed: list[str] = []
    for msg in outputs.get("messages", []):
        if msg.get("type") == "tool" and msg.get("status", "success") != "success":
            failed.append(msg.get("name", "unknown"))
    for hist in outputs.get("message_histories", {}).values():
        for msg in hist.get("messages", []):
            if msg.get("type") == "tool" and msg.get("status", "success") != "success":
                failed.append(msg.get("name", "unknown"))
    if len(failed) >= 2:
        return True, {"failed_tool_count": len(failed), "tools": failed}

    if trace.get("error"):
        return True, {"error": str(trace["error"])[:200]}

    return False, {}


def detect_handoff_failure(
    trace: dict,
    source: str,
    expected_agents: frozenset[str] | None = None,
) -> tuple[bool, dict]:
    """Missing expected sub-agent or absent travel plan (travel_agent only)."""
    if source == "baseline":
        return False, {}

    outputs = trace.get("outputs") or {}
    present = set(outputs.get("message_histories", {}).keys())

    if expected_agents:
        missing = expected_agents - present
        if missing:
            return True, {"missing_agents": sorted(missing)}

    if not outputs.get("travelplan"):
        return True, {"trigger": "no_travelplan"}

    return False, {}


# ── Classification ─────────────────────────────────────────────────────────

def classify_trace(
    run_id: str,
    source: str,
    trace: dict,
    tool_calls: list[ToolCallRecord],
    config: AnalysisConfig,
) -> RunClassification:
    error_classes: list[ErrorClass] = []
    flags: dict[str, Any] = {}

    def _check(ec: ErrorClass, hit: bool, meta: dict) -> None:
        if hit:
            error_classes.append(ec)
            flags[ec.value] = meta

    _check(ErrorClass.REPEATED_TOOL_CALL, *detect_repeated_tool_call(tool_calls))
    _check(ErrorClass.REPEATED_QUERY, *detect_repeated_query(tool_calls, config.repeated_query_threshold))
    _check(ErrorClass.DEAD_LOOP, *detect_dead_loop(trace, tool_calls, source, config.dead_loop_min_repetitions))
    _check(ErrorClass.CASCADING_ERROR, *detect_cascading_error(trace, tool_calls))
    _check(ErrorClass.HANDOFF_FAILURE, *detect_handoff_failure(trace, source, config.expected_travel_agents))

    return RunClassification(
        run_id=run_id,
        source=source,
        error_classes=error_classes,
        flags=flags,
    )


# ── Aggregation ────────────────────────────────────────────────────────────

def aggregate(classifications: list[RunClassification]) -> ErrorReport:
    per_class: dict[str, int] = {ec.value: 0 for ec in ErrorClass}
    for rc in classifications:
        for ec in rc.error_classes:
            per_class[ec.value] += 1

    total = len(classifications)
    per_class_pct = {
        k: round(v / total * 100, 1) if total else 0.0
        for k, v in per_class.items()
    }
    return ErrorReport(
        classifications=classifications,
        per_class=per_class,
        per_class_pct=per_class_pct,
        total_runs=total,
        clean_runs=sum(1 for rc in classifications if rc.is_clean()),
    )


# ── Export ─────────────────────────────────────────────────────────────────

def export_json(report: ErrorReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_runs": report.total_runs,
        "clean_runs": report.clean_runs,
        "per_class": report.per_class,
        "per_class_pct": report.per_class_pct,
        "classifications": [
            {
                "run_id": rc.run_id,
                "source": rc.source,
                "error_classes": [ec.value for ec in rc.error_classes],
                "flags": rc.flags,
            }
            for rc in report.classifications
        ],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# ── Rich table ─────────────────────────────────────────────────────────────

def _print_report(report: ErrorReport, console: Console) -> None:
    by_source: dict[str, list[RunClassification]] = defaultdict(list)
    for rc in report.classifications:
        by_source[rc.source].append(rc)

    def _fmt(n: int, total: int) -> str:
        pct = round(n / total * 100) if total else 0
        return f"{n}  |  {pct}%"

    t = Table(
        title="Process-Level Error Distribution (Layer 1)",
        show_lines=True,
        expand=False,
    )
    t.add_column("Error Class", style="bold cyan", no_wrap=True)
    for src in SOURCES:
        t.add_column(f"{src.replace('_', ' ').title()}\nN  |  %", justify="center")
    t.add_column("Total N", justify="center", style="bold red")

    for ec in ErrorClass:
        cells: list[str] = []
        total_n = 0
        for src in SOURCES:
            runs = by_source.get(src, [])
            n = sum(1 for rc in runs if ec in rc.error_classes)
            cells.append(_fmt(n, len(runs)))
            total_n += n
        t.add_row(ec.value, *cells, str(total_n))

    t.add_section()
    for label, pred, style in [
        ("≥1 error", lambda rc: not rc.is_clean(), "bold"),
        ("Clean",    lambda rc: rc.is_clean(),     "bold green"),
    ]:
        cells = [
            _fmt(
                sum(1 for rc in by_source.get(src, []) if pred(rc)),
                len(by_source.get(src, [])),
            )
            for src in SOURCES
        ]
        total = sum(1 for rc in report.classifications if pred(rc))
        t.add_row(f"[{style}]{label}[/{style}]", *cells, str(total), style=style)

    console.print(t)


# ── Main entry point ───────────────────────────────────────────────────────

def run_analysis(
    traces_dir: str | Path = "data/travelplans/LangSmithTraces",
    output_path: str | Path = "data/evaluation/error_report.json",
    config: AnalysisConfig | None = None,
) -> ErrorReport:
    traces_dir = Path(traces_dir)
    output_path = Path(output_path)
    if config is None:
        config = AnalysisConfig()

    console = Console()
    classifications: list[RunClassification] = []

    for source in SOURCES:
        source_dir = traces_dir / source
        if not source_dir.exists():
            console.print(f"[yellow]warning:[/yellow] {source_dir} not found, skipping")
            continue
        files = sorted(source_dir.glob("*.json"))
        console.print(f"[bold]{source}:[/bold] {len(files)} trace files")
        for path in files:
            trace = _load_trace(path)
            if trace is None:
                continue
            tool_calls = extract_tool_calls(trace, source)
            rc = classify_trace(path.stem, source, trace, tool_calls, config)
            classifications.append(rc)

    report = aggregate(classifications)
    _print_report(report, console)
    export_json(report, output_path)
    console.print(f"\n[dim]Report written to {output_path}[/dim]")
    return report


# ── CLI ────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 1 process-error analysis from local LangSmith trace files."
    )
    parser.add_argument(
        "--traces-dir", type=Path,
        default=Path("data/travelplans/LangSmithTraces"),
        help="Root dir with baseline/ and travel_agent/ subdirs",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/evaluation/error_report.json"),
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    args = parser.parse_args()
    config = AnalysisConfig(repeated_query_threshold=args.similarity_threshold)
    run_analysis(args.traces_dir, args.output, config)


if __name__ == "__main__":
    _main()
