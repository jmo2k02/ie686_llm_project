#!/usr/bin/env python3
"""
Constraint fail-frequency analysis across baseline and travel_agent evaluation results.

Usage:
    python scripts/evaluate/error_analysis.py
    python scripts/evaluate/error_analysis.py --data-dir data/evaluation
"""
import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

SOURCES = ["baseline", "travel_agent"]
VERDICT_FAIL = {"FAIL", "MISSING_INFO"}


def load_constraints(data_dir: Path) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {s: [] for s in SOURCES}
    for source in SOURCES:
        source_dir = data_dir / source
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.glob("*.json")):
            data = json.loads(path.read_text())
            if data.get("status") != "ok":
                continue
            for c in data.get("scorecard", {}).get("aggregated_constraints", []):
                records[source].append(c)
    return records


def build_rows(records_by_source: dict[str, list[dict]]) -> list[dict]:
    stats: dict[str, dict] = {}
    for source, constraints in records_by_source.items():
        for c in constraints:
            cid = c["id"]
            if cid not in stats:
                stats[cid] = {"text": c["constraint_text"], "type": c["constraint_type"]}
            entry = stats[cid].setdefault(source, {"fails": 0, "applicable": 0})
            if c["final_verdict"] != "NA":
                entry["applicable"] += 1
                if c["final_verdict"] in VERDICT_FAIL:
                    entry["fails"] += 1
    rows = []
    for cid, info in stats.items():
        total_fails = sum(info.get(s, {}).get("fails", 0) for s in SOURCES)
        rows.append({"id": cid, "total_fails": total_fails, **info})
    return rows


def _display_text(row: dict) -> str:
    if row["type"] == "hard":
        return row["text"].split(":")[0].strip()
    return row["text"]


def _cell(row: dict, source: str) -> str:
    e = row.get(source, {})
    return f"{e.get('fails', 0)} / {e.get('applicable', 0)}"


def print_analysis(rows: list[dict], console: Console) -> None:
    for ctype, label in [("hard", "Hard Constraints (HC)"),
                         ("commonsense", "Commonsense Constraints (CC)")]:
        section = sorted(
            [r for r in rows if r["type"] == ctype],
            key=lambda r: (-r["total_fails"], r["id"]),
        )
        t = Table(title=label, show_lines=False, expand=False)
        t.add_column("ID", style="bold cyan", no_wrap=True)
        t.add_column("Constraint", max_width=60)
        t.add_column("Baseline\nfails/app", justify="center")
        t.add_column("Travel Agent\nfails/app", justify="center")
        t.add_column("Total\nfails", justify="center", style="bold red")
        for r in section:
            t.add_row(
                r["id"],
                _display_text(r),
                _cell(r, "baseline"),
                _cell(r, "travel_agent"),
                str(r["total_fails"]),
            )
        console.print(t)
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show which constraints fail most often across baseline and travel_agent."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/evaluation"))
    args = parser.parse_args()
    console = Console()
    print_analysis(build_rows(load_constraints(args.data_dir)), console)


if __name__ == "__main__":
    main()
