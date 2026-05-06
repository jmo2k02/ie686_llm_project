"""
Live integration test: cluster + hub matrix + inter-cluster TRANSIT (real Google APIs).

Same layout as ``test_live_search.py``: timestamped JSON under the repo's ``.output/tests/…``.

Usage::

    cd travelplanner
    uv run python test_live_routing.py

Output::

    ../.output/tests/routing-check-results/{timestamp}_{slug}.json

Runs ``input.json`` and every ``examples/routing_check/inputs/*.json`` (each as its own
``place_graph_file`` task). This calls ``execute_routing_check_task`` only — **no LLM**.

Requires ``GOOGLE_MAPS_API_KEY`` and Geocoding + Routes + Route Matrix APIs enabled on the project.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from travelplanner.integrations.routing_execution import execute_routing_check_task
from travelplanner.schema.place_distance_graph import PlaceDistanceGraphBuildConfig
from travelplanner.schema.system_state import TaskModel

_EXAMPLES = Path(__file__).resolve().parent / "examples" / "routing_check"


def _ensure_env() -> None:
    missing = [
        name for name in ("GOOGLE_MAPS_API_KEY",) if not os.getenv(name, "").strip()
    ]
    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}")
        print("Set them in your shell or .env file.")
        sys.exit(1)


def _save_result(payload: dict[str, Any], slug: str) -> Path:
    evidence_dir = (
        Path(__file__).resolve().parent.parent
        / ".output"
        / "tests"
        / "routing-check-results"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = evidence_dir / f"{timestamp}_{slug}.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path


def _discover_input_jobs() -> list[tuple[str, str, str]]:
    """(slug, relative_path_from routing_check/, description)."""
    jobs: list[tuple[str, str, str]] = [
        (
            "cluster-routing-live-barcelona-dense",
            "input.json",
            "Barcelona dense_urban (canonical input.json)",
        ),
    ]
    inputs_dir = _EXAMPLES / "inputs"
    if inputs_dir.is_dir():
        for path in sorted(inputs_dir.glob("*.json")):
            stem = path.stem
            slug = f"cluster-routing-live-{stem}"
            jobs.append((slug, f"inputs/{path.name}", stem.replace("_", " ")))
    return jobs


def _run_one(
    *,
    api_key: str,
    slug: str,
    places_json_path: str,
    graph_overrides: PlaceDistanceGraphBuildConfig,
    input_echo: dict[str, Any],
) -> dict[str, Any]:
    task = TaskModel(
        name=f"live-{slug}",
        type="routing-check",
        text=json.dumps(
            {"kind": "place_graph_file", "places_json_path": places_json_path}
        ),
        is_valid=True,
        validation_comment=None,
    )
    art = execute_routing_check_task(
        task,
        api_key=api_key,
        places_json_base_dir=_EXAMPLES,
        graph_config=graph_overrides,
    )
    n_transit = len((art.content or {}).get("inter_cluster_transit") or [])
    return {
        "slug": slug,
        "places_json_path": places_json_path,
        "input": input_echo,
        "artifact_name": art.name,
        "artifact_type": art.type,
        "description": art.description,
        "inter_cluster_transit_pairs": n_transit,
        "content": art.content,
    }


def main() -> None:
    print("=" * 60)
    print("Live cluster routing (addresses → graph + inter-cluster TRANSIT)")
    print("=" * 60)
    print()

    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=repo_root / ".env", override=False)
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

    _ensure_env()
    api_key = os.environ["GOOGLE_MAPS_API_KEY"].strip()

    graph_overrides = PlaceDistanceGraphBuildConfig(
        cluster_link_m=500.0,
        use_transit_for_hub_pairs=True,
        sleep_between_matrix_requests_s=0.12,
        cluster_link_adaptive=False,
        hub_pair_bicycle_max_m=3500.0,
        departure_time_rfc3339="2026-06-15T10:00:00+02:00",
    )

    jobs = _discover_input_jobs()
    saved: list[tuple[str, Path]] = []
    print(f"Running {len(jobs)} input envelope(s)…")
    for slug, rel_path, label in jobs:
        print()
        print(f"[{slug}] {label} ({rel_path})")
        path_full = _EXAMPLES / rel_path
        try:
            input_echo = json.loads(path_full.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"  SKIP: cannot read {path_full}: {exc}")
            error_blob = {
                "slug": slug,
                "places_json_path": rel_path,
                "input": None,
                "error": f"Could not read input file: {exc}",
            }
            out_path = _save_result(error_blob, slug)
            saved.append((slug, out_path))
            print(f"  Saved (skip): {out_path}")
            continue
        try:
            blob = _run_one(
                api_key=api_key,
                slug=slug,
                places_json_path=rel_path,
                graph_overrides=graph_overrides,
                input_echo=input_echo,
            )
        except Exception as exc:  # noqa: BLE001
            blob = {
                "slug": slug,
                "places_json_path": rel_path,
                "input": input_echo,
                "error": str(exc),
            }
            print(f"  ERROR: {exc}")
        else:
            content = blob.get("content") or {}
            schema_ver = content.get("schema_version")
            if schema_ver is None:
                print("  WARNING: artifact content has no schema_version field")
            elif schema_ver != "1.5":
                print(f"  ERROR: schema_version is '{schema_ver}', expected '1.5'")
                blob["error"] = (
                    f"schema_version mismatch: got '{schema_ver}', expected '1.5'"
                )
            else:
                print(f"  schema_version: {schema_ver} (ok)")

            print(f"  description: {blob.get('description')}")
            print(
                f"  inter_cluster_transit_pairs: {blob.get('inter_cluster_transit_pairs')}"
            )

        out_path = _save_result(blob, slug)
        saved.append((slug, out_path))
        print(f"  Saved: {out_path}")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for slug, path in saved:
        err = json.loads(path.read_text(encoding="utf-8")).get("error")
        status = "error" if err else "ok"
        print(f"  [{status}] {slug} -> {path}")

    evidence_dir = (
        Path(__file__).resolve().parent.parent
        / ".output"
        / "tests"
        / "routing-check-results"
    )
    print()
    print(f"Output directory: {evidence_dir}")


if __name__ == "__main__":
    main()
