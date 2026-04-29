"""Hotel Search Agent test runner.

Run from travelplanner/ directory:
    uv run python test_hotel_search.py
"""
from __future__ import annotations

import subprocess


def run_check(command: list[str]) -> None:
    """Run command and exit on failure."""
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    """Run all hotel search agent tests."""
    print("==> compile-check")
    run_check(["uv", "run", "python", "-m", "compileall", "src"])

    print("==> hotel search unit tests")
    run_check(
        [
            "uv",
            "run",
            "python",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests/hotel_search/test_unit",
            "-p",
            "test_*.py",
            "-v",
        ]
    )

    print("==> hotel search integration tests (requires LITEAPI_API_KEY)")
    run_check(
        [
            "uv",
            "run",
            "python",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests/hotel_search/test_integration",
            "-p",
            "test_*.py",
            "-v",
        ]
    )

    print("==> done")


if __name__ == "__main__":
    main()
