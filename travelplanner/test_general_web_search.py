from __future__ import annotations

import subprocess


def run_check(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    print("==> compile-check")
    run_check(["uv", "run", "python", "-m", "compileall", "src"])
    print("==> general web search tests")
    run_check(
        [
            "uv",
            "run",
            "python",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-v",
        ]
    )
    print("==> done")


if __name__ == "__main__":
    main()
