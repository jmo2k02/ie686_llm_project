"""Pytest defaults: isolate routing HTTP cache per session (no ~/.cache pollution)."""

from __future__ import annotations

import os

import pytest

# Mocked HTTP tests reuse the same origin/destination strings; disable cache by default so they
# never read a real response left from another test. Integration tests for the cache unset this.
os.environ.setdefault("TRAVELPLANNER_ROUTING_CACHE_DISABLE", "1")


@pytest.fixture(scope="session", autouse=True)
def _isolate_travelplanner_routing_cache(tmp_path_factory: pytest.TempPathFactory) -> None:
    root = tmp_path_factory.mktemp("tp_routing_cache")
    previous = os.environ.get("TRAVELPLANNER_ROUTING_CACHE_DIR")
    os.environ["TRAVELPLANNER_ROUTING_CACHE_DIR"] = str(root)
    yield
    if previous is None:
        os.environ.pop("TRAVELPLANNER_ROUTING_CACHE_DIR", None)
    else:
        os.environ["TRAVELPLANNER_ROUTING_CACHE_DIR"] = previous
