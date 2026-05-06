"""Routing API disk cache (versioned, success-only)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from travelplanner.integrations.google_route_matrix import compute_route_matrix
from travelplanner.integrations.routing_api_cache import (
    read_cached_body,
    reset_routing_cache_metrics,
    routing_cache_metrics,
    write_cached_body,
)


def test_cache_key_changes_with_integration_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRAVELPLANNER_ROUTING_CACHE_DIR", str(tmp_path))
    from travelplanner.integrations import routing_api_cache as rac

    monkeypatch.setattr(rac, "ROUTING_CACHE_INTEGRATION_VERSION", "test_iv_a")
    d1 = rac.routing_cache_version_dir()
    monkeypatch.setattr(rac, "ROUTING_CACHE_INTEGRATION_VERSION", "test_iv_b")
    d2 = rac.routing_cache_version_dir()
    assert d1 != d2
    assert "test_iv_a" in str(d1)
    assert "test_iv_b" in str(d2)


def test_compute_route_matrix_uses_cache_on_second_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TRAVELPLANNER_ROUTING_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TRAVELPLANNER_ROUTING_CACHE_DISABLE", raising=False)
    reset_routing_cache_metrics()

    sample = json.dumps(
        [
            {
                "originIndex": 0,
                "destinationIndex": 0,
                "distanceMeters": 100,
                "duration": "60s",
                "condition": "ROUTE_EXISTS",
            }
        ]
    )

    mock_resp = MagicMock()
    mock_resp.read.return_value = sample.encode("utf-8")
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_resp
    mock_cm.__exit__.return_value = None

    with patch("travelplanner.integrations.google_route_matrix.urllib.request.urlopen", return_value=mock_cm):
        a = compute_route_matrix(
            origins=[(41.0, 2.0)],
            destinations=[(41.001, 2.001)],
            api_key="test-key-abc",
            travel_mode="WALK",
        )
        b = compute_route_matrix(
            origins=[(41.0, 2.0)],
            destinations=[(41.001, 2.001)],
            api_key="test-key-abc",
            travel_mode="WALK",
        )

    assert len(a) == 1
    assert len(b) == 1
    hits, misses = routing_cache_metrics()
    assert hits >= 1
    assert misses >= 1
    assert mock_cm.__enter__.call_count == 1


def test_write_then_read_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRAVELPLANNER_ROUTING_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TRAVELPLANNER_ROUTING_CACHE_DISABLE", raising=False)
    key = "a" * 64
    write_cached_body(key_hex=key, body_text='[{"x":1}]')
    assert read_cached_body(key_hex=key) == '[{"x":1}]'
