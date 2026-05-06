"""Hub matrix policy: skip redundant tiles and unused travel modes."""

from __future__ import annotations

from unittest.mock import patch

from travelplanner.integrations.google_route_matrix import (
    compute_mode_bucket_matrices,
    hub_pair_mode_matrix,
)


def test_hub_pair_mode_matrix_codes() -> None:
    hubs = [(0.0, 0.0), (0.0, 0.002), (1.0, 1.0)]  # ~200m, ~150km
    em = hub_pair_mode_matrix(
        hubs,
        bicycle_max_m=500.0,
        transit_max_m=200_000.0,
        use_transit=True,
    )
    assert int(em[0, 1]) == 1  # bicycle
    assert int(em[0, 2]) in (2, 3)  # transit or drive


def test_compute_mode_bucket_matrices_skips_when_no_transit_needed() -> None:
    """Far-apart hubs → only DRIVE; TRANSIT and BICYCLE tiles should not HTTP."""
    # ~1 degree apart (~111 km) — all DRIVE
    hubs = [(41.0, 2.0), (42.0, 2.0), (41.5, 3.0)]
    calls: list[str] = []

    def _track(mode: str, **_k: object) -> object:
        calls.append(mode)
        return []  # type: ignore[return-value]

    with patch(
        "travelplanner.integrations.google_route_matrix.compute_route_matrix",
        side_effect=lambda *, travel_mode, **kw: _track(travel_mode),
    ):
        _, _, _, skipped = compute_mode_bucket_matrices(
            hubs=hubs,
            api_key="k",
            bicycle_max_m=2500.0,
            transit_max_m=18_000.0,
            use_transit=True,
            departure_time_rfc3339=None,
            sleep_between_requests_s=0.0,
        )
    assert "TRANSIT" not in calls
    assert "BICYCLE" not in calls
    assert "DRIVE" in calls
    assert skipped >= 0
