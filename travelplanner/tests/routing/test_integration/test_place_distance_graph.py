"""Unit tests for many-place distance graph (cluster + hub matrix + composed edges)."""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest

from travelplanner.integrations.google_route_matrix import RouteMatrixElement
from travelplanner.integrations.google_routes import TravelMode
from travelplanner.integrations.place_distance_graph import build_place_distance_graph
from travelplanner.schema.place_distance_graph import (
    MapPlaceInputModel,
    PlaceDistanceGraphBuildConfig,
)


def _fake_all_modes_matrix() -> dict[tuple[int, int, TravelMode], RouteMatrixElement]:
    """Two hubs, no TRANSIT in config → BICYCLE + DRIVE for each ordered pair."""
    out: dict[tuple[int, int, TravelMode], RouteMatrixElement] = {}
    for mode in ("BICYCLE", "DRIVE"):
        m = cast(TravelMode, mode)
        raw01 = {
            "originIndex": 0,
            "destinationIndex": 1,
            "distanceMeters": 5000,
            "duration": "900s",
        }
        out[(0, 1, m)] = RouteMatrixElement(
            origin_index=0,
            destination_index=1,
            distance_meters=5000,
            duration_seconds=900.0,
            condition="ROUTE_EXISTS",
            raw=raw01,
        )
        raw10 = {
            "originIndex": 1,
            "destinationIndex": 0,
            "distanceMeters": 5100,
            "duration": "910s",
        }
        out[(1, 0, m)] = RouteMatrixElement(
            origin_index=1,
            destination_index=0,
            distance_meters=5100,
            duration_seconds=910.0,
            condition="ROUTE_EXISTS",
            raw=raw10,
        )
    return out


def test_map_place_accepts_lat_lng_aliases() -> None:
    m = MapPlaceInputModel.model_validate(
        {"id": "a", "lat": 41.4, "lng": 2.19, "name": "X", "category": "hotel"},
    )
    assert m.latitude == 41.4
    assert m.longitude == 2.19


def test_map_place_accepts_address_only() -> None:
    m = MapPlaceInputModel.model_validate(
        {
            "id": "p1",
            "name": "Cafè",
            "category": "restaurant",
            "address": "Carrer de la Princesa, 15, Barcelona",
        },
    )
    assert m.latitude is None
    assert m.longitude is None
    assert "Barcelona" in (m.address or "")


def test_map_place_rejects_neither_coords_nor_address() -> None:
    with pytest.raises(ValueError, match="latitude and longitude|address"):
        MapPlaceInputModel.model_validate({"id": "x", "name": "X"})


def test_build_graph_address_only_mock_geocode() -> None:
    places = [
        MapPlaceInputModel(
            id="a1",
            name="A1",
            category="hotel",
            address="Somewhere A, Barcelona",
        ),
        MapPlaceInputModel(
            id="a2",
            name="A2",
            category="hotel",
            address="Somewhere A2, Barcelona",
        ),
        MapPlaceInputModel(
            id="b1",
            name="B1",
            category="restaurant",
            address="Somewhere B, Barcelona",
        ),
        MapPlaceInputModel(
            id="b2",
            name="B2",
            category="restaurant",
            address="Somewhere B2, Barcelona",
        ),
    ]

    def _fake_geocode(addr: str, *, api_key: str) -> tuple[float, float]:
        table = {
            "Somewhere A, Barcelona": (41.387, 2.168),
            "Somewhere A2, Barcelona": (41.388, 2.169),
            "Somewhere B, Barcelona": (41.410, 2.190),
            "Somewhere B2, Barcelona": (41.411, 2.191),
        }
        return table[addr.strip()]

    cfg = PlaceDistanceGraphBuildConfig(
        cluster_link_m=500.0,
        use_transit_for_hub_pairs=False,
        sleep_between_matrix_requests_s=0.0,
    )
    fake = _fake_all_modes_matrix()
    with (
        patch(
            "travelplanner.integrations.place_distance_graph.geocode_address_to_lat_lng",
            side_effect=_fake_geocode,
        ),
        patch(
            "travelplanner.integrations.place_distance_graph.compute_all_travel_modes_hub_matrices",
            return_value=(fake, 2, 16),
        ),
    ):
        graph = build_place_distance_graph(places, api_key="dummy", config=cfg)

    assert graph.stats is not None
    assert graph.stats.cluster_count == 2


def test_build_graph_two_clusters_mock_matrix() -> None:
    """Four corners: two tight pairs → two clusters; hub matrix mocked."""
    places = [
        MapPlaceInputModel(
            id="a1", name="A1", category="hotel", latitude=41.387, longitude=2.168
        ),
        MapPlaceInputModel(
            id="a2", name="A2", category="hotel", latitude=41.388, longitude=2.169
        ),
        MapPlaceInputModel(
            id="b1", name="B1", category="restaurant", latitude=41.410, longitude=2.190
        ),
        MapPlaceInputModel(
            id="b2", name="B2", category="restaurant", latitude=41.411, longitude=2.191
        ),
    ]
    cfg = PlaceDistanceGraphBuildConfig(
        cluster_link_m=500.0,
        use_transit_for_hub_pairs=False,
        sleep_between_matrix_requests_s=0.0,
    )
    fake = _fake_all_modes_matrix()
    with patch(
        "travelplanner.integrations.place_distance_graph.compute_all_travel_modes_hub_matrices",
        return_value=(fake, 2, 16),
    ):
        graph = build_place_distance_graph(places, api_key="dummy", config=cfg)

    assert graph.stats is not None
    assert graph.stats.cluster_count == 2
    assert graph.stats.matrix_http_requests == 2
    assert graph.geojson is not None
    assert graph.geojson["type"] == "FeatureCollection"
    assert len(graph.hub_hub_legs) == 4
    cross = [e for e in graph.edges if e.quality == "hub_chain"]
    assert (
        len(cross) == 8
    )  # 4*2 directed cross-cluster pairs (each direction between clusters)
    assert all(e.travel_mode_effective == "COMPOSED" for e in cross)
    assert all(
        e.primary_hub_travel_mode in ("BICYCLE", "TRANSIT", "DRIVE") for e in cross
    )
    clusters_sorted = sorted(graph.clusters, key=lambda c: c.cluster_id)
    h0, h1 = clusters_sorted[0].hub_place_id, clusters_sorted[1].hub_place_id
    leg = graph.hub_hub_leg(h0, h1, "BICYCLE")
    assert leg is not None
    assert leg.duration_seconds is not None
    assert abs(leg.duration_seconds / 60.0 - 15.0) < 0.01


def test_rejects_duplicate_ids() -> None:
    places = [
        MapPlaceInputModel(id="x", latitude=41.0, longitude=2.0),
        MapPlaceInputModel(id="x", latitude=41.1, longitude=2.1),
    ]
    with pytest.raises(ValueError, match="unique"):
        build_place_distance_graph(places, api_key="k")
