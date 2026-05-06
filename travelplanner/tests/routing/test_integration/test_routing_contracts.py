from __future__ import annotations

import pytest

from travelplanner.integrations.routing_contracts import (
    PlaceGraphFileTaskPayload,
    SingleOdTaskPayload,
    parse_routing_check_task_text,
)


def test_parse_single_od() -> None:
    text = '{"kind":"single_od","origin_address":"A","destination_address":"B","travel_mode":"walk"}'
    p = parse_routing_check_task_text(text)
    assert isinstance(p, SingleOdTaskPayload)
    assert p.origin_address == "A"
    assert p.destination_address == "B"


def test_parse_place_graph_file() -> None:
    text = '{"kind":"place_graph_file","places_json_path":"./places.json"}'
    p = parse_routing_check_task_text(text)
    assert isinstance(p, PlaceGraphFileTaskPayload)
    assert p.cluster_context is None


def test_parse_place_graph_file_with_cluster_context() -> None:
    text = (
        '{"kind":"place_graph_file","places_json_path":"./places.json",'
        '"cluster_context":"dense_urban"}'
    )
    p = parse_routing_check_task_text(text)
    assert isinstance(p, PlaceGraphFileTaskPayload)
    assert p.cluster_context == "dense_urban"


def test_parse_rejects_prose() -> None:
    with pytest.raises(ValueError):
        parse_routing_check_task_text("check distances between hotels")
