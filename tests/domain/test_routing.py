import pytest

from rag_eval_harness.domain.errors import AppError
from rag_eval_harness.domain.routing import parse_routing_set


def _case(**overrides):
    case = {"id": "r-001", "question": "Xin chào", "expected_route": "direct_answer", "reviewed": True}
    case.update(overrides)
    return case


def test_load_valid():
    cases = parse_routing_set([_case(), _case(id="r-002", expected_route="rag_retrieval")])

    assert [c.expected_route for c in cases] == ["direct_answer", "rag_retrieval"]


def test_unknown_route_invalid():
    with pytest.raises(AppError) as exc:
        parse_routing_set([_case(expected_route="web_search")])

    assert exc.value.code == "DATASET_INVALID"
    assert {"item": "r-001", "field": "expected_route", "rule": "enum"} in exc.value.details


def test_duplicate_id():
    with pytest.raises(AppError) as exc:
        parse_routing_set([_case(), _case()])

    assert exc.value.code == "DATASET_DUPLICATE_ID"
