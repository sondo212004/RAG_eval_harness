import pytest

from rag_eval_harness.app.promote import promote
from rag_eval_harness.domain.errors import AppError
from tests.fakes import FakeStore


# AC14
def test_promote_valid():
    store = FakeStore({("r1", "eval"): {"incomplete": False}, ("r1", "bench"): {"incomplete": False, "b": 1}})

    promote("r1", store)

    assert store.load("baseline", "eval") == {"incomplete": False}
    assert store.load("baseline", "bench") == {"incomplete": False, "b": 1}


def test_promote_without_bench_removes_stale_baseline_bench():
    store = FakeStore({("r1", "eval"): {"incomplete": False}, ("baseline", "bench"): {"old": True}})

    promote("r1", store)

    assert not store.exists("baseline", "bench")


@pytest.mark.parametrize(
    ("data", "code"),
    [
        ({}, "RUN_NOT_FOUND"),
        ({("r1", "eval"): {"incomplete": True}}, "RUN_INCOMPLETE"),
        ({("r1", "eval"): {"incomplete": False}, ("r1", "bench"): {"incomplete": True}}, "RUN_INCOMPLETE"),
    ],
)
def test_reject_incomplete_or_missing(data, code):
    store = FakeStore({**data, ("baseline", "eval"): {"keep": True}})

    with pytest.raises(AppError) as exc:
        promote("r1", store)

    assert exc.value.code == code
    assert store.load("baseline", "eval") == {"keep": True}
