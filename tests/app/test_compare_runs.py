import pytest

from rag_eval_harness.app.compare_runs import compare_runs
from rag_eval_harness.domain.errors import AppError
from tests.fakes import FakeStore


def _eval(f=0.8, version="v1", incomplete=False):
    return {"dataset_version": version, "incomplete": incomplete, "scores": {"total": {"faithfulness": f}}}


def _bench(p95=1000.0, version="rv1", incomplete=False):
    return {"dataset_version": version, "incomplete": incomplete, "routes": {"direct_answer": {"p95_ms": p95}}}


def _store(**runs):
    data = {}
    for name, (ev, bn) in runs.items():
        data[(name, "eval")] = ev
        if bn is not None:
            data[(name, "bench")] = bn
    return FakeStore(data)


def test_baseline_not_found():
    with pytest.raises(AppError) as exc:
        compare_runs("r1", _store(r1=(_eval(), None)))

    assert exc.value.code == "BASELINE_NOT_FOUND"


def test_run_not_found():
    with pytest.raises(AppError) as exc:
        compare_runs("r9", _store(baseline=(_eval(), None)))

    assert exc.value.code == "RUN_NOT_FOUND"


def test_scores_only_when_bench_missing():
    store = _store(baseline=(_eval(), _bench()), r1=(_eval(0.70), None))

    result = compare_runs("r1", store)

    assert result.violations == ["total.faithfulness"]


def test_bench_p95_compared_when_both_present():
    store = _store(baseline=(_eval(), _bench(1000)), r1=(_eval(), _bench(1500)))

    assert compare_runs("r1", store).violations == ["p95.direct_answer"]


def test_bench_dataset_mismatch():
    store = _store(baseline=(_eval(), _bench()), r1=(_eval(), _bench(version="other")))

    with pytest.raises(AppError) as exc:
        compare_runs("r1", store)

    assert exc.value.code == "DATASET_MISMATCH"


def test_bench_incomplete_rejected():
    store = _store(baseline=(_eval(), _bench()), r1=(_eval(), _bench(incomplete=True)))

    with pytest.raises(AppError) as exc:
        compare_runs("r1", store)

    assert exc.value.code == "RUN_INCOMPLETE"
