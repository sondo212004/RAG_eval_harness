import pytest

from rag_eval_harness.domain.compare import RunSummary, compare
from rag_eval_harness.domain.errors import AppError


def _run(scores=None, p95=None, version="abc123def456", incomplete=False):
    return RunSummary(
        dataset_version=version,
        scores=scores
        or {
            "total": {"faithfulness": 0.80, "answer_relevancy": 0.85, "context_recall": 0.70},
            "legal": {"faithfulness": 0.82, "answer_relevancy": 0.86, "context_recall": 0.72},
        },
        p95_ms_by_route=p95 or {"rag_retrieval": 4000.0, "direct_answer": 1000.0},
        incomplete=incomplete,
    )


# AC4
def test_metric_drop_fails_total():
    current = _run()
    current.scores["total"]["faithfulness"] = 0.75  # giảm đúng 0.05 → vi phạm

    result = compare(_run(), current)

    assert not result.passed
    assert "total.faithfulness" in result.violations


def test_metric_drop_fails_domain():
    current = _run()
    current.scores["legal"]["context_recall"] = 0.60

    result = compare(_run(), current)

    assert not result.passed
    assert result.violations == ["legal.context_recall"]


def test_metric_drop_below_threshold_passes():
    current = _run()
    current.scores["total"]["faithfulness"] = 0.751

    assert compare(_run(), current).passed


def test_domain_missing_on_one_side_is_skipped():
    current = _run()
    del current.scores["legal"]

    assert compare(_run(), current).passed


# AC5
def test_route_p95_regression_fails():
    current = _run(p95={"rag_retrieval": 4000.0, "direct_answer": 1201.0})

    result = compare(_run(), current)

    assert not result.passed
    assert result.violations == ["p95.direct_answer"]


def test_route_p95_exactly_1_2x_passes():
    current = _run(p95={"rag_retrieval": 4800.0, "direct_answer": 1000.0})

    assert compare(_run(), current).passed


# AC6
def test_within_threshold_passes_with_diff_table():
    current = _run()
    current.scores["total"]["faithfulness"] = 0.83

    result = compare(_run(), current)

    assert result.passed
    assert result.violations == []
    row = next(d for d in result.diffs if d.key == "total.faithfulness")
    assert (row.baseline, row.current, row.delta) == (0.80, 0.83, pytest.approx(0.03))


# Mục 6
def test_dataset_mismatch_rejected():
    with pytest.raises(AppError) as exc:
        compare(_run(), _run(version="ffffffffffff"))

    assert exc.value.code == "DATASET_MISMATCH"


@pytest.mark.parametrize("side", ["baseline", "current"])
def test_incomplete_run_rejected(side):
    runs = {"baseline": _run(), "current": _run()}
    runs[side] = _run(incomplete=True)

    with pytest.raises(AppError) as exc:
        compare(runs["baseline"], runs["current"])

    assert exc.value.code == "RUN_INCOMPLETE"
