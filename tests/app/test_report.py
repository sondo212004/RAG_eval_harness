import pytest

from rag_eval_harness.app.report import update_readme
from rag_eval_harness.domain.errors import AppError

EVAL = {
    "run_id": "r1",
    "dataset_version": "v1",
    "model": "gen-model",
    "judge_model": "judge-model",
    "started_at": "2026-10-26T09:00:00+00:00",
    "scores": {
        "total": {"faithfulness": 0.8, "answer_relevancy": 0.85, "context_recall": 0.7},
        "legal": {"faithfulness": 0.81234, "answer_relevancy": 0.9, "context_recall": 0.75},
    },
}
BENCH = {
    "routes": {
        "rag_retrieval": {
            "count": 20,
            "accuracy": 0.95,
            "p50_ms": 2100,
            "p95_ms": 4300,
            "tokens_in": 40000,
            "tokens_out": 6000,
            "cost_usd": 0.027,
            "errors": 0,
        },
    }
}
README = "# Title\n\nintro\n\n<!-- results:start -->\nold\n<!-- results:end -->\n\nfooter\n"


# AC9
def test_inserts_between_markers():
    out = update_readme(README, EVAL, BENCH)

    head, rest = out.split("<!-- results:start -->\n")
    block, tail = rest.split("<!-- results:end -->")
    assert head == "# Title\n\nintro\n\n"
    assert tail == "\n\nfooter\n"
    assert "old" not in block
    assert "| legal | 0.812 | 0.900 | 0.750 |" in block
    assert "| hardware | — | — | — |" in block
    assert "| rag_retrieval | 20 | 95.0% | 2100 | 4300 | 40000 | 6000 | $0.0270 |" in block
    assert "`r1`" in block and "`v1`" in block


def test_idempotent():
    once = update_readme(README, EVAL, BENCH)

    assert update_readme(once, EVAL, BENCH) == once


def test_without_bench_omits_bench_table():
    assert "rag_retrieval" not in update_readme(README, EVAL, None)


def test_markers_missing():
    with pytest.raises(AppError) as exc:
        update_readme("# no markers\n", EVAL, None)

    assert exc.value.code == "REPORT_MARKERS_MISSING"
