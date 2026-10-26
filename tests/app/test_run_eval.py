import pytest

from rag_eval_harness.app.run_eval import EvalDeps, run_eval
from rag_eval_harness.domain.dataset import QAItem
from rag_eval_harness.domain.errors import AppError
from tests.fakes import FakeChat, FakeClock, FakeIds, FakeJudge, FakeStore, FakeTracer, answer, provider_error


def _qa(id, domain="legal", reviewed=True):
    return QAItem(id, f"q-{id}", f"gt-{id}", ("gold",), domain, "src", reviewed)


def _scores(f, a=0.9, c=0.8):
    return {"faithfulness": f, "answer_relevancy": a, "context_recall": c}


def _deps(chat, judge=None, store=None):
    return EvalDeps(
        chat=chat,
        judge=judge or FakeJudge({q: _scores(0.8) for q in chat._answers}),
        store=store or FakeStore(),
        clock=FakeClock(),
        ids=FakeIds(),
        tracer=FakeTracer(),
    )


def _run(items, deps, **kw):
    return run_eval(items, dataset_version="v1", run_id="r1", model="gen-model", deps=deps, **kw)


# AC3
def test_result_has_per_item_domain_and_total_scores():
    items = [_qa("a"), _qa("b"), _qa("c", domain="scada")]
    chat = FakeChat({i.question: answer() for i in items})
    judge = FakeJudge({"q-a": _scores(0.6), "q-b": _scores(0.8), "q-c": _scores(1.0)})
    deps = _deps(chat, judge)

    result = _run(items, deps)

    assert result["run_id"] == "r1"
    assert result["dataset_version"] == "v1"
    assert result["model"] == "gen-model"
    assert result["judge_model"] == "judge-model"
    assert result["started_at"] == "2026-10-26T09:00:00+00:00"
    assert result["items"][0] == {"id": "a", "domain": "legal", "status": "ok", "scores": _scores(0.6)}
    assert result["scores"]["legal"]["faithfulness"] == pytest.approx(0.7)
    assert result["scores"]["scada"]["faithfulness"] == pytest.approx(1.0)
    assert result["scores"]["total"]["faithfulness"] == pytest.approx(0.8)
    assert deps.store.load("r1", "eval") == result


def test_request_id_from_id_generator_sent_to_adapter():
    chat = FakeChat({"q-a": answer()})

    _run([_qa("a")], _deps(chat))

    assert chat.calls == [("q-a", "req-1")]


# AC2
def test_unreviewed_excluded_and_count_recorded():
    items = [_qa("a"), _qa("b", reviewed=False)]
    chat = FakeChat({"q-a": answer(), "q-b": answer()})

    result = _run(items, _deps(chat))

    assert [i["id"] for i in result["items"]] == ["a"]
    assert result["excluded_unreviewed"] == 1


def test_include_unreviewed_flag_keeps_all():
    items = [_qa("a"), _qa("b", reviewed=False)]
    chat = FakeChat({"q-a": answer(), "q-b": answer()})

    result = _run(items, _deps(chat), include_unreviewed=True)

    assert result["excluded_unreviewed"] == 0
    assert len(result["items"]) == 2


# Mục 6 — PROVIDER_ERROR
def test_provider_error_retried_3_times_then_item_marked_error():
    items = [_qa("a")] + [_qa(f"ok{i}") for i in range(9)]
    answers = {i.question: answer() for i in items}
    answers["q-a"] = [provider_error()]
    chat = FakeChat(answers)

    result = _run(items, _deps(chat))

    assert sum(1 for q, _ in chat.calls if q == "q-a") == 4  # 1 lần + 3 retry
    assert result["items"][0] == {"id": "a", "domain": "legal", "status": "error", "error_code": "PROVIDER_ERROR"}
    assert result["incomplete"] is False  # 1/10 = 10%, chưa vượt


def test_transient_error_recovers():
    chat = FakeChat({"q-a": [provider_error(), provider_error(), answer()]})

    result = _run([_qa("a")], _deps(chat))

    assert result["items"][0]["status"] == "ok"


def test_error_items_excluded_from_averages():
    items = [_qa("a"), _qa("b")]
    chat = FakeChat({"q-a": [provider_error()], "q-b": answer()})
    judge = FakeJudge({"q-b": _scores(0.5)})

    result = _run(items, _deps(chat, judge))

    assert result["scores"]["total"]["faithfulness"] == pytest.approx(0.5)


# Mục 6 — RUN_INCOMPLETE
def test_error_rate_over_10_percent_marks_incomplete():
    items = [_qa(f"i{i}") for i in range(5)]
    answers = {i.question: answer() for i in items}
    answers["q-i0"] = [provider_error()]

    result = _run(items, _deps(FakeChat(answers)))

    assert result["error_rate"] == pytest.approx(0.2)
    assert result["incomplete"] is True


# Mục 6 — RUN_EXISTS
def test_run_exists_not_overwritten():
    chat = FakeChat({"q-a": answer()})
    store = FakeStore({("r1", "eval"): {"old": True}})

    with pytest.raises(AppError) as exc:
        _run([_qa("a")], _deps(chat, store=store))

    assert exc.value.code == "RUN_EXISTS"
    assert chat.calls == []
    assert store.load("r1", "eval") == {"old": True}
