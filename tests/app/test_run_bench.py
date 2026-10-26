import pytest

from rag_eval_harness.app.ports import Usage
from rag_eval_harness.app.run_bench import BenchDeps, run_bench
from rag_eval_harness.domain.errors import AppError
from rag_eval_harness.domain.routing import RoutingCase
from tests.fakes import FakeChat, FakeClock, FakeIds, FakeStore, FakeTimer, FakeTracer, answer, provider_error

PRICING = {"input_per_1m": 0.30, "output_per_1m": 2.50}


def _case(id, expected, reviewed=True):
    return RoutingCase(id, f"q-{id}", expected, reviewed)


def _deps(chat, latencies, store=None):
    return BenchDeps(
        chat=chat,
        store=store or FakeStore(),
        clock=FakeClock(),
        ids=FakeIds(),
        timer=FakeTimer(latencies),
        tracer=FakeTracer(),
    )


def _run(cases, deps):
    return run_bench(cases, dataset_version="rv1", run_id="r1", model="gen-model", pricing=PRICING, deps=deps)


# AC7
def test_route_accuracy_latency_tokens_cost_per_route():
    cases = [_case("a", "rag_retrieval"), _case("b", "rag_retrieval"), _case("c", "direct_answer")]
    chat = FakeChat(
        {
            "q-a": answer(route="rag_retrieval", usage=Usage(1000, 200)),
            "q-b": answer(route="direct_answer", usage=Usage(3000, 400)),
            "q-c": answer(route="direct_answer", usage=Usage(100, 50)),
        }
    )

    result = _run(cases, _deps(chat, [100, 300, 50]))

    rag = result["routes"]["rag_retrieval"]
    assert rag["count"] == 2
    assert rag["accuracy"] == 0.5
    assert (rag["p50_ms"], rag["p95_ms"]) == (100, 300)  # nearest-rank
    assert (rag["tokens_in"], rag["tokens_out"]) == (4000, 600)
    assert rag["cost_usd"] == pytest.approx(4000 / 1e6 * 0.30 + 600 / 1e6 * 2.50)
    assert result["routes"]["direct_answer"]["accuracy"] == 1.0
    assert result["items"][1]["actual_route"] == "direct_answer"
    assert result["dataset_version"] == "rv1"
    assert result["pricing"] == PRICING


def test_missing_route_counted_wrong_as_unknown():
    chat = FakeChat({"q-a": answer(route=None)})

    result = _run([_case("a", "rag_retrieval")], _deps(chat, [10]))

    assert result["items"][0]["actual_route"] == "unknown"
    assert result["routes"]["rag_retrieval"]["accuracy"] == 0.0


def test_missing_usage_leaves_tokens_and_cost_empty():
    chat = FakeChat({"q-a": answer(route="rag_retrieval", usage=None)})

    rag = _run([_case("a", "rag_retrieval")], _deps(chat, [10]))["routes"]["rag_retrieval"]

    assert (rag["tokens_in"], rag["tokens_out"], rag["cost_usd"]) == (None, None, None)


def test_provider_error_item_excluded_from_latency_and_counted():
    cases = [_case("a", "rag_retrieval"), _case("b", "rag_retrieval")]
    chat = FakeChat({"q-a": [provider_error()], "q-b": answer(route="rag_retrieval")})

    result = _run(cases, _deps(chat, [5, 5, 5, 5, 40]))  # 4 lần lỗi rồi b

    rag = result["routes"]["rag_retrieval"]
    assert rag["errors"] == 1
    assert rag["p95_ms"] == 40
    assert result["incomplete"] is True  # 1/2 lỗi


def test_unreviewed_cases_excluded():
    cases = [_case("a", "direct_answer"), _case("b", "direct_answer", reviewed=False)]
    chat = FakeChat({"q-a": answer(route="direct_answer")})

    result = _run(cases, _deps(chat, [10]))

    assert result["excluded_unreviewed"] == 1
    assert [i["id"] for i in result["items"]] == ["a"]


def test_saved_as_bench_kind_and_not_overwritten():
    store = FakeStore()
    chat = FakeChat({"q-a": answer(route="direct_answer")})
    _run([_case("a", "direct_answer")], _deps(chat, [10], store))

    assert store.exists("r1", "bench")
    with pytest.raises(AppError) as exc:
        _run([_case("a", "direct_answer")], _deps(chat, [10], store))
    assert exc.value.code == "RUN_EXISTS"
