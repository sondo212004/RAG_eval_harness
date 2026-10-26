import json

from rag_eval_harness.infra.tracing import build_tracer
from tests.fakes import FakeClock


class FakeLangSmith:
    def __init__(self, fail=False):
        self.runs = []
        self.fail = fail

    def create_run(self, **kwargs):
        if self.fail:
            raise RuntimeError("network down")
        self.runs.append(kwargs)


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# AC8
def test_no_key_local_only(tmp_path):
    created = []
    trace_file = tmp_path / "traces" / "r1.jsonl"

    tracer = build_tracer({}, trace_file, FakeClock(), langsmith_factory=lambda key: created.append(key))
    tracer.trace("eval.item", {"requestId": "req-1", "id": "a"})

    assert created == []
    assert _read(trace_file) == [
        {"ts": "2026-10-26T09:00:00+00:00", "name": "eval.item", "requestId": "req-1", "id": "a"}
    ]


def test_key_sends_to_fake_client(tmp_path):
    fake = FakeLangSmith()
    env = {"LANGSMITH_API_KEY": "secret", "LANGSMITH_PROJECT": "rag-eval"}
    trace_file = tmp_path / "r1.jsonl"

    tracer = build_tracer(env, trace_file, FakeClock(), langsmith_factory=lambda key: fake)
    tracer.trace("eval.item", {"requestId": "req-1"})

    assert fake.runs == [
        {
            "name": "eval.item",
            "run_type": "chain",
            "inputs": {"requestId": "req-1"},
            "project_name": "rag-eval",
        }
    ]
    assert len(_read(trace_file)) == 1  # vẫn ghi local


def test_langsmith_failure_does_not_break_run(tmp_path):
    env = {"LANGSMITH_API_KEY": "secret"}
    trace_file = tmp_path / "r1.jsonl"

    tracer = build_tracer(env, trace_file, FakeClock(), langsmith_factory=lambda key: FakeLangSmith(fail=True))
    tracer.trace("eval.item", {"requestId": "req-1"})

    assert len(_read(trace_file)) == 1


def test_api_key_never_written_to_local_trace(tmp_path):
    env = {"LANGSMITH_API_KEY": "secret"}
    trace_file = tmp_path / "r1.jsonl"

    build_tracer(env, trace_file, FakeClock(), langsmith_factory=lambda key: FakeLangSmith()).trace("x", {})

    assert "secret" not in trace_file.read_text(encoding="utf-8")
