"""Tracing (AC8): luôn ghi JSONL local; có LANGSMITH_API_KEY thì đẩy thêm lên LangSmith."""

import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path

from ..app.ports import Clock, Tracer

log = logging.getLogger(__name__)
DEFAULT_PROJECT = "rag-eval-harness"


class LocalJsonTracer:
    def __init__(self, path: Path, clock: Clock):
        self.path = path
        self.clock = clock

    def trace(self, name: str, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": self.clock.now().isoformat(), "name": name, **payload}, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


class LangSmithTracer:
    def __init__(self, client, project: str):
        self.client = client
        self.project = project

    def trace(self, name: str, payload: dict) -> None:
        try:
            self.client.create_run(name=name, run_type="chain", inputs=payload, project_name=self.project)
        except Exception as e:  # trace phụ trợ — không được làm hỏng run
            log.warning(json.dumps({"event": "langsmith_failed", "error": type(e).__name__, **_rid(payload)}))


class MultiTracer:
    def __init__(self, *tracers: Tracer):
        self.tracers = tracers

    def trace(self, name: str, payload: dict) -> None:
        for t in self.tracers:
            t.trace(name, payload)


def _rid(payload: dict) -> dict:
    return {"requestId": payload["requestId"]} if "requestId" in payload else {}


def _langsmith_client(api_key: str):
    from langsmith import Client

    return Client(api_key=api_key)


def build_tracer(
    env: Mapping[str, str],
    trace_file: Path,
    clock: Clock,
    langsmith_factory: Callable[[str], object] = _langsmith_client,
) -> Tracer:
    local = LocalJsonTracer(trace_file, clock)
    api_key = env.get("LANGSMITH_API_KEY")
    if not api_key:
        return local
    remote = LangSmithTracer(langsmith_factory(api_key), env.get("LANGSMITH_PROJECT", DEFAULT_PROJECT))
    return MultiTracer(local, remote)
