"""Use case `bench` (AC7): độ đúng route, latency p50/p95, token, cost theo từng route."""

from dataclasses import dataclass

from ..domain.dataset import filter_reviewed
from ..domain.errors import AppError
from ..domain.routing import ROUTES, RoutingCase
from ..domain.scoring import error_rate, is_incomplete, summarize_route
from .ports import ChatAdapter, ChatAnswer, Clock, IdGenerator, ResultStore, Timer, Tracer
from .retry import call_with_retry


@dataclass
class BenchDeps:
    chat: ChatAdapter
    store: ResultStore
    clock: Clock
    ids: IdGenerator
    timer: Timer
    tracer: Tracer


def _timed_ask(chat: ChatAdapter, timer: Timer, question: str, request_id: str) -> tuple[ChatAnswer, float]:
    start = timer.monotonic_ms()
    try:
        reply = chat.ask(question, request_id)
    finally:
        end = timer.monotonic_ms()
    return reply, end - start


def _bench_case(case: RoutingCase, request_id: str, deps: BenchDeps) -> dict:
    base = {"id": case.id, "expected_route": case.expected_route}
    try:
        reply, latency = call_with_retry(_timed_ask, deps.chat, deps.timer, case.question, request_id)
    except AppError as e:
        if e.code != "PROVIDER_ERROR":
            raise
        return {**base, "status": "error", "error_code": e.code}
    actual = reply.route or "unknown"
    return {
        **base,
        "status": "ok",
        "actual_route": actual,
        "correct": actual == case.expected_route,
        "latency_ms": latency,
        "tokens_in": reply.usage.input_tokens if reply.usage else None,
        "tokens_out": reply.usage.output_tokens if reply.usage else None,
    }


def run_bench(
    cases: list[RoutingCase],
    *,
    dataset_version: str,
    run_id: str,
    model: str,
    pricing: dict[str, float],
    deps: BenchDeps,
) -> dict:
    if deps.store.exists(run_id, "bench"):
        raise AppError("RUN_EXISTS", f"Bench của run {run_id} đã tồn tại, không ghi đè")

    started_at = deps.clock.now().isoformat()
    kept, excluded = filter_reviewed(cases)
    if not kept:
        raise AppError("DATASET_INVALID", "Không còn câu routing nào sau khi lọc reviewed")

    items = []
    for case in kept:
        request_id = deps.ids.new()
        row = _bench_case(case, request_id, deps)
        deps.tracer.trace("bench.item", {"requestId": request_id, "run_id": run_id, **row})
        items.append(row)

    errors = sum(1 for i in items if i["status"] == "error")
    result = {
        "run_id": run_id,
        "dataset_version": dataset_version,
        "model": model,
        "started_at": started_at,
        "pricing": pricing,
        "excluded_unreviewed": excluded,
        "error_rate": error_rate(errors, len(items)),
        "incomplete": is_incomplete(errors, len(items)),
        "routes": {
            route: summarize_route([i for i in items if i["expected_route"] == route], pricing)
            for route in ROUTES
            if any(i["expected_route"] == route for i in items)
        },
        "items": items,
    }
    deps.store.save(run_id, "bench", result)
    return result
