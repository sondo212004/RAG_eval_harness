"""Use case `eval` (AC2, AC3): hỏi adapter → chấm RAGAS → ghi results/<run_id>.json."""

from dataclasses import dataclass

from ..domain.dataset import QAItem, filter_reviewed
from ..domain.errors import AppError
from ..domain.scoring import aggregate_scores, error_rate, is_incomplete
from .ports import ChatAdapter, Clock, IdGenerator, Judge, ResultStore, Tracer
from .retry import call_with_retry


@dataclass
class EvalDeps:
    chat: ChatAdapter
    judge: Judge
    store: ResultStore
    clock: Clock
    ids: IdGenerator
    tracer: Tracer


def _eval_item(item: QAItem, request_id: str, deps: EvalDeps) -> dict:
    base = {"id": item.id, "domain": item.domain}
    try:
        reply = call_with_retry(deps.chat.ask, item.question, request_id)
        scores = call_with_retry(deps.judge.score, item.question, reply.answer, reply.contexts, item.ground_truth)
    except AppError as e:
        if e.code != "PROVIDER_ERROR":
            raise
        return {**base, "status": "error", "error_code": e.code}
    return {**base, "status": "ok", "scores": scores}


def run_eval(
    items: list[QAItem],
    *,
    dataset_version: str,
    run_id: str,
    model: str,
    deps: EvalDeps,
    include_unreviewed: bool = False,
) -> dict:
    if deps.store.exists(run_id, "eval"):
        raise AppError("RUN_EXISTS", f"Run {run_id} đã tồn tại, không ghi đè")

    started_at = deps.clock.now().isoformat()
    kept, excluded = (list(items), 0) if include_unreviewed else filter_reviewed(items)
    if not kept:
        raise AppError("DATASET_INVALID", "Không còn mục nào sau khi lọc reviewed")

    results = []
    for item in kept:
        request_id = deps.ids.new()
        row = _eval_item(item, request_id, deps)
        deps.tracer.trace("eval.item", {"requestId": request_id, "run_id": run_id, **row})
        results.append(row)

    errors = sum(1 for r in results if r["status"] == "error")
    result = {
        "run_id": run_id,
        "dataset_version": dataset_version,
        "model": model,
        "judge_model": deps.judge.model,
        "started_at": started_at,
        "excluded_unreviewed": excluded,
        "error_rate": error_rate(errors, len(results)),
        "incomplete": is_incomplete(errors, len(results)),
        "scores": aggregate_scores([(r["domain"], r["scores"]) for r in results if r["status"] == "ok"]),
        "items": results,
    }
    deps.store.save(run_id, "eval", result)
    return result
