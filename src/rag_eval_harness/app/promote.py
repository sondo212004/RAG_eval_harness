"""Use case `promote` (AC14): đặt một run hợp lệ làm baseline."""

from ..domain.errors import AppError
from .compare_runs import BASELINE
from .ports import ResultStore


def promote(run_id: str, store: ResultStore) -> None:
    eval_run = store.load(run_id, "eval")
    if eval_run is None:
        raise AppError("RUN_NOT_FOUND", f"Không thấy results/{run_id}.json")
    bench_run = store.load(run_id, "bench")
    if eval_run["incomplete"] or (bench_run and bench_run["incomplete"]):
        raise AppError("RUN_INCOMPLETE", f"Run {run_id} có > 10% mục lỗi, không dùng làm baseline")

    store.save(BASELINE, "eval", eval_run)
    if bench_run:
        store.save(BASELINE, "bench", bench_run)
    else:  # tránh baseline eval mới đi kèm bench cũ
        store.delete(BASELINE, "bench")
