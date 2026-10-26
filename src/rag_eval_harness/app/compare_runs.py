"""Use case `compare`: ghép file eval + bench thành RunSummary rồi so với baseline."""

from ..domain.compare import CompareResult, RunSummary, compare
from ..domain.errors import AppError
from .ports import ResultStore

BASELINE = "baseline"


def _summary(eval_run: dict, bench_run: dict | None) -> RunSummary:
    return RunSummary(
        dataset_version=eval_run["dataset_version"],
        scores=eval_run["scores"],
        p95_ms_by_route={r: v["p95_ms"] for r, v in bench_run["routes"].items() if v["p95_ms"] is not None}
        if bench_run
        else {},
        incomplete=eval_run["incomplete"] or bool(bench_run and bench_run["incomplete"]),
    )


def compare_runs(run_id: str, store: ResultStore) -> CompareResult:
    base_eval = store.load(BASELINE, "eval")
    if base_eval is None:
        raise AppError("BASELINE_NOT_FOUND", "Không thấy results/baseline.json — chạy `make promote RUN=<id>`")
    cur_eval = store.load(run_id, "eval")
    if cur_eval is None:
        raise AppError("RUN_NOT_FOUND", f"Không thấy results/{run_id}.json")

    base_bench, cur_bench = store.load(BASELINE, "bench"), store.load(run_id, "bench")
    if not (base_bench and cur_bench):  # thiếu một phía → chỉ so RAGAS
        base_bench = cur_bench = None
    elif base_bench["dataset_version"] != cur_bench["dataset_version"]:
        raise AppError("DATASET_MISMATCH", "Bộ routing của bench khác baseline")

    return compare(_summary(base_eval, base_bench), _summary(cur_eval, cur_bench))
