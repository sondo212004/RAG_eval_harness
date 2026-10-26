"""So run hiện tại với baseline (AC4–AC6)."""

from dataclasses import dataclass, field

from .errors import AppError

METRIC_DROP_THRESHOLD = 0.05  # giảm tuyệt đối ≥ ngưỡng → fail
P95_MAX_RATIO = 1.2  # p95 > 1.2 × baseline → fail
_EPS = 1e-9  # chống sai số float ở đúng biên (0.80 - 0.75)


@dataclass
class RunSummary:
    dataset_version: str
    scores: dict[str, dict[str, float]]  # scope ("total" | domain) → metric → điểm trung bình
    p95_ms_by_route: dict[str, float] = field(default_factory=dict)
    incomplete: bool = False


@dataclass(frozen=True)
class Diff:
    key: str
    baseline: float
    current: float
    delta: float


@dataclass
class CompareResult:
    passed: bool
    violations: list[str]
    diffs: list[Diff]


def compare(baseline: RunSummary, current: RunSummary) -> CompareResult:
    if baseline.incomplete or current.incomplete:
        raise AppError("RUN_INCOMPLETE", "Run có > 10% mục lỗi, không dùng để so sánh")
    if baseline.dataset_version != current.dataset_version:
        raise AppError(
            "DATASET_MISMATCH",
            f"dataset_version khác nhau: {baseline.dataset_version} ≠ {current.dataset_version}",
        )

    diffs: list[Diff] = []
    violations: list[str] = []

    for scope, base_metrics in baseline.scores.items():
        cur_metrics = current.scores.get(scope)
        if cur_metrics is None:
            continue
        for metric, base in base_metrics.items():
            if metric not in cur_metrics:
                continue
            key = f"{scope}.{metric}"
            cur = cur_metrics[metric]
            diffs.append(Diff(key, base, cur, cur - base))
            if base - cur >= METRIC_DROP_THRESHOLD - _EPS:
                violations.append(key)

    for route, base in baseline.p95_ms_by_route.items():
        if route not in current.p95_ms_by_route:
            continue
        key = f"p95.{route}"
        cur = current.p95_ms_by_route[route]
        diffs.append(Diff(key, base, cur, cur - base))
        if cur > base * P95_MAX_RATIO + _EPS:
            violations.append(key)

    return CompareResult(passed=not violations, violations=violations, diffs=diffs)
