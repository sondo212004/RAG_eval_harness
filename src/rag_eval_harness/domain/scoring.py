"""Tổng hợp điểm RAGAS và thống kê latency/token/cost — thuần, không gọi LLM."""

import math

from .dataset import DOMAINS

METRICS = ("faithfulness", "answer_relevancy", "context_recall")
MAX_ERROR_RATE = 0.10  # lỗi > 10% → run không hợp lệ (RUN_INCOMPLETE)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank: phần tử thứ ceil(p/100 · n) của dãy đã sắp."""
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100 * len(ordered))) - 1]


def error_rate(errors: int, total: int) -> float:
    return errors / total if total else 0.0


def is_incomplete(errors: int, total: int) -> bool:
    return error_rate(errors, total) > MAX_ERROR_RATE


def aggregate_scores(scored: list[tuple[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """(domain, điểm từng metric) của các mục OK → trung bình "total" + theo domain có dữ liệu."""
    scopes = {"total": [s for _, s in scored]}
    for domain in DOMAINS:
        rows = [s for d, s in scored if d == domain]
        if rows:
            scopes[domain] = rows
    return {
        scope: {m: mean([r[m] for r in rows]) for m in METRICS if all(m in r for r in rows)}
        for scope, rows in scopes.items()
        if rows
    }


def summarize_route(items: list[dict], pricing: dict[str, float]) -> dict:
    """items cùng expected_route; mục OK có `correct`, `latency_ms`, `tokens_in|out` (None nếu adapter không trả)."""
    ok = [i for i in items if i["status"] == "ok"]
    latencies = [i["latency_ms"] for i in ok]
    has_usage = bool(ok) and all(i["tokens_in"] is not None for i in ok)
    tokens_in = sum(i["tokens_in"] for i in ok) if has_usage else None
    tokens_out = sum(i["tokens_out"] for i in ok) if has_usage else None
    cost = (
        tokens_in / 1e6 * pricing["input_per_1m"] + tokens_out / 1e6 * pricing["output_per_1m"] if has_usage else None
    )
    return {
        "count": len(ok),
        "errors": len(items) - len(ok),
        "accuracy": mean([1.0 if i["correct"] else 0.0 for i in ok]) if ok else None,
        "p50_ms": percentile(latencies, 50) if ok else None,
        "p95_ms": percentile(latencies, 95) if ok else None,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost,
    }
