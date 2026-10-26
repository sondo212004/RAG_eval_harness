"""Render bảng kết quả Markdown và chèn vào README giữa 2 marker (AC9)."""

from .dataset import DOMAINS
from .errors import AppError
from .scoring import METRICS

START = "<!-- results:start -->"
END = "<!-- results:end -->"
EMPTY = "—"


def _fmt(v, spec: str) -> str:
    return EMPTY if v is None else format(v, spec)


def render_results(eval_run: dict, bench_run: dict | None) -> str:
    lines = [
        f"Run `{eval_run['run_id']}` · dataset `{eval_run['dataset_version']}` · "
        f"model `{eval_run['model']}` · judge `{eval_run['judge_model']}` · {eval_run['started_at'][:10]}",
        "",
        "| Scope | " + " | ".join(METRICS) + " |",
        "|---" * (len(METRICS) + 1) + "|",
    ]
    for scope in ("total", *DOMAINS):
        scores = eval_run["scores"].get(scope, {})
        lines.append(f"| {scope} | " + " | ".join(_fmt(scores.get(m), ".3f") for m in METRICS) + " |")

    if bench_run:
        lines += [
            "",
            "| Route | n | Route accuracy | p50 ms | p95 ms | tokens in | tokens out | cost |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for route, r in bench_run["routes"].items():
            cost = EMPTY if r["cost_usd"] is None else f"${r['cost_usd']:.4f}"
            lines.append(
                f"| {route} | {r['count']} | {_fmt(r['accuracy'], '.1%')} | {_fmt(r['p50_ms'], '.0f')} | "
                f"{_fmt(r['p95_ms'], '.0f')} | {_fmt(r['tokens_in'], 'd')} | {_fmt(r['tokens_out'], 'd')} | {cost} |"
            )
    return "\n".join(lines) + "\n"


def replace_between_markers(text: str, block: str) -> str:
    start, end = text.find(START), text.find(END)
    if start == -1 or end == -1 or end < start:
        raise AppError("REPORT_MARKERS_MISSING", f"README thiếu cặp marker {START} … {END}")
    return text[: start + len(START)] + "\n" + block + text[end:]
