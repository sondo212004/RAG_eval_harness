"""Use case `report` (AC9): cập nhật bảng kết quả trong README."""

from ..domain.report import render_results, replace_between_markers


def update_readme(readme: str, eval_run: dict, bench_run: dict | None) -> str:
    return replace_between_markers(readme, render_results(eval_run, bench_run))
