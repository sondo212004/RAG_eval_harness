"""Bộ câu routing cho bench (AC7)."""

from dataclasses import dataclass

from .dataset import ensure_unique_ids
from .errors import AppError

ROUTES = ("rag_retrieval", "direct_answer")
REQUIRED_FIELDS = ("id", "question", "expected_route", "reviewed")


@dataclass(frozen=True)
class RoutingCase:
    id: str
    question: str
    expected_route: str
    reviewed: bool


def parse_routing_set(records: list[dict]) -> list[RoutingCase]:
    if not records:
        raise AppError("DATASET_INVALID", "Bộ routing rỗng")

    errors = []
    for i, r in enumerate(records):
        item = r.get("id") or f"#{i}"
        missing = [f for f in REQUIRED_FIELDS if r.get(f) in (None, "")]
        errors += [{"item": item, "field": f, "rule": "required"} for f in missing]
        if "expected_route" not in missing and r["expected_route"] not in ROUTES:
            errors.append({"item": item, "field": "expected_route", "rule": "enum"})
    if errors:
        raise AppError("DATASET_INVALID", f"{len(errors)} lỗi trong bộ routing", errors)

    ensure_unique_ids(records)
    return [RoutingCase(r["id"], r["question"], r["expected_route"], r["reviewed"]) for r in records]
