"""Validate bộ Q/A (AC1, AC2) và tính dataset_version."""

import hashlib
from collections import Counter
from dataclasses import dataclass

from .errors import AppError

DOMAINS = ("legal", "hardware", "om", "scada")  # thứ tự hiển thị trong báo cáo
REQUIRED_FIELDS = ("id", "question", "ground_truth", "gold_contexts", "domain", "source", "reviewed")


@dataclass(frozen=True)
class QAItem:
    id: str
    question: str
    ground_truth: str
    gold_contexts: tuple[str, ...]
    domain: str
    source: str
    reviewed: bool


def _record_errors(index: int, record: dict) -> list[dict]:
    item = record.get("id") or f"#{index}"
    missing = [f for f in REQUIRED_FIELDS if record.get(f) in (None, "", [])]
    errors = [{"item": item, "field": f, "rule": "required"} for f in missing]
    if "domain" not in missing and record["domain"] not in DOMAINS:
        errors.append({"item": item, "field": "domain", "rule": "enum"})
    if "reviewed" in record and not isinstance(record["reviewed"], bool):
        errors.append({"item": item, "field": "reviewed", "rule": "bool"})
    if isinstance(record.get("gold_contexts"), str):
        errors.append({"item": item, "field": "gold_contexts", "rule": "list"})
    return errors


def parse_dataset(records: list[dict]) -> list[QAItem]:
    if not records:
        raise AppError("DATASET_INVALID", "Dataset rỗng")

    errors = [e for i, r in enumerate(records) for e in _record_errors(i, r)]
    if errors:
        raise AppError("DATASET_INVALID", f"{len(errors)} lỗi trong dataset", errors)

    ensure_unique_ids(records)
    return [
        QAItem(
            id=r["id"],
            question=r["question"],
            ground_truth=r["ground_truth"],
            gold_contexts=tuple(r["gold_contexts"]),
            domain=r["domain"],
            source=r["source"],
            reviewed=r["reviewed"],
        )
        for r in records
    ]


def ensure_unique_ids(records: list[dict]) -> None:
    counts = Counter(r["id"] for r in records)
    duplicates = [i for i, n in counts.items() if n > 1]
    if duplicates:
        raise AppError(
            "DATASET_DUPLICATE_ID",
            f"id trùng: {', '.join(duplicates)}",
            [{"item": i, "field": "id", "rule": "unique"} for i in duplicates],
        )


def filter_reviewed[T](items: list[T]) -> tuple[list[T], int]:
    """Trả (mục đã duyệt, số mục bị loại)."""
    kept = [i for i in items if i.reviewed]
    return kept, len(items) - len(kept)


def dataset_version(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:12]
