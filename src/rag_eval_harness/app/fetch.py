"""Use case `fetch` (AC12): tải corpus theo manifest; file lỗi bị bỏ qua, không dừng cả lô."""

from dataclasses import dataclass, field

from ..domain.corpus import checksum_ok
from ..domain.errors import AppError
from .ports import CorpusSink, Downloader

SKIP_CODE = "CORPUS_CHECKSUM_MISMATCH"


@dataclass
class FetchReport:
    saved: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)


def fetch_corpus(manifest: list[dict], downloader: Downloader, sink: CorpusSink) -> FetchReport:
    report = FetchReport()
    for entry in manifest:
        try:
            content = downloader.get(entry["url"])
        except AppError as e:
            report.skipped.append({"name": entry["name"], "code": SKIP_CODE, "reason": e.message})
            continue
        if not checksum_ok(content, entry["sha256"]):
            report.skipped.append({"name": entry["name"], "code": SKIP_CODE, "reason": "sha256 mismatch"})
            continue
        sink.save(entry["name"], content)
        report.saved.append(entry["name"])
    return report
