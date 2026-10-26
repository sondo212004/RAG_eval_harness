import hashlib

from rag_eval_harness.app.fetch import fetch_corpus
from rag_eval_harness.domain.errors import AppError
from tests.fakes import FakeDownloader, FakeSink


def _entry(name, content):
    return {"name": name, "url": f"https://x/{name}", "sha256": hashlib.sha256(content).hexdigest()}


# AC12
def test_checksum_mismatch_skipped_batch_continues():
    manifest = [_entry("a.pdf", b"A"), _entry("b.pdf", b"B"), _entry("c.pdf", b"C")]
    downloader = FakeDownloader({"https://x/a.pdf": b"A", "https://x/b.pdf": b"TAMPERED", "https://x/c.pdf": b"C"})
    sink = FakeSink()

    report = fetch_corpus(manifest, downloader, sink)

    assert sorted(sink.saved) == ["a.pdf", "c.pdf"]
    assert report.saved == ["a.pdf", "c.pdf"]
    assert report.skipped == [{"name": "b.pdf", "code": "CORPUS_CHECKSUM_MISMATCH", "reason": "sha256 mismatch"}]


def test_download_error_skipped_batch_continues():
    manifest = [_entry("a.pdf", b"A"), _entry("b.pdf", b"B")]
    downloader = FakeDownloader({"https://x/a.pdf": AppError("DOWNLOAD_FAILED", "404"), "https://x/b.pdf": b"B"})

    report = fetch_corpus(manifest, downloader, FakeSink())

    assert report.saved == ["b.pdf"]
    assert report.skipped == [{"name": "a.pdf", "code": "CORPUS_CHECKSUM_MISMATCH", "reason": "404"}]
