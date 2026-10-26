import hashlib
import zipfile

import pytest

from rag_eval_harness.domain.errors import AppError
from rag_eval_harness.infra.corpus import DirSink, ZipReader, extract_text, load_manifest, prepare_entries


def test_extract_text_from_html_drops_markup_and_scripts():
    html = "<html><script>var x=1</script><body><h1>QCVN 01</h1><p>Nước sạch</p></body></html>".encode()

    text = extract_text("a.html", html)

    assert "QCVN 01" in text and "Nước sạch" in text
    assert "var x" not in text


def test_extract_text_txt_utf8():
    assert extract_text("a.txt", "Bơm ly tâm".encode()) == "Bơm ly tâm"


def test_unsupported_extension():
    with pytest.raises(AppError) as exc:
        extract_text("a.docx", b"")

    assert exc.value.code == "CORPUS_UNSUPPORTED"


def test_load_manifest_jsonl_requires_fields(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        '{"name": "a.pdf", "domain": "om", "sha256": "x", "url": "https://x"}\n{"name": "b.pdf"}\n', "utf-8"
    )

    with pytest.raises(AppError) as exc:
        load_manifest(path)

    assert exc.value.code == "CONFIG_INVALID"
    assert {"item": "b.pdf", "field": "sha256", "rule": "required"} in exc.value.details


def test_prepare_reads_from_zip_with_same_checksum_rules(tmp_path):
    zpath = tmp_path / "data.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("public/legal/a.html", b"<p>A</p>")
    entries = [
        {
            "name": "a.html",
            "domain": "legal",
            "sha256": hashlib.sha256(b"<p>A</p>").hexdigest(),
            "zip_path": "public/legal/a.html",
        },
        {"name": "web.pdf", "domain": "om", "sha256": "x", "url": "https://x/web.pdf"},
    ]

    prepared = prepare_entries(entries)
    reader = ZipReader(zpath)

    assert prepared == [{"name": "a.html", "url": "public/legal/a.html", "sha256": entries[0]["sha256"]}]
    assert reader.get("public/legal/a.html") == b"<p>A</p>"
    with pytest.raises(AppError) as exc:
        reader.get("public/missing.pdf")
    assert exc.value.code == "DOWNLOAD_FAILED"


def test_dir_sink_writes_bytes(tmp_path):
    DirSink(tmp_path / "corpus").save("a.pdf", b"%PDF")

    assert (tmp_path / "corpus" / "a.pdf").read_bytes() == b"%PDF"
