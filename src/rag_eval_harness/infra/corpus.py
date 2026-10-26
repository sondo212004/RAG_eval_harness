"""Corpus: manifest JSONL, tải qua HTTP (`fetch`) hoặc lấy từ zip local (`prepare`), trích text."""

import io
import json
import zipfile
from pathlib import Path

import httpx

from ..domain.errors import AppError

REQUIRED = ("name", "domain", "sha256")


def load_manifest(path: Path) -> list[dict]:
    entries = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = []
    for i, e in enumerate(entries):
        item = e.get("name") or f"#{i}"
        errors += [{"item": item, "field": f, "rule": "required"} for f in REQUIRED if not e.get(f)]
        if not (e.get("url") or e.get("zip_path")):
            errors.append({"item": item, "field": "url|zip_path", "rule": "required"})
    if errors:
        raise AppError("CONFIG_INVALID", f"Manifest {path} không hợp lệ", errors)
    return entries


def url_entries(entries: list[dict]) -> list[dict]:
    return [{"name": e["name"], "url": e["url"], "sha256": e["sha256"]} for e in entries if e.get("url")]


def prepare_entries(entries: list[dict]) -> list[dict]:
    """Mục chỉ có trong zip (không còn URL gốc) → dạng fetch_corpus hiểu, url = đường dẫn trong zip."""
    return [
        {"name": e["name"], "url": e["zip_path"], "sha256": e["sha256"]}
        for e in entries
        if e.get("zip_path") and not e.get("url")
    ]


class HttpDownloader:
    def __init__(self, client: httpx.Client):
        self.client = client

    def get(self, url: str) -> bytes:
        try:
            resp = self.client.get(url, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise AppError("DOWNLOAD_FAILED", f"{type(e).__name__}: {url}") from e
        return resp.content


class ZipReader:
    def __init__(self, zip_path: Path):
        self.zip_path = zip_path

    def get(self, inner_path: str) -> bytes:
        try:
            with zipfile.ZipFile(self.zip_path) as z:
                return z.read(inner_path)
        except (KeyError, FileNotFoundError, zipfile.BadZipFile) as e:
            raise AppError("DOWNLOAD_FAILED", f"Không đọc được {inner_path} trong zip") from e


class DirSink:
    def __init__(self, root: Path):
        self.root = Path(root)

    def save(self, name: str, content: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / name).write_bytes(content)


def extract_text(name: str, content: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".txt":
        return content.decode("utf-8", errors="replace")
    if suffix in (".html", ".htm"):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text("\n", strip=True)
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)
    raise AppError("CORPUS_UNSUPPORTED", f"Định dạng không hỗ trợ: {name}")
