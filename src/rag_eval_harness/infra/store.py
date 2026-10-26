"""Lưu kết quả: results/<run_id>.json (eval) và results/<run_id>.bench.json (SPEC-001 Vòng 8)."""

import json
from pathlib import Path

from ..app.ports import Kind

_SUFFIX = {"eval": ".json", "bench": ".bench.json"}


class FileResultStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, name: str, kind: Kind) -> Path:
        return self.root / f"{name}{_SUFFIX[kind]}"

    def exists(self, name: str, kind: Kind) -> bool:
        return self._path(name, kind).exists()

    def load(self, name: str, kind: Kind) -> dict | None:
        path = self._path(name, kind)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def save(self, name: str, kind: Kind, payload: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        # sort_keys + LF để file commit vào git có diff ổn định
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self._path(name, kind).write_text(text, encoding="utf-8", newline="\n")

    def delete(self, name: str, kind: Kind) -> None:
        self._path(name, kind).unlink(missing_ok=True)

    def latest_run_id(self) -> str | None:
        """run_id dạng YYYYMMDD-HHMM-xxxx nên sắp chuỗi = sắp thời gian."""
        names = [
            p.name.removesuffix(".json")
            for p in self.root.glob("*.json")
            if not p.name.endswith(".bench.json") and p.name != "baseline.json"
        ]
        return max(names, default=None)
