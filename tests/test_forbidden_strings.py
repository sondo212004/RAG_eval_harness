"""AC11 — file được git track không chứa chuỗi cấm.

Danh sách cấm KHÔNG commit: đọc từ env `FORBIDDEN_STRINGS` (CI: secret, mỗi dòng một chuỗi)
hoặc `config/forbidden_strings.txt` (local, đã .gitignore). Không có danh sách → skip ở local,
CI đặt `REQUIRE_FORBIDDEN_LIST=1` để biến skip thành fail.
"""

import os
import subprocess
from pathlib import Path

import pytest

from rag_eval_harness.domain.forbidden import find_forbidden, parse_needles

ROOT = Path(__file__).resolve().parents[1]


def _needles():
    if os.environ.get("FORBIDDEN_STRINGS"):
        return parse_needles(os.environ["FORBIDDEN_STRINGS"])
    local = ROOT / "config" / "forbidden_strings.txt"
    return parse_needles(local.read_text(encoding="utf-8")) if local.exists() else []


def test_tracked_files_clean():
    needles = _needles()
    if not needles:
        assert not os.environ.get("REQUIRE_FORBIDDEN_LIST"), "CI phải cấu hình FORBIDDEN_STRINGS"
        pytest.skip("chưa có danh sách chuỗi cấm")

    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    files = {}
    for rel in tracked:
        try:
            files[rel] = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    files.update({f"<path>{rel}": rel for rel in tracked})  # tên file cũng không được chứa

    assert find_forbidden(files, needles) == []
