"""AC10 — domain/ thuần: không import app/, infra/ hay thư viện I/O."""

import ast
from pathlib import Path

PACKAGE = "rag_eval_harness"
DOMAIN_DIR = Path(__file__).resolve().parents[1] / "src" / PACKAGE / "domain"

FORBIDDEN_PREFIXES = (
    f"{PACKAGE}.app",
    f"{PACKAGE}.infra",
    # I/O, thời gian, ngẫu nhiên — phải tiêm từ ngoài vào
    "os",
    "io",
    "pathlib",
    "shutil",
    "subprocess",
    "socket",
    "time",
    "random",
    "uuid",
    "logging",
    "sqlite3",
    "urllib",
    "http",
    # thư viện bên ngoài
    "httpx",
    "requests",
    "yaml",
    "qdrant_client",
    "google",
    "ragas",
    "langsmith",
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_text_splitters",
    "langgraph",
    "sentence_transformers",
    "flashrank",
    "pypdf",
    "bs4",
    "torch",
)


def _imports(tree: ast.AST, module: str) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # import tương đối → quy về tuyệt đối
                base = module.rsplit(".", node.level)[0]
                names.append(f"{base}.{node.module}" if node.module else base)
            else:
                names.append(node.module)
    return names


def find_violations(root: Path) -> list[str]:
    violations = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root.parent).with_suffix("")
        module = ".".join((PACKAGE, *rel.parts))
        for name in _imports(ast.parse(path.read_text(encoding="utf-8")), module):
            if any(name == p or name.startswith(p + ".") for p in FORBIDDEN_PREFIXES):
                violations.append(f"{path.name}: {name}")
    return violations


def test_domain_has_no_forbidden_imports():
    assert DOMAIN_DIR.is_dir()
    assert find_violations(DOMAIN_DIR) == []


def test_checker_detects_violation(tmp_path):
    domain = tmp_path / "domain"
    domain.mkdir()
    (domain / "bad.py").write_text("from ..infra import llm\nimport httpx\n", encoding="utf-8")

    assert find_violations(domain) == ["bad.py: rag_eval_harness.infra", "bad.py: httpx"]
