"""Quét chuỗi cấm (AC11) — danh sách cấm nằm ngoài git."""


def parse_needles(text: str) -> list[str]:
    lines = (line.strip().lower() for line in text.splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def find_forbidden(files: dict[str, str], needles: list[str]) -> list[tuple[str, int, str]]:
    """Trả (đường dẫn, số dòng, chuỗi cấm) — không phân biệt hoa thường."""
    hits = []
    for path, content in files.items():
        for line_no, line in enumerate(content.lower().splitlines(), start=1):
            hits += [(path, line_no, n) for n in needles if n in line]
    return hits
