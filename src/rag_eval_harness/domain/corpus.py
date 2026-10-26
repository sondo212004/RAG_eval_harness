"""Kiểm tính toàn vẹn file corpus theo manifest (AC12)."""

import hashlib


def checksum_ok(content: bytes, expected_sha256: str) -> bool:
    return hashlib.sha256(content).hexdigest() == expected_sha256.lower()


def chunk_id(doc_id: str, position: str, text: str) -> str:
    """Id tất định (AC15): cùng tài liệu + vị trí + nội dung → cùng id. 32 hex = vừa một UUID."""
    return hashlib.sha256(f"{doc_id}|{position}|{text}".encode()).hexdigest()[:32]
