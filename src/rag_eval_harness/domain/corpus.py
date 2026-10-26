"""Kiểm tính toàn vẹn file corpus theo manifest (AC12)."""

import hashlib


def checksum_ok(content: bytes, expected_sha256: str) -> bool:
    return hashlib.sha256(content).hexdigest() == expected_sha256.lower()
