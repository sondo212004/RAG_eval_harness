"""Cài đặt thật của Clock / IdGenerator / Timer — chỉ dùng ở rìa (CLI), test tiêm fake."""

import time
import uuid
from datetime import UTC, datetime

from ..app.ports import Clock, IdGenerator


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class UuidIds:
    def new(self) -> str:
        return uuid.uuid4().hex


class PerfTimer:
    def monotonic_ms(self) -> float:
        return time.perf_counter() * 1000


def new_run_id(clock: Clock, ids: IdGenerator) -> str:
    """YYYYMMDD-HHMM-xxxx (UTC): sắp chuỗi = sắp thời gian, đuôi ngẫu nhiên tránh trùng."""
    return f"{clock.now():%Y%m%d-%H%M}-{ids.new()[:4]}"
