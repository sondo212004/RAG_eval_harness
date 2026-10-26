import re

from rag_eval_harness.infra.system import PerfTimer, SystemClock, UuidIds, new_run_id
from tests.fakes import FakeClock, FakeIds


def test_run_id_sortable_by_time():
    assert new_run_id(FakeClock(), FakeIds("abcd")) == "20261026-0900-abcd"


def test_real_implementations_shape():
    assert SystemClock().now().tzinfo is not None
    assert re.fullmatch(r"[0-9a-f]{32}", UuidIds().new())
    t = PerfTimer()
    assert t.monotonic_ms() <= t.monotonic_ms()
