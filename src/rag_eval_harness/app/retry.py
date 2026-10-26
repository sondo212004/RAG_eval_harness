from collections.abc import Callable

from ..domain.errors import AppError

MAX_RETRIES = 3  # SPEC-001 mục 6: retry tối đa 3 lần sau lần gọi đầu


def call_with_retry[T](fn: Callable[..., T], *args) -> T:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn(*args)
        except AppError as e:
            if e.code != "PROVIDER_ERROR" or attempt == MAX_RETRIES:
                raise
    raise AssertionError("unreachable")
