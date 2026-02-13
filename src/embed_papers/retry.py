from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def call_with_retry(
    operation: Callable[[], T],
    *,
    is_retryable: Callable[[Exception], bool],
    max_attempts: int = 5,
    initial_delay_seconds: float = 0.5,
    max_delay_seconds: float = 8.0,
    jitter_seconds: float = 0.1,
) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_attempts or not is_retryable(exc):
                raise

            sleep_seconds = min(
                max_delay_seconds,
                initial_delay_seconds * (2 ** (attempt - 1)),
            )
            sleep_seconds += random.uniform(0.0, jitter_seconds)
            time.sleep(sleep_seconds)

    raise RuntimeError("unreachable")
