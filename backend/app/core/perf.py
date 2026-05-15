import time
from typing import Any

from app.core.logger_handler import logger


def perf_counter() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def log_perf(metric: str, start: float, **fields: Any) -> None:
    payload = " ".join(
        f"{key}={value}"
        for key, value in fields.items()
        if value is not None
    )
    suffix = f" {payload}" if payload else ""
    logger.info("PERF_METRIC name=%s elapsed_ms=%.2f%s", metric, elapsed_ms(start), suffix)
