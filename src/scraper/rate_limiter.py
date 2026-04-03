from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Simple rate limiter with minimum interval between requests."""

    def __init__(self, min_interval: float = 1.0):
        """
        Args:
            min_interval: Minimum seconds between requests.
        """
        self.min_interval = min_interval
        self._last_request_time: float = 0.0

    def wait(self) -> None:
        """Block until the minimum interval has passed since the last request."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.monotonic()

    async def async_wait(self) -> None:
        """Async version of wait."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self._last_request_time = time.monotonic()
