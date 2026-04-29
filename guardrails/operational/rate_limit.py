"""
guardrails/operational/rate_limit.py
=====================================
Sliding-window rate limiter.

Prevents abuse, runaway agentic loops, and accidental API cost explosions
by capping the number of requests a given user_id can make within a
rolling time window.

Implementation notes
--------------------
- Uses a deque per user_id to store timestamps of recent calls.
- Thread-safe via a single threading.Lock.
- For multi-process deployments (e.g. gunicorn with multiple workers),
  replace the in-process deque with a Redis-backed implementation using
  redis-py and the ZADD / ZREMRANGEBYSCORE / ZCARD commands.

Usage
-----
    # Initialise ONCE at module/app level (not per request):
    limiter = RateLimiter(max_calls=10, window_seconds=60)

    # Check on each incoming request:
    if not limiter.is_allowed(user_id):
        return guardrail_rejection("rate_limited")
"""

import time
from collections import defaultdict, deque
from threading import Lock


class RateLimiter:
    """
    Sliding-window rate limiter, thread-safe for a single process.

    Parameters
    ----------
    max_calls:       Maximum number of allowed calls within *window_seconds*.
    window_seconds:  Length of the sliding time window in seconds.

    Example
    -------
    >>> limiter = RateLimiter(max_calls=3, window_seconds=10)
    >>> limiter.is_allowed("user_1")
    True
    >>> limiter.is_allowed("user_1")
    True
    >>> limiter.is_allowed("user_1")
    True
    >>> limiter.is_allowed("user_1")  # 4th call within window — blocked
    False
    """

    def __init__(self, max_calls: int, window_seconds: int) -> None:
        self.max_calls = max_calls
        self.window = window_seconds
        # Per-user deque of UNIX timestamps for calls within the window
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def is_allowed(self, user_id: str) -> bool:
        """
        Return True if *user_id* is within their rate limit, False otherwise.

        Also records the current call if allowed (i.e. calling this method
        both checks AND consumes one slot from the budget).
        """
        now = time.time()
        with self._lock:
            dq = self._calls[user_id]
            # Evict timestamps that have fallen outside the sliding window
            while dq and dq[0] < now - self.window:
                dq.popleft()
            if len(dq) >= self.max_calls:
                return False          # limit reached — blocked
            dq.append(now)            # record this call
            return True

    def remaining(self, user_id: str) -> int:
        """
        Return the number of calls *user_id* can still make in the current window.
        Useful for displaying a "X requests remaining" indicator in a UI.
        """
        now = time.time()
        with self._lock:
            dq = self._calls[user_id]
            valid = sum(1 for t in dq if t >= now - self.window)
            return max(0, self.max_calls - valid)

    def reset(self, user_id: str) -> None:
        """
        Clear all recorded calls for *user_id*.
        Useful in tests or for admin-level overrides.
        """
        with self._lock:
            self._calls[user_id].clear()
