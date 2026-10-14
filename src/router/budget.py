"""Spending guardrails for the answering endpoints.

Routing reads surface features, so a client can reshape questions to raise
its own share of large-model calls (SECURITY.md), and any service can be hit
by more traffic than its budget allows. Two limits bound the cost:

- LARGE_CALLS_PER_CLIENT_PER_HOUR: once a client has used this many
  large-model calls (routed or escalated) in the past hour, its requests are
  served by the small model, with no escalation, until the window frees up.
- DAILY_SPEND_LIMIT_USD: once the service has spent this much today (local
  date), every request is served by the small model for the rest of the day.

Over a limit the service degrades instead of refusing: the client still gets
an answer, marked budget_limited. 0 turns a limit off. Clients are told
apart by address; behind an authenticating gateway you would key them by API
key instead. State is in memory, per process.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from datetime import date

WINDOW_S = 3600.0


class SpendGuard:
    def __init__(self, large_calls_per_hour: int, daily_limit_usd: float, clock=time.time, today=None):
        self.large_calls_per_hour = large_calls_per_hour
        self.daily_limit_usd = daily_limit_usd
        self.clock = clock
        self.today = today or (lambda: date.today().isoformat())
        self._lock = threading.Lock()
        self._large_calls: dict[str, deque[float]] = defaultdict(deque)
        self._day, self._spent = "", 0.0

    def _roll_day(self) -> None:
        if (day := self.today()) != self._day:
            self._day, self._spent = day, 0.0

    def check(self, client: str) -> str | None:
        """None if the client may use the large model, else the limit that applies."""
        with self._lock:
            self._roll_day()
            if self.daily_limit_usd > 0 and self._spent >= self.daily_limit_usd:
                return "daily_spend_limit"
            if self.large_calls_per_hour > 0:
                calls = self._large_calls[client]
                while calls and calls[0] <= self.clock() - WINDOW_S:
                    calls.popleft()
                if len(calls) >= self.large_calls_per_hour:
                    return "client_large_call_limit"
        return None

    def record(self, client: str, result) -> None:
        """Count a finished request's spend and its large-model calls."""
        with self._lock:
            self._roll_day()
            self._spent += result.total_cost_usd
            now = self.clock()
            for call in result.calls:
                if call.tier == "large":
                    self._large_calls[client].append(now)

    def status(self) -> dict:
        with self._lock:
            self._roll_day()
            return {"spent_today_usd": round(self._spent, 6),
                    "daily_limit_usd": self.daily_limit_usd or None,
                    "large_calls_per_client_per_hour": self.large_calls_per_hour or None}
