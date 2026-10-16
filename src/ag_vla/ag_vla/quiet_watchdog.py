"""Tracks how long it has been since something last arrived."""


class QuietWatchdog:
    """True once nothing has arrived for longer than timeout_s. It never trips before the first arrival."""

    def __init__(self, timeout_s):
        self.timeout_s = timeout_s
        self._last_s = None

    def feed(self, now_s):
        self._last_s = now_s

    def expired(self, now_s):
        return self._last_s is not None and now_s - self._last_s > self.timeout_s
