"""Backs the robot straight up when the CBF slack stays high, i.e. when no command can satisfy the barriers."""
import numpy as np

from .constants import (
    CONTROL_LOOP_DT, SLACK_REVERSE_THRESHOLD, SLACK_REVERSE_TICKS,
    REVERSE_DISTANCE_M, REVERSE_SPEED_M_S, REVERSE_TIMEOUT_FACTOR,
)


class ReverseRecovery:
    """Counts consecutive ticks with slack above `slack_threshold`. After `slack_ticks` of them it
    drives (-speed, 0) until odometry shows `distance` travelled from where it started, or until
    the timeout. The reverse is blind: the ESDF has no view behind the robot."""

    def __init__(self, slack_threshold=SLACK_REVERSE_THRESHOLD, slack_ticks=SLACK_REVERSE_TICKS,
                 distance=REVERSE_DISTANCE_M, speed=REVERSE_SPEED_M_S, dt=CONTROL_LOOP_DT,
                 timeout_factor=REVERSE_TIMEOUT_FACTOR):
        self.slack_threshold = slack_threshold
        self.slack_ticks = slack_ticks
        self.distance = distance
        self.speed = speed
        self._max_reverse_ticks = int(np.ceil(timeout_factor * distance / speed / dt))
        self._high_slack_ticks = 0
        self._reverse_ticks = 0
        self._start_xy = None

    @property
    def reversing(self):
        return self._start_xy is not None

    def update(self, pose, slack):
        """(v, omega) to publish while reversing, else None (use the filter output)."""
        xy = np.asarray(pose[:2], dtype=float)
        if self._start_xy is None:
            self._high_slack_ticks = self._high_slack_ticks + 1 if slack > self.slack_threshold else 0
            if self._high_slack_ticks < self.slack_ticks:
                return None
            self._start_xy = xy
            self._reverse_ticks = 0
            self._high_slack_ticks = 0
        self._reverse_ticks += 1
        if np.linalg.norm(xy - self._start_xy) >= self.distance or self._reverse_ticks > self._max_reverse_ticks:
            self._start_xy = None
            return None
        return -self.speed, 0.0
