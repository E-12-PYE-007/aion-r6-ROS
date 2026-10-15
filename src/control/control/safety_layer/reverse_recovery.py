"""Backs the robot up when it is stuck: either the CBF slack stays high (no command can satisfy
the barriers), or the robot is pinned at the clearance boundary, where standing still is feasible
(so slack stays ~0) but the nominal command still asks to drive forward. While reversing the robot
also angles gently away from the binding obstacle, using the ESDF gradient the CBF filter reports,
so it does not back straight into the same deadlock on the next approach."""
import numpy as np

from .constants import (
    CONTROL_LOOP_DT, SLACK_REVERSE_THRESHOLD, SLACK_REVERSE_TICKS,
    REVERSE_DISTANCE_M, REVERSE_SPEED_M_S, REVERSE_TIMEOUT_FACTOR,
    STALL_REVERSE_TICKS, STALL_SPEED_EPS, STALL_NOMINAL_V_EPS,
    REVERSE_TURN_OMEGA, REVERSE_MAX_YAW,
)


def _wrap(angle):
    """Wrap an angle to (-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class ReverseRecovery:
    """Counts consecutive 'stuck' ticks and, after enough of them, drives backward until odometry
    shows `distance` travelled from where it started, or until the timeout. A tick counts as stuck
    two ways, each with its own counter:
      - slack: CBF slack above `slack_threshold` for `slack_ticks` ticks (no feasible command), or
      - stall: the nominal asks for forward motion (v_nom > nominal_v_eps) while the filtered output
        is effectively stationary (|v| < speed_eps and |omega| < speed_eps) for `stall_ticks` ticks.
    While reversing, the robot arcs toward the open direction: at the moment the reverse starts it
    freezes a turn sign from `away_yaw` (the odom-frame heading of the ESDF gradient, i.e. away from
    the binding obstacle) and applies `turn_omega` in that direction until the heading has changed
    by `max_yaw`, then finishes straight back. The sign is frozen (not recomputed per tick) to avoid
    chasing a moving gradient. The reverse is otherwise blind: the ESDF has no view behind the robot,
    so the arc is kept small and short."""

    def __init__(self, slack_threshold=SLACK_REVERSE_THRESHOLD, slack_ticks=SLACK_REVERSE_TICKS,
                 distance=REVERSE_DISTANCE_M, speed=REVERSE_SPEED_M_S, dt=CONTROL_LOOP_DT,
                 timeout_factor=REVERSE_TIMEOUT_FACTOR, stall_ticks=STALL_REVERSE_TICKS,
                 stall_speed_eps=STALL_SPEED_EPS, stall_nominal_v_eps=STALL_NOMINAL_V_EPS,
                 turn_omega=REVERSE_TURN_OMEGA, max_yaw=REVERSE_MAX_YAW):
        self.slack_threshold = slack_threshold
        self.slack_ticks = slack_ticks
        self.distance = distance
        self.speed = speed
        self.stall_ticks = stall_ticks
        self.stall_speed_eps = stall_speed_eps
        self.stall_nominal_v_eps = stall_nominal_v_eps
        self.turn_omega = turn_omega
        self.max_yaw = max_yaw
        self._max_reverse_ticks = int(np.ceil(timeout_factor * distance / speed / dt))
        self._high_slack_ticks = 0
        self._stall_ticks = 0
        self._reverse_ticks = 0
        self._start_xy = None
        self._start_yaw = None
        self._turn_sign = 0.0

    @property
    def reversing(self):
        return self._start_xy is not None

    def update(self, pose, slack, u_nom, u_safe, away_yaw=None):
        """(v, omega) to publish while reversing, else None (use the filter output).
        `u_nom` is the pure-pursuit (v, omega) before the filter; `u_safe` is the filtered output.
        `away_yaw` is the odom-frame heading pointing away from the binding obstacle (from the CBF
        filter's ESDF gradient), or None if it is unreliable -- then the reverse goes straight back."""
        xy = np.asarray(pose[:2], dtype=float)
        yaw = float(pose[2])
        if self._start_xy is None:
            self._high_slack_ticks = self._high_slack_ticks + 1 if slack > self.slack_threshold else 0
            stalled = (u_nom[0] > self.stall_nominal_v_eps
                       and abs(u_safe[0]) < self.stall_speed_eps
                       and abs(u_safe[1]) < self.stall_speed_eps)
            self._stall_ticks = self._stall_ticks + 1 if stalled else 0
            if self._high_slack_ticks < self.slack_ticks and self._stall_ticks < self.stall_ticks:
                return None
            self._start_xy = xy
            self._start_yaw = yaw
            self._turn_sign = 0.0 if away_yaw is None else float(np.sign(_wrap(away_yaw - yaw)))
            self._reverse_ticks = 0
            self._high_slack_ticks = 0
            self._stall_ticks = 0
        self._reverse_ticks += 1
        if np.linalg.norm(xy - self._start_xy) >= self.distance or self._reverse_ticks > self._max_reverse_ticks:
            self._start_xy = None
            return None
        turning = self._turn_sign != 0.0 and abs(_wrap(yaw - self._start_yaw)) < self.max_yaw
        omega = self._turn_sign * self.turn_omega if turning else 0.0
        return -self.speed, omega
