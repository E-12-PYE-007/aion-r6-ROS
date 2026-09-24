"""Obstacle-blind pure-pursuit command along the latest action chunk."""
import math

from .constants import (
    V_MAX, OMEGA_MAX, PP_LOOKAHEAD_DISTANCE_M, PP_HERMITE_SAMPLES_PER_SEGMENT, PP_K_CURV,
)


def sign(value):
    return -1.0 if value < 0.0 else 1.0


def saturate(linear_vel, angular_vel):
    """Limit v and omega together, scaling both so the turning radius is preserved."""
    if abs(linear_vel) <= V_MAX:
        if abs(angular_vel) <= OMEGA_MAX:
            return linear_vel, angular_vel
        rd = linear_vel / angular_vel
        return (
            OMEGA_MAX * sign(linear_vel) * abs(rd),
            OMEGA_MAX * sign(angular_vel),
        )

    if abs(angular_vel) <= 0.001:
        return V_MAX * sign(linear_vel), 0.0

    rd = linear_vel / angular_vel
    if abs(rd) >= V_MAX / OMEGA_MAX:
        return (
            V_MAX * sign(linear_vel),
            V_MAX * sign(angular_vel) / abs(rd),
        )
    return (
        OMEGA_MAX * sign(linear_vel) * abs(rd),
        OMEGA_MAX * sign(angular_vel),
    )


def generate_waypoints(relative_poses):
    """Hermite-interpolated (x, y) path through consecutive chunk waypoints, so heading
    shapes the path as well as position."""
    waypoints = []
    for i in range(len(relative_poses) - 1):
        x0, y0, theta0 = relative_poses[i].x, relative_poses[i].y, relative_poses[i].theta
        x1, y1, theta1 = relative_poses[i + 1].x, relative_poses[i + 1].y, relative_poses[i + 1].theta

        scale = math.hypot(x1 - x0, y1 - y0)
        m0x, m0y = scale * math.cos(theta0), scale * math.sin(theta0)
        m1x, m1y = scale * math.cos(theta1), scale * math.sin(theta1)

        for j in range(PP_HERMITE_SAMPLES_PER_SEGMENT):
            t = j / PP_HERMITE_SAMPLES_PER_SEGMENT
            t2, t3 = t * t, t * t * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            waypoints.append((
                h00 * x0 + h10 * m0x + h01 * x1 + h11 * m1x,
                h00 * y0 + h10 * m0y + h01 * y1 + h11 * m1y,
            ))
    return waypoints


class PurePursuit:
    """Follows an action chunk. A chunk's poses are relative to where the robot was when it
    arrived, so each new seq_num re-anchors the path at the current pose."""

    def __init__(self):
        self._last_seq = None
        self._anchor_pose = None      # pose captured when the current chunk was first seen
        self._waypoints = []          # Hermite path for the current chunk, in the anchor frame
        self._waypoint_idx = 0

    def nominal(self, action_chunk, current_pose):
        """(v, omega) toward the first path point beyond the lookahead distance;
        (0, 0) once no such point is left."""
        if self._last_seq is None or action_chunk.seq_num != self._last_seq:
            self._anchor_pose = current_pose
            self._waypoints = generate_waypoints(action_chunk.relative_poses)
            self._waypoint_idx = 0
            self._last_seq = action_chunk.seq_num

        anchor_x, anchor_y, anchor_theta = self._anchor_pose
        current_x, current_y, current_theta = current_pose

        dx = current_x - anchor_x
        dy = current_y - anchor_y
        relative_x = dx * math.cos(anchor_theta) + dy * math.sin(anchor_theta)
        relative_y = -dx * math.sin(anchor_theta) + dy * math.cos(anchor_theta)

        lookahead_x = lookahead_y = euclid_dist = 0.0
        found_lookahead = False
        for i in range(self._waypoint_idx, len(self._waypoints)):
            point_x, point_y = self._waypoints[i]
            lookahead_x = point_x - relative_x
            lookahead_y = point_y - relative_y
            euclid_dist = math.hypot(lookahead_x, lookahead_y)
            if euclid_dist > PP_LOOKAHEAD_DISTANCE_M:
                self._waypoint_idx = i
                found_lookahead = True
                break

        if not found_lookahead:
            return 0.0, 0.0

        delta_theta = current_theta - anchor_theta
        body_x = lookahead_x * math.cos(delta_theta) + lookahead_y * math.sin(delta_theta)
        body_y = -lookahead_x * math.sin(delta_theta) + lookahead_y * math.cos(delta_theta)

        curvature = 2.0 * body_y / (euclid_dist * euclid_dist)
        target_linear = V_MAX / (1.0 + PP_K_CURV * abs(curvature))

        return saturate(target_linear, curvature * target_linear)
