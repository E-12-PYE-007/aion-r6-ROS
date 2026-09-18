"""Ground-truth reference path for testing mpc_path_follower: straight -> left 90deg arc ->
straight -> right 90deg arc -> straight, walked by arc length. Test fixture only.
"""
import numpy as np


class ChicanePath:
    def __init__(self):
        # (kind, length_or_radius, turn_angle) — turn_angle > 0 is left/CCW, < 0 is right/CW.
        raw_segments = [
            ('straight', 2.0, None),
            ('arc', 1.0, np.pi / 2),
            ('straight', 1.0, None),
            ('arc', 1.0, -np.pi / 2),
            ('straight', 2.0, None),
        ]

        self._segments = []  # (kind, arc_length, start_pose, curvature_or_None)
        pose = np.array([0.0, 0.0, 0.0])
        for kind, param, turn_angle in raw_segments:
            if kind == 'straight':
                length = param
                self._segments.append(('straight', length, pose.copy(), None))
                pose = pose + np.array([length * np.cos(pose[2]), length * np.sin(pose[2]), 0.0])
            else:
                radius = param
                length = radius * abs(turn_angle)
                curvature = turn_angle / length  # = +-1/radius
                self._segments.append(('arc', length, pose.copy(), curvature))
                theta0 = pose[2]
                pose = np.array([
                    pose[0] + (np.sin(theta0 + turn_angle) - np.sin(theta0)) / curvature,
                    pose[1] - (np.cos(theta0 + turn_angle) - np.cos(theta0)) / curvature,
                    theta0 + turn_angle,
                ])
        self.total_length = sum(seg[1] for seg in self._segments)

        # Dense lookup table for nearest_arclength()'s brute-force projection.
        self._lut_s = np.linspace(0.0, self.total_length, 2000)
        self._lut_xy = np.array([self.pose_at_arclength(s)[:2] for s in self._lut_s])

    def pose_at_arclength(self, s):
        """Pose at arc length s along the path. Clamped to [0, total_length] (holds the end pose)."""
        s = float(np.clip(s, 0.0, self.total_length))
        covered = 0.0
        for kind, length, start_pose, curvature in self._segments:
            if s <= covered + length:
                local_s = s - covered
                x0, y0, theta0 = start_pose
                if kind == 'straight':
                    return np.array([x0 + local_s * np.cos(theta0), y0 + local_s * np.sin(theta0), theta0])
                dtheta = curvature * local_s
                return np.array([
                    x0 + (np.sin(theta0 + dtheta) - np.sin(theta0)) / curvature,
                    y0 - (np.cos(theta0 + dtheta) - np.cos(theta0)) / curvature,
                    theta0 + dtheta,
                ])
            covered += length
        return self._segments[-1][2]  # unreachable given the clip above; last segment's start pose as a fallback

    def nearest_arclength(self, xy):
        """Arc length of the path point closest to xy (brute-force over the lookup table)."""
        d2 = np.sum((self._lut_xy - np.asarray(xy)) ** 2, axis=1)
        return self._lut_s[np.argmin(d2)]
