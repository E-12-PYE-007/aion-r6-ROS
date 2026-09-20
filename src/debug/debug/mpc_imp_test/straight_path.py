"""Straight-line ground-truth reference path for testing mpc_path_follower - replaces
ChicanePath for this test (no benefit to exercising curves right now). Test fixture only.
"""
import numpy as np

# Chosen against esdf_single_obs's recorded ESDF: checked directly that this line
# crosses real (non-sentinel) negative values around x=4.1-4.4, not just the known-cell
# boundary or the single-voxel noise spike found earlier.
PATH_ORIGIN = (2.0, -3.5)
PATH_HEADING = 0.0


class StraightPath:
    def __init__(self, origin, heading):
        self.origin = np.asarray(origin, dtype=float)
        self.direction = np.array([np.cos(heading), np.sin(heading)])

    def pose_at_arclength(self, s):
        xy = self.origin + s * self.direction
        heading = np.arctan2(self.direction[1], self.direction[0])
        return np.array([xy[0], xy[1], heading])

    def project(self, xy):
        """(arclength, perpendicular distance) of xy's projection onto the line."""
        offset = np.asarray(xy, dtype=float) - self.origin
        s = np.dot(offset, self.direction)
        perp = offset - s * self.direction
        return s, float(np.linalg.norm(perp))

    def lookahead_point(self, xy, lookahead):
        """Point on the line exactly `lookahead` from xy, ahead along the line's own
        direction. If xy has drifted further than `lookahead` off the line, falls back
        to the closest point on the line (no point at the exact lookahead distance
        exists ahead in that case)."""
        s_proj, perp_dist = self.project(xy)
        reach = max(lookahead ** 2 - min(perp_dist, lookahead) ** 2, 0.0) ** 0.5
        return self.pose_at_arclength(s_proj + reach)[:2]
