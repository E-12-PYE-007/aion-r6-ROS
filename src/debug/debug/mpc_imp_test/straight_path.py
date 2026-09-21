"""Straight-line ground-truth reference path for testing mpc_path_follower - replaces
ChicanePath for this test (no benefit to exercising curves right now). Test fixture only.
"""
import numpy as np

# Chosen against esdf_single_obs's recorded ESDF: PATH_ORIGIN sits at the sensed cone's
# apex (the sensor origin itself), PATH_HEADING=0 runs straight down the cone's own central
# axis. A small, shallow obstacle (negative cells only down to about -0.18, not the deep
# saturated core of the map's other blob) sits almost exactly on that axis at x~2.0-2.1,
# about 1.8m out - close enough to reach quickly, and small enough (a fraction of a metre
# across) to test avoidance against a compact obstacle rather than a large one.
PATH_ORIGIN = (0.25, 0.0)
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
