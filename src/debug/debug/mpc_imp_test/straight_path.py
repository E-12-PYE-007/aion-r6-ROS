"""Straight-line ground-truth reference path for testing mpc_path_follower - replaces
ChicanePath for this test (no benefit to exercising curves right now). Test fixture only.
"""
import numpy as np

# Chosen against esdf_single_obs's recorded ESDF: PATH_ORIGIN sits at the sensed cone's
# apex (the sensor origin itself). A small, shallow obstacle (negative cells only down to
# about -0.18, not the deep saturated core of the map's other blob) occupies roughly
# x=[2.0,2.2], y=[-0.05,0.3], about 1.8m out - close enough to reach quickly, and small
# enough (a fraction of a metre across) to test avoidance against a compact obstacle
# rather than a large one.
#
# PATH_HEADING=0.0 runs dead-on through the obstacle's own edge (min ESDF distance
# along it: -0.05, i.e. inside the solid obstacle, not just inside the margin) - a
# near-worst-case, almost-exactly-on-axis scenario. Tilting the line by about -4.6
# degrees from the same origin (PATH_HEADING = -0.080904) instead sits it at y~-0.15
# by the obstacle's x-range - clear of the solid obstacle but inside SAFETY_MARGIN_M
# by about 0.10m: a near-miss needing a small correction, not a full detour discovery.
# The plain (unshaped) MPC formulation handles that case cleanly; this dead-on value
# is the harder scenario used to test whether a longer horizon can resolve it too.
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
