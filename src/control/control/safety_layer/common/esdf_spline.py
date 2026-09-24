"""Bicubic B-spline lookup of an ESDF patch inside a CasADi graph."""
import casadi as ca
import numpy as np

from .constants import NVBLOX_MAX_DISTANCE_M


def clamped_uniform_knots(n_ctrl, degree, extent):
    """Clamped uniform knots for a degree-`degree` B-spline with `n_ctrl` control points over [0, extent]."""
    n_interior = n_ctrl - degree - 1
    interior = list(np.linspace(0, extent, n_interior + 2)[1:-1]) if n_interior > 0 else []
    return [0.0] * (degree + 1) + interior + [float(extent)] * (degree + 1)


class EsdfSpline:
    def __init__(self, patch_size, patch_resolution, spline_degree):
        self.patch_size = patch_size
        self.patch_resolution = patch_resolution
        self.spline_degree = spline_degree
        self.half_patch = patch_size // 2
        self.n_coeffs = patch_size * patch_size
        self._knots = clamped_uniform_knots(patch_size, spline_degree, (patch_size - 1) * patch_resolution)

    def pack(self, esdf_patch):
        """Flattened spline coefficients for a (patch_size, patch_size) patch.

        Shifted down by NVBLOX_MAX_DISTANCE_M, and shifted back up in distance(): ca.bspline
        returns 0 outside its knot domain whatever the coefficients are, and the shift makes
        that 0 read as the max distance (free space) instead of an obstacle surface. Inside
        the domain the two shifts cancel."""
        patch = np.asarray(esdf_patch, dtype=float).reshape(self.patch_size, self.patch_size)
        return (patch - NVBLOX_MAX_DISTANCE_M).ravel(order='C')

    def distance(self, p_xy, p_x0_xy, psi, coeffs):
        """ESDF distance at odom-frame point `p_xy`, for a patch centred on odom-frame `p_x0_xy`
        (`coeffs` from pack()). `psi` is the yaw of the odom frame in the ESDF frame; only the
        offset from p_x0_xy is rotated, since the translation cancels."""
        offset_odom = p_xy - p_x0_xy
        c, s = ca.cos(psi), ca.sin(psi)
        rotation = ca.vertcat(ca.horzcat(c, -s), ca.horzcat(s, c))
        offset_map = rotation @ offset_odom
        centre = self.half_patch * self.patch_resolution
        query = ca.vertcat(centre + offset_map[0], centre + offset_map[1])
        knots = self._knots
        return ca.bspline(query, coeffs, [knots, knots], [self.spline_degree] * 2, 1, {}) + NVBLOX_MAX_DISTANCE_M
