"""Bicubic B-spline lookup of an ESDF patch inside a CasADi graph."""
import casadi as ca
import numpy as np

from .constants import NVBLOX_MAX_DISTANCE_M


def uniform_knots(n_ctrl, degree, resolution):
    """Uniform knots that put control point i at i * resolution (its Greville abscissa), so each
    cell's value sits where the cell is. The knot span reaches 0.2 m beyond the patch."""
    shift = (degree + 1) // 2
    return [(j - shift) * resolution for j in range(n_ctrl + degree + 1)]


class EsdfSpline:
    def __init__(self, patch_size, patch_resolution, spline_degree):
        self.patch_size = patch_size
        self.patch_resolution = patch_resolution
        self.spline_degree = spline_degree
        self.half_patch = patch_size // 2
        self.n_coeffs = patch_size * patch_size
        self._knots = uniform_knots(patch_size, spline_degree, patch_resolution)

    def pack(self, esdf_patch):
        """Flattened spline coefficients for a (patch_size, patch_size) patch.

        Shifted down by NVBLOX_MAX_DISTANCE_M, and shifted back up in distance(): ca.bspline
        returns 0 outside its knot span whatever the coefficients are, and the shift makes
        that 0 read as the max distance (free space) instead of an obstacle surface. Inside
        the span the two shifts cancel."""
        patch = np.asarray(esdf_patch, dtype=float).reshape(self.patch_size, self.patch_size)
        return (patch - NVBLOX_MAX_DISTANCE_M).ravel(order='C')

    def distance(self, p_xy, p_x0_xy, psi, coeffs, delta):
        """ESDF distance at odom-frame point `p_xy`, for a patch cut around odom-frame `p_x0_xy`
        (`coeffs` from pack()). `psi` is the yaw of the odom frame in the ESDF frame; only the
        offset from p_x0_xy is rotated, since the translation cancels. `delta` is p_x0_xy's
        offset from the patch's centre cell in the ESDF frame (EsdfMap.get_patch)."""
        offset_odom = p_xy - p_x0_xy
        c, s = ca.cos(psi), ca.sin(psi)
        rotation = ca.vertcat(ca.horzcat(c, -s), ca.horzcat(s, c))
        offset_map = rotation @ offset_odom
        centre = self.half_patch * self.patch_resolution
        query = ca.vertcat(centre + offset_map[1] + delta[1], centre + offset_map[0] + delta[0])
        knots = self._knots
        return ca.bspline(query, coeffs, [knots, knots], [self.spline_degree] * 2, 1, {}) + NVBLOX_MAX_DISTANCE_M
