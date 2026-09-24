import casadi as ca
import numpy as np

from ..common.constants import (
    PATCH_SIZE, PATCH_RESOLUTION, SPLINE_DEGREE, SAFETY_MARGIN_M, V_MAX, OMEGA_MAX,
)
from ..common.esdf_spline import EsdfSpline
from .constants import (
    LOOKAHEAD_M, CBF_EXTRA_MARGIN_M, CBF_ALPHA, CBF_WEIGHT_V, CBF_WEIGHT_OMEGA, CBF_SLACK_WEIGHT,
)


class CbfFilter:
    """CBF-QP safety filter for a unicycle. Returns the command closest to a nominal
    [v, omega] that keeps a point LOOKAHEAD_M ahead of the robot outside the ESDF safety
    margin, or the gentlest violation (via the slack) if that is not achievable. A fresh
    QP is built and solved per call."""

    def __init__(self, patch_size=PATCH_SIZE, patch_resolution=PATCH_RESOLUTION,
                 spline_degree=SPLINE_DEGREE, safety_margin=SAFETY_MARGIN_M + CBF_EXTRA_MARGIN_M,
                 lookahead=LOOKAHEAD_M, alpha=CBF_ALPHA,
                 weight_v=CBF_WEIGHT_V, weight_omega=CBF_WEIGHT_OMEGA,
                 slack_weight=CBF_SLACK_WEIGHT, v_max=V_MAX, omega_max=OMEGA_MAX):
        self.patch_size = patch_size
        self.patch_resolution = patch_resolution
        self.half_patch = patch_size // 2
        self._esdf = EsdfSpline(patch_size, patch_resolution, spline_degree)
        self.n_coeffs = self._esdf.n_coeffs
        self.safety_margin = safety_margin
        self.lookahead = lookahead
        self.alpha = alpha
        self.v_max = v_max
        self.omega_max = omega_max

        # Parameter layout: [x0(3) | psi(1) | esdf_coeffs(n_coeffs) | u_nom(2)]
        self.n_params = 3 + 1 + self.n_coeffs + 2

        qp, self.lbw, self.ubw, self.lbg, self.ubg = self._build_qp(weight_v, weight_omega, slack_weight)
        self.solver = ca.qpsol('cbf_qp', 'qrqp', qp, {'print_iter': False, 'print_header': False})

    def _barrier(self, pose, anchor_xy, psi, coeffs):
        """h = ESDF distance at the lookahead point of `pose` - safety margin, for a patch
        centred on `anchor_xy`. The anchor must stay fixed when h is differentiated: the patch
        moves with the robot, so differentiating with the anchor tied to the pose would cancel
        the position dependence and hide the effect of driving forward."""
        theta = pose[2]
        lookahead_point = pose[0:2] + self.lookahead * ca.vertcat(ca.cos(theta), ca.sin(theta))
        return self._esdf.distance(lookahead_point, anchor_xy, psi, coeffs) - self.safety_margin

    def _build_qp(self, weight_v, weight_omega, slack_weight):
        P = ca.MX.sym('P', self.n_params)
        p_x0 = P[0:3]
        p_psi = P[3]
        p_coeffs = P[4:4 + self.n_coeffs]
        p_u_nom = P[4 + self.n_coeffs: 6 + self.n_coeffs]

        u = ca.MX.sym('u', 2)      # [v, omega]
        s = ca.MX.sym('s')         # CBF constraint slack, >= 0
        w = ca.vertcat(u, s)

        # ca.gradient needs a purely symbolic argument, and p_x0 is a slice of P: differentiate
        # against a standalone pose symbol (the patch anchor stays p_x0), then substitute p_x0 in.
        pose_sym = ca.MX.sym('pose_sym', 3)
        h_sym = self._barrier(pose_sym, p_x0[0:2], p_psi, p_coeffs)
        grad_h_sym = ca.gradient(h_sym, pose_sym)   # dh/d[x, y, theta]
        h, grad_h = ca.substitute([h_sym, grad_h_sym], [pose_sym], [p_x0])
        theta = p_x0[2]

        # The unicycle is control-affine with no drift: xdot = g(x) u, with g's columns
        # [cos(theta), sin(theta), 0] for v and [0, 0, 1] for omega, so dh/dt = grad_h . g . u.
        g_v = ca.vertcat(ca.cos(theta), ca.sin(theta), 0)
        g_omega = ca.vertcat(0, 0, 1)
        Lg_h_v = ca.dot(grad_h, g_v)
        Lg_h_omega = ca.dot(grad_h, g_omega)

        # Slacked CBF condition: dh/dt + s >= -alpha * h
        cbf_constraint = Lg_h_v * u[0] + Lg_h_omega * u[1] + s + self.alpha * h

        u_err = u - p_u_nom
        W = ca.diag(ca.vertcat(weight_v, weight_omega))
        f = u_err.T @ W @ u_err + slack_weight * s**2

        qp = {'x': w, 'p': P, 'f': f, 'g': cbf_constraint}
        lbw = [0.0, -self.omega_max, 0.0]
        ubw = [self.v_max, self.omega_max, ca.inf]
        lbg = [0.0]
        ubg = [ca.inf]
        return qp, lbw, ubw, lbg, ubg

    def pack_params(self, x0, psi, esdf_patch, u_nom):
        return np.concatenate([
            np.asarray(x0, dtype=float), [psi], self._esdf.pack(esdf_patch),
            np.asarray(u_nom, dtype=float),
        ])

    def filter(self, x0, psi, esdf_patch, u_nom):
        """Returns (u_safe, h, slack): the safe [v, omega], the barrier value at the lookahead
        point (positive = outside the margin), and the slack the constraint needed (0 normally)."""
        p = self.pack_params(x0, psi, esdf_patch, u_nom)
        sol = self.solver(p=p, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg,
                          x0=np.concatenate([u_nom, [0.0]]))
        w_opt = np.array(sol['x']).flatten()
        u_safe, slack = w_opt[0:2], float(w_opt[2])

        h = float(self._h_function()(x0, psi, p[4:4 + self.n_coeffs]))
        return u_safe, h, slack

    def _h_function(self):
        if not hasattr(self, '_h_fn_cache'):
            x0 = ca.MX.sym('x0', 3)
            psi = ca.MX.sym('psi')
            coeffs = ca.MX.sym('coeffs', self.n_coeffs)
            self._h_fn_cache = ca.Function('h', [x0, psi, coeffs], [self._barrier(x0, x0[0:2], psi, coeffs)])
        return self._h_fn_cache


if __name__ == '__main__':
    cbf = CbfFilter()
    x0 = np.array([0.0, 0.0, 0.0])
    psi = 0.0
    esdf_patch = np.full((cbf.patch_size, cbf.patch_size), 2.0)  # free space
    u_nom = np.array([0.3, 0.0])
    u_safe, h, slack = cbf.filter(x0, psi, esdf_patch, u_nom)
    print(f"free space: u_safe={u_safe}, h={h:.3f}, slack={slack:.4f}")

    # An obstacle at the lookahead point.
    row0 = cbf.half_patch
    col0 = cbf.half_patch + round(cbf.lookahead / cbf.patch_resolution)
    esdf_patch[row0, col0] = -0.5
    u_safe, h, slack = cbf.filter(x0, psi, esdf_patch, u_nom)
    print(f"obstacle ahead: u_safe={u_safe}, h={h:.3f}, slack={slack:.4f}")
