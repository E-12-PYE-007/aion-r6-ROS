import casadi as ca
import numpy as np

from ..constants import V_MAX, OMEGA_MAX
from .constants import (
    PATCH_SIZE, PATCH_RESOLUTION, SPLINE_DEGREE,
    CBF_BODY_POINTS_M, CBF_BODY_CLEARANCE_M, CBF_ALPHA, CBF_WEIGHT_V, CBF_WEIGHT_OMEGA, CBF_SLACK_WEIGHT,
)
from .esdf_spline import EsdfSpline


class CbfFilter:
    """CBF-QP safety filter for a unicycle. Returns the command closest to a nominal
    [v, omega] that keeps every body point (base_link frame, x forward) at least `clearance`
    outside the ESDF obstacles, or the gentlest violation (via one shared slack) if that is not
    achievable. Each body point has its own barrier constraint in the same QP. A fresh QP is
    built and solved per call."""

    def __init__(self, patch_size=PATCH_SIZE, patch_resolution=PATCH_RESOLUTION,
                 spline_degree=SPLINE_DEGREE, body_points=CBF_BODY_POINTS_M,
                 clearance=CBF_BODY_CLEARANCE_M, alpha=CBF_ALPHA,
                 weight_v=CBF_WEIGHT_V, weight_omega=CBF_WEIGHT_OMEGA,
                 slack_weight=CBF_SLACK_WEIGHT, v_max=V_MAX, omega_max=OMEGA_MAX):
        self.patch_size = patch_size
        self.patch_resolution = patch_resolution
        self.half_patch = patch_size // 2
        self._esdf = EsdfSpline(patch_size, patch_resolution, spline_degree)
        self.n_coeffs = self._esdf.n_coeffs
        self.body_points = np.asarray(body_points, dtype=float).reshape(-1, 2)
        self.clearance = clearance
        self.alpha = alpha
        self.v_max = v_max
        self.omega_max = omega_max
        self.n_solve_failures = 0

        # Parameter layout: [x0(3) | psi(1) | delta(2) | esdf_coeffs(n_coeffs) | u_nom(2)]
        self.n_params = 3 + 1 + 2 + self.n_coeffs + 2

        qp, self.lbw, self.ubw, self.lbg, self.ubg = self._build_qp(weight_v, weight_omega, slack_weight)
        self.solver = ca.qpsol('cbf_qp', 'qpoases', qp, {'printLevel': 'none', 'error_on_fail': False})

    def _barrier(self, pose, anchor_xy, psi, delta, coeffs, body_point):
        """h = ESDF distance at `body_point` of `pose` - clearance, for a patch centred on
        `anchor_xy`. The anchor must stay fixed when h is differentiated: the patch moves with
        the robot, so differentiating with the anchor tied to the pose would cancel the position
        dependence and hide the effect of driving forward."""
        c, s = ca.cos(pose[2]), ca.sin(pose[2])
        point = pose[0:2] + ca.vertcat(c * body_point[0] - s * body_point[1],
                                       s * body_point[0] + c * body_point[1])
        return self._esdf.distance(point, anchor_xy, psi, coeffs, delta) - self.clearance

    def _build_qp(self, weight_v, weight_omega, slack_weight):
        P = ca.MX.sym('P', self.n_params)
        p_x0 = P[0:3]
        p_psi = P[3]
        p_delta = P[4:6]
        p_coeffs = P[6:6 + self.n_coeffs]
        p_u_nom = P[6 + self.n_coeffs: 8 + self.n_coeffs]

        u = ca.MX.sym('u', 2)      # [v, omega]
        s = ca.MX.sym('s')         # CBF constraint slack, >= 0, shared by all body points
        w = ca.vertcat(u, s)

        # The unicycle is control-affine with no drift: xdot = g(x) u, with g's columns
        # [cos(theta), sin(theta), 0] for v and [0, 0, 1] for omega, so dh/dt = grad_h . g . u.
        theta = p_x0[2]
        g_v = ca.vertcat(ca.cos(theta), ca.sin(theta), 0)
        g_omega = ca.vertcat(0, 0, 1)

        # ca.gradient needs a purely symbolic argument, and p_x0 is a slice of P: differentiate
        # against a standalone pose symbol (the patch anchor stays p_x0), then substitute p_x0 in.
        pose_sym = ca.MX.sym('pose_sym', 3)
        constraints, barriers, grads_xy = [], [], []
        for body_point in self.body_points:
            h_sym = self._barrier(pose_sym, p_x0[0:2], p_psi, p_delta, p_coeffs, body_point)
            grad_h_sym = ca.gradient(h_sym, pose_sym)   # dh/d[x, y, theta]
            h, grad_h = ca.substitute([h_sym, grad_h_sym], [pose_sym], [p_x0])
            # Slacked CBF condition: dh/dt + s >= -alpha * h
            constraints.append(ca.dot(grad_h, g_v) * u[0] + ca.dot(grad_h, g_omega) * u[1] + s + self.alpha * h)
            barriers.append(h)
            grads_xy.append(grad_h[0:2].T)              # ESDF spatial gradient (odom frame), points away from obstacle
        self._h_all = ca.Function('h_all', [P], [ca.vertcat(*barriers)])
        # Per-body-point ESDF spatial gradient; row i is (dd/dx, dd/dy) at body point i. Used to
        # steer the reverse recovery away from the binding obstacle -- a side evaluation, not part
        # of the QP solve.
        self._grad_all = ca.Function('grad_all', [P], [ca.vertcat(*grads_xy)])

        u_err = u - p_u_nom
        W = ca.diag(ca.vertcat(weight_v, weight_omega))
        f = u_err.T @ W @ u_err + slack_weight * s**2

        n = len(constraints)
        qp = {'x': w, 'p': P, 'f': f, 'g': ca.vertcat(*constraints)}
        lbw = [0.0, -self.omega_max, 0.0]
        ubw = [self.v_max, self.omega_max, ca.inf]
        return qp, lbw, ubw, [0.0] * n, [ca.inf] * n

    def pack_params(self, x0, psi, delta, esdf_patch, u_nom):
        return np.concatenate([
            np.asarray(x0, dtype=float), [psi], np.asarray(delta, dtype=float), self._esdf.pack(esdf_patch),
            np.asarray(u_nom, dtype=float),
        ])

    def filter(self, x0, psi, delta, esdf_patch, u_nom):
        """`delta` is the robot's offset from the patch's centre cell (EsdfMap.get_patch), 0 if
        the patch is centred exactly on the robot.
        Returns (u_safe, h, slack, escape_dir): the safe [v, omega], the smallest barrier value
        over the body points (positive = every point outside its clearance), the slack the
        constraints needed (0 normally), and the odom-frame ESDF gradient (dd/dx, dd/dy) at the
        binding (min-h) body point -- points away from the nearest obstacle, for the reverse
        recovery to angle toward. If the QP solver fails, u_safe is a stop and n_solve_failures
        counts it."""
        p = self.pack_params(x0, psi, delta, esdf_patch, u_nom)
        sol = self.solver(p=p, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg,
                          x0=np.concatenate([u_nom, [0.0]]))
        w_opt = np.array(sol['x']).flatten()
        barriers = np.array(self._h_all(p)).flatten()
        i_bind = int(np.argmin(barriers))
        h = float(barriers[i_bind])
        escape_dir = np.array(self._grad_all(p))[i_bind]   # (dd/dx, dd/dy) at the binding point
        if not self.solver.stats()['success'] or not np.all(np.isfinite(w_opt)):
            self.n_solve_failures += 1
            return np.zeros(2), h, 0.0, escape_dir
        return w_opt[0:2], h, float(w_opt[2]), escape_dir


if __name__ == '__main__':
    cbf = CbfFilter()
    x0 = np.array([0.0, 0.0, 0.0])
    psi = 0.0
    esdf_patch = np.full((cbf.patch_size, cbf.patch_size), 2.0)  # free space
    u_nom = np.array([0.3, 0.0])
    delta = np.zeros(2)
    u_safe, h, slack, escape_dir = cbf.filter(x0, psi, delta, esdf_patch, u_nom)
    print(f"free space: u_safe={u_safe}, h={h:.3f}, slack={slack:.4f}")

    # A disc obstacle 0.5 m ahead and 0.3 m to the left of the robot.
    rows, cols = np.mgrid[0:cbf.patch_size, 0:cbf.patch_size]
    x = (cols - cbf.half_patch) * cbf.patch_resolution
    y = (rows - cbf.half_patch) * cbf.patch_resolution
    esdf_patch = np.clip(np.hypot(x - 0.5, y - 0.3) - 0.15, -0.2, 2.0)
    u_safe, h, slack, escape_dir = cbf.filter(x0, psi, delta, esdf_patch, u_nom)
    print(f"obstacle ahead-left: u_safe={u_safe}, h={h:.3f}, slack={slack:.4f}")
