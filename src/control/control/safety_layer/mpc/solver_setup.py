import casadi as ca
import numpy as np

from ..common.constants import (
    PATCH_SIZE, PATCH_RESOLUTION, SPLINE_DEGREE, SAFETY_MARGIN_M, V_MAX, OMEGA_MAX,
)
from ..common.esdf_spline import EsdfSpline
from .constants import (
    SLACK_WEIGHT, PREDICTION_DT, CONTROL_HORIZON_M, PREDICTION_HORIZON_N,
    Q_DIAG, R_DIAG, QF, IPOPT_MAX_ITER, LOW_V_WEIGHT, LOW_V_THRESHOLD,
)


class UnicycleMPC:
    """Multiple-shooting NMPC for a unicycle model [x, y, theta], solved with IPOPT via CasADi."""

    def __init__(self, dt=PREDICTION_DT, M=CONTROL_HORIZON_M, N=PREDICTION_HORIZON_N,
                 v_max=V_MAX, omega_max=OMEGA_MAX,
                 Q=None, R=None, Q_f=None, solver_opts=None,
                 patch_size=PATCH_SIZE, patch_resolution=PATCH_RESOLUTION, spline_degree=SPLINE_DEGREE,
                 safety_margin=SAFETY_MARGIN_M, slack_weight=SLACK_WEIGHT,
                 low_v_weight=LOW_V_WEIGHT, low_v_threshold=LOW_V_THRESHOLD):
        self.dt = dt
        self.M = M                                     # control horizon
        self.N = N                                     # prediction horizon
        self.v_max = v_max
        self.omega_max = omega_max
        self.Q = np.diag(Q_DIAG) if Q is None else np.asarray(Q)
        self.R = np.diag(R_DIAG) if R is None else np.asarray(R)
        self.Q_f = np.asarray(QF) if Q_f is None else np.asarray(Q_f)

        self.patch_size = patch_size                   # must match EsdfMap
        self.patch_resolution = patch_resolution
        self._esdf = EsdfSpline(patch_size, patch_resolution, spline_degree)
        self.n_coeffs = self._esdf.n_coeffs
        self.safety_margin = safety_margin
        self.slack_weight = slack_weight
        self.low_v_weight = low_v_weight
        self.low_v_threshold = low_v_threshold

        # Parameter layout:
        # [x0(3) | x_ref_0..N (3*(N+1)) | u_ref_0..N-1 (2*N) | psi(1) | esdf_coeffs (n_coeffs)]
        self.n_params = 3 + 3 * (N + 1) + 2 * N + 1 + self.n_coeffs

        self.f = self._build_dynamics()
        nlp, self.lbw, self.ubw, self.lbg, self.ubg, self.w0_default = self._build_nlp()
        self.n_w = nlp['x'].shape[0]
        self.n_g = nlp['g'].shape[0]

        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': IPOPT_MAX_ITER,
            'ipopt.warm_start_init_point': 'yes',
            # Adaptive: the default monotone strategy resets the barrier parameter every solve,
            # which makes warm-started solves slower than cold ones.
            'ipopt.mu_strategy': 'adaptive',
            'print_time': False,
        }
        if solver_opts:
            opts.update(solver_opts)
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.reset()

    def _build_dynamics(self):
        # Forward Euler unicycle
        x = ca.MX.sym('x', 3)
        u = ca.MX.sym('u', 2)
        x_next = x + self.dt * ca.vertcat(
            u[0] * ca.cos(x[2]),            # v*cos(theta)
            u[0] * ca.sin(x[2]),            # v*sin(theta)
            u[1],                           # omega
        )
        return ca.Function('f', [x, u], [x_next])

    def _build_nlp(self):
        N, M = self.N, self.M
        P = ca.MX.sym('P', self.n_params)
        p_x0 = P[0:3]

        def p_x_ref(k):
            i = 3 + 3 * k
            return P[i:i + 3]

        def p_u_ref(k):
            i = 3 + 3 * (N + 1) + 2 * k
            return P[i:i + 2]

        idx_psi = 3 + 3 * (N + 1) + 2 * N
        p_psi = P[idx_psi]                                       # odom -> ESDF frame yaw
        p_coeffs = P[idx_psi + 1: idx_psi + 1 + self.n_coeffs]   # ESDF patch spline coefficients
        p_x0_xy = p_x0[0:2]                                      # odom-frame point the patch is centred on

        w = []      # decision variables (X, U and S)
        w0 = []     # default initial guess
        lbw = []    # lower bound on decision variables
        ubw = []    # upper bound on decision variables
        J = 0       # cost function
        g = []      # constraint equations
        lbg = []    # lower bound on constraints
        ubg = []    # upper bound on constraints

        X_k = ca.MX.sym('X_0', 3)
        w.append(X_k)
        w0 += [0, 0, 0]
        lbw += [-ca.inf, -ca.inf, -ca.inf]
        ubw += [ca.inf, ca.inf, ca.inf]
        g.append(X_k - p_x0)                # X_0 is the measured state
        lbg += [0, 0, 0]
        ubg += [0, 0, 0]

        for k in range(N):
            # U_k is a decision variable up to M, after that U_{M-1} is reused
            if k < M:
                U_k = ca.MX.sym(f'U_{k}', 2)
                w.append(U_k)
                w0 += [0, 0]
                lbw += [0, -self.omega_max]
                ubw += [self.v_max, self.omega_max]

            # Stage cost
            x_err = X_k - p_x_ref(k)
            u_err = U_k - p_u_ref(k)
            J += x_err.T @ self.Q @ x_err + u_err.T @ self.R @ u_err

            # Soft low-velocity floor
            if self.low_v_weight:
                v_deficit = ca.fmax(0.0, self.low_v_threshold - U_k[0])
                J += self.low_v_weight * v_deficit**2

            # Propagate dynamics
            X_next = self.f(X_k, U_k)

            # Next state decision variable
            X_k = ca.MX.sym(f'X_{k + 1}', 3)
            w.append(X_k)
            w0 += [0, 0, 0]
            lbw += [-ca.inf, -ca.inf, -ca.inf]
            ubw += [ca.inf, ca.inf, ca.inf]

            # Shooting constraint
            g.append(X_k - X_next)
            lbg += [0, 0, 0]
            ubg += [0, 0, 0]

            # Obstacle constraint on X_{k+1}, slacked: a robot already inside the margin
            # would make a hard constraint infeasible however U_0 is chosen, at every tick.
            S_k = ca.MX.sym(f'S_{k}')
            w.append(S_k)
            w0 += [0.0]
            lbw += [0.0]
            ubw += [ca.inf]

            dist_k = self._esdf.distance(X_k[0:2], p_x0_xy, p_psi, p_coeffs)
            g.append(dist_k + S_k - self.safety_margin)
            lbg += [0]
            ubg += [ca.inf]
            J += self.slack_weight * S_k**2

        # Terminal cost
        x_err_N = X_k - p_x_ref(N)
        J += x_err_N.T @ self.Q_f @ x_err_N

        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        nlp = {'f': J, 'x': w, 'p': P, 'g': g}
        return nlp, lbw, ubw, lbg, ubg, w0

    def reset(self):
        """Clear the warm start. Call after a discontinuity such as a failed solve."""
        self._w_guess = np.array(self.w0_default, dtype=float)
        self._lam_x0 = None
        self._lam_g0 = None

    def pack_params(self, x0, x_ref, u_ref, psi, esdf_patch):
        """psi: odom -> ESDF frame yaw. esdf_patch: (patch_size, patch_size), centred on x0."""
        x_ref = np.asarray(x_ref, dtype=float).reshape(self.N + 1, 3)
        u_ref = np.asarray(u_ref, dtype=float).reshape(self.N, 2)
        return np.concatenate([
            np.asarray(x0, dtype=float), x_ref.flatten(), u_ref.flatten(),
            [psi], self._esdf.pack(esdf_patch),
        ])

    def extract_predicted_states(self, w_opt):
        """Predicted X_0..X_N from the decision vector."""
        states = [w_opt[0:3]]
        idx = 3
        for k in range(self.N):
            if k < self.M:
                idx += 2  # skip U_k
            states.append(w_opt[idx:idx + 3])
            idx += 3
            idx += 1  # skip S_k
        return np.array(states)

    def extract_slacks(self, w_opt):
        """S_k for k = 1..N. At the optimum S_k = max(0, safety_margin - dist_k), the margin
        violation in metres; nonzero means the margin is being paid through, not respected."""
        idx = 3
        slacks = []
        for k in range(self.N):
            if k < self.M:
                idx += 2  # skip U_k
            idx += 3  # skip X_{k+1}
            slacks.append(w_opt[idx])
            idx += 1
        return np.array(slacks)

    def solve(self, x0, x_ref, u_ref, psi, esdf_patch, warm_start=True):
        """Solve one MPC step. Returns (u0, predicted_states, solved_ok)."""
        p = self.pack_params(x0, x_ref, u_ref, psi, esdf_patch)

        kwargs = dict(x0=self._w_guess, lbx=self.lbw, ubx=self.ubw,
                      lbg=self.lbg, ubg=self.ubg, p=p)
        if warm_start and self._lam_x0 is not None:
            kwargs['lam_x0'] = self._lam_x0
            kwargs['lam_g0'] = self._lam_g0

        sol = self.solver(**kwargs)
        solved_ok = bool(self.solver.stats()['success'])

        w_opt = np.array(sol['x']).flatten()
        self._w_guess = w_opt
        self._lam_x0 = sol['lam_x']
        self._lam_g0 = sol['lam_g']

        predicted_states = self.extract_predicted_states(w_opt)
        u0 = w_opt[3:5]
        return u0, predicted_states, solved_ok


if __name__ == '__main__':
    mpc = UnicycleMPC()
    print(f"NLP: {mpc.n_w} decision vars, {mpc.n_g} constraints, {mpc.n_params} parameters")

    x0 = [0.0, 0.0, 0.0]
    x_ref = np.tile([1.0, 0.0, 0.0], (mpc.N + 1, 1))
    u_ref = np.tile([0.2, 0.0], (mpc.N, 1))
    psi = 0.0
    esdf_patch = np.full((mpc.patch_size, mpc.patch_size), 2.0)  # free space
    u0, predicted_states, ok = mpc.solve(x0, x_ref, u_ref, psi, esdf_patch)
    print(f"solved_ok={ok}, u0={u0}")
    print(f"final predicted state: {predicted_states[-1]}")
