import casadi as ca
import numpy as np


class UnicycleMPC:
    """Multiple-shooting NMPC for a unicycle model [x, y, theta], solved with IPOPT via CasADi."""

    def __init__(self, dt=1 / 15, M=15, N=15, v_max=0.3, omega_max=0.3,
                 Q=None, R=None, alpha=3.0, solver_opts=None):
        self.dt = dt
        self.M = M                                     # control horizon
        self.N = N                                     # prediction horizon
        self.v_max = v_max
        self.omega_max = omega_max
        self.Q = np.diag([1.0, 1.0, 0.1]) if Q is None else np.asarray(Q)
        self.R = np.diag([0.1, 0.1]) if R is None else np.asarray(R)
        self.alpha = alpha
        self.Q_f = alpha * self.Q

        # Packed parameter layout: [x0(3) | x_ref_0..N (3*(N+1)) | u_ref_0..N-1 (2*N)]
        self.n_params = 3 + 3 * (N + 1) + 2 * N

        self.f = self._build_dynamics()
        nlp, self.lbw, self.ubw, self.lbg, self.ubg, self.w0_default = self._build_nlp()
        self.n_w = nlp['x'].shape[0]
        self.n_g = nlp['g'].shape[0]

        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 100,
            'ipopt.warm_start_init_point': 'yes',
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

        w = []      # decision variables (X and U)
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
        ubw += [ca.inf, ca.inf, ca.inf]     # no bounds set in w - handled by the g constraint below
        g.append(X_k - p_x0)                # X_0 must equal the measured state (fed as a parameter)
        lbg += [0, 0, 0]
        ubg += [0, 0, 0]

        for k in range(N):
            # Control decision U_k -- only add U_k as a decision variable up to M, else reuse U_{M-1}
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

        # Terminal cost
        x_err_N = X_k - p_x_ref(N)
        J += x_err_N.T @ self.Q_f @ x_err_N

        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        nlp = {'f': J, 'x': w, 'p': P, 'g': g}
        return nlp, lbw, ubw, lbg, ubg, w0

    def reset(self):
        """Clear warm-start state. Call after a discontinuity (e.g. re-localization, teleport)."""
        self._w_guess = np.array(self.w0_default, dtype=float)
        self._lam_x0 = None
        self._lam_g0 = None

    def pack_params(self, x0, x_ref, u_ref):
        """x_ref: (N+1, 3) array-like of reference states. u_ref: (N, 2) array-like of reference inputs."""
        x_ref = np.asarray(x_ref, dtype=float).reshape(self.N + 1, 3)
        u_ref = np.asarray(u_ref, dtype=float).reshape(self.N, 2)
        return np.concatenate([np.asarray(x0, dtype=float), x_ref.flatten(), u_ref.flatten()])

    def extract_predicted_states(self, w_opt):
        """Walk the decision vector using the same X_k/U_k layout the build loop above uses."""
        states = [w_opt[0:3]]
        idx = 3
        for k in range(self.N):
            if k < self.M:
                idx += 2  # skip U_k
            states.append(w_opt[idx:idx + 3])
            idx += 3
        return np.array(states)

    def solve(self, x0, x_ref, u_ref, warm_start=True):
        """Solve one MPC step. Returns (u0, predicted_states, solved_ok)."""
        p = self.pack_params(x0, x_ref, u_ref)

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
