"""Tuning for the obstacle-avoidance MPC. Limits, ESDF patch and margin are in common/constants.py."""

# --- Horizon ---
# The solver's shooting step is independent of how often the node re-solves
# (common CONTROL_LOOP_DT). N * PREDICTION_DT is about the 2.67 s a VLA chunk spans.
PREDICTION_DT = 0.2          # [s]
CONTROL_HORIZON_M = 14       # free control decision variables
PREDICTION_HORIZON_N = 14

# --- Cost weights ---
Q_DIAG = [1.0, 1.0, 0.1]     # stage cost on [x, y, theta] tracking error
R_DIAG = [0.1, 0.1]          # stage cost on [v, omega]
SLACK_WEIGHT = 1000.0        # slack on the ESDF margin constraint

# Terminal cost: the DARE solution for the unicycle linearised at v = V_MAX, theta = 0,
# with Q_DIAG and R_DIAG. Recompute if Q_DIAG, R_DIAG, V_MAX or PREDICTION_DT change:
#   A = I + PREDICTION_DT * [[0,0,0],[0,0,V_MAX],[0,0,0]],  B = PREDICTION_DT * [[1,0],[0,0],[0,1]]
#   scipy.linalg.solve_discrete_are(A, B, diag(Q_DIAG), diag(R_DIAG))
QF = [
    [2.1583123952, 0.0, 0.0],
    [0.0, 10.0137468867, 1.8745080614],
    [0.0, 1.8745080614, 1.0137804721],
]

# Soft floor on forward speed: stage cost LOW_V_WEIGHT * max(0, LOW_V_THRESHOLD - v)^2.
# Stops the robot settling to a standstill in front of an obstacle dead ahead of its
# reference, while the cost stays bounded (v is in [0, V_MAX]). 0 disables it.
LOW_V_WEIGHT = 100.0
LOW_V_THRESHOLD = 0.15       # [m/s]

# --- IPOPT ---
IPOPT_MAX_ITER = 300         # 100 cut off solves that would have converged; revisit for deployment

# --- Node behaviour ---
WAYPOINT_DT = 1.0 / 3.0             # [s] spacing of waypoints within a VLA chunk
MAX_CONSECUTIVE_SOLVE_FAILURES = 3  # publish zero velocity after this many failed solves in a row
