"""One-stop tuning file for the Aion R6 obstacle-avoidance MPC. Solver behaviour,
ESDF patch geometry, and node wiring are defined here rather than scattered across
solver_setup.py / mpc_path_follower.py."""

# --- ESDF patch geometry ---
# PATCH_SIZE/PATCH_RESOLUTION fix the solver's NLP structure, so they must match
# what EsdfMap actually produces.
PATCH_SIZE = 44
PATCH_RESOLUTION = 0.1
SPLINE_DEGREE = 3
NVBLOX_MAX_DISTANCE_M = 2.0

# --- Safety ---
# Real Aion R6 half-width + clearance, in metres. Must be set before deployment -
# UnicycleMPC refuses to build without an explicit value.
SAFETY_MARGIN_M = 0.20
SLACK_WEIGHT = 1000.0

# --- MPC horizon and dynamics limits ---
# PREDICTION_DT is the solver's own shooting-step spacing - deliberately decoupled
# from how often the node actually replans (CONTROL_LOOP_DT, below). Coarser here
# lets the horizon cover the full ~2.67s VLA chunk (N_WAYPOINTS * WAYPOINT_DT)
# without the NLP growing to match a much finer step.
PREDICTION_DT = 0.2          # [s] solver's internal shooting-step spacing
CONTROL_HORIZON_M = 14       # number of free control decision variables
PREDICTION_HORIZON_N = 14    # steps * PREDICTION_DT ~ 2.8s, covers the full VLA chunk
V_MAX = 0.6                  # [m/s]
OMEGA_MAX = 1.0              # [rad/s]

# How often the node actually re-solves and refreshes cmd_vel - independent of
# PREDICTION_DT. Must stay <= 0.125s to hold the 8Hz floor (so VLA updates aren't missed).
CONTROL_LOOP_DT = 1 / 15     # [s] ~15Hz

# --- Cost weights ---
Q_DIAG = [1.0, 1.0, 0.1]     # stage cost weights on [x, y, theta] tracking error
R_DIAG = [0.1, 0.1]          # stage cost weights on [v, omega] control effort

# Terminal cost matrix: solution to the discrete algebraic Riccati equation for the
# unicycle model linearised at [v=V_MAX, theta=0], step size PREDICTION_DT, with
# Q_DIAG/R_DIAG above - the infinite-horizon LQR cost-to-go, used as a stand-in for
# the cost the finite horizon can't see past X_N. Recompute if Q_DIAG, R_DIAG,
# V_MAX, or PREDICTION_DT change:
#   A = I + PREDICTION_DT * [[0,0,0],[0,0,V_MAX],[0,0,0]]
#   B = PREDICTION_DT * [[1,0],[0,0],[0,1]]
#   scipy.linalg.solve_discrete_are(A, B, diag(Q_DIAG), diag(R_DIAG))
QF = [
    [2.1583123952, 0.0, 0.0],
    [0.0, 10.0137468867, 1.8745080614],
    [0.0, 1.8745080614, 1.0137804721],
]
QF_MULTIPLIER = 1.0
TERMINAL_COST_Q = [[element * QF_MULTIPLIER for element in row] for row in QF]

# --- IPOPT ---
# 100 was too tight for the 2.8s/N=14 horizon: a dumped real failure needed 112
# iterations to actually converge (not infeasible - just cut off), and once one
# solve times out, reset() clears the warm start, so the next solve also starts
# cold and also risks needing >100 - a self-sustaining freeze. Needs a proper look
# before real deployment (112 iterations already costs close to one control-loop
# period on this dev machine); raised for now to unblock testing.
IPOPT_MAX_ITER = 300

# --- Node topics and frames ---
ACTION_CHUNK_TOPIC = '/vla/action_chunk'
ODOM_TOPIC = '/odom'   # published by the EKF - pose of base_link in the odom frame
ODOM_FRAME = 'odom'
BASE_FRAME = 'base_link'
CMD_VEL_TOPIC = 'cmd_vel'
ESDF_TOPIC = '/nvblox_node/static_map_slice'
MAX_SLACK_TOPIC = 'mpc/max_slack'  # debug: max S_k over the horizon, one per solve

# --- Node behaviour ---
WAYPOINT_DT = 1.0 / 3.0             # spacing between waypoints within a VLA action chunk [s]
MAX_CONSECUTIVE_SOLVE_FAILURES = 3  # stop publishing solver output after this many failed ticks in a row
