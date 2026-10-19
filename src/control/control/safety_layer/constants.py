"""Constants for the CBF safety layer. Rover limits and shared topics are in control/constants.py."""

# --- ESDF patch ---
PATCH_SIZE = 22               # cells per side, must be even; fixes the spline structure
PATCH_RESOLUTION = 0.1        # [m/cell], the deployment voxel size
SPLINE_DEGREE = 3
NVBLOX_MAX_DISTANCE_M = 2.0   # nvblox's ESDF clamp; unknown cells are read as this

# --- Robot body (base_link frame, x forward) ---
BODY_LENGTH_M = 0.49          # [m] measured
BODY_WIDTH_M = 0.35           # [m] measured

# --- Control loop ---
CONTROL_LOOP_DT = 1 / 15      # [s] must stay <= 0.125 s so no 8 Hz VLA chunk is missed

# --- CBF-QP tuning ---
# A single disc at the nose (front-centre, ahead of the pivot). Being offset ahead keeps omega in the
# barrier, so the CBF can steer around obstacles and not only brake; a point at the pivot would make h
# independent of heading and drop omega entirely. The clearance is sized so the disc spans the full
# front width. Rear/side coverage is intentionally dropped -- the barrier guards forward driving.
CBF_BODY_POINTS_M = [(BODY_LENGTH_M / 2, 0.0)]
# [m] ESDF distance the nose point must keep = half-width (so the disc reaches the front corners) + an
# ~0.08 m true safety margin, so the front corners still clear obstacles by ~0.08 m as they did before.
CBF_BODY_CLEARANCE_M = BODY_WIDTH_M / 2 + 0.08
CBF_ALPHA = 0.5               # class-K gain in dh/dt >= -alpha * h; larger = later, harder braking
CBF_WEIGHT_V = 3.0            # QP weight on deviation from nominal v
CBF_WEIGHT_OMEGA = 1.0        # QP weight on deviation from nominal omega
CBF_SLACK_WEIGHT = 500.0      # QP weight on constraint slack; keeps the QP feasible inside the clearance

# --- Reverse recovery ---
SLACK_REVERSE_THRESHOLD = 0.01   # slack above this means no command satisfies the barriers
SLACK_REVERSE_TICKS = 8          # consecutive ticks above the threshold (about 0.5 s) that trigger the reverse
REVERSE_DISTANCE_M = 0.10        # [m] backed up straight after the in-place turn, measured on odometry
REVERSE_SPEED_M_S = 0.1          # [m/s]
REVERSE_TIMEOUT_FACTOR = 1.5     # give up after this multiple of the nominal reverse time
STALL_REVERSE_TICKS = 8          # consecutive stalled ticks (about 0.5 s) that trigger the reverse
STALL_SPEED_EPS = 0.05           # [m/s, rad/s] filtered |v| and |omega| below this count as stationary
STALL_NOMINAL_V_EPS = 0.02       # [m/s] nominal forward speed above this means the robot is being asked to move
REVERSE_TURN_OMEGA = 0.3         # [rad/s] in-place turn rate toward the open side, before backing up
REVERSE_MAX_YAW = 0.5            # [rad] how far to turn in place (~30 deg) before backing up straight
REVERSE_GRAD_MIN = 1e-3          # ESDF gradient magnitude below this is treated as no clear direction -> straight back

# --- Topics and frames ---
ODOM_FRAME = 'odom'
ESDF_TOPIC = '/nvblox_node/static_map_slice'
SLACK_TOPIC = 'safety_layer/slack'   # debug: constraint slack in use, one per tick
CBF_H_TOPIC = 'cbf/h'                # debug: smallest barrier value over the body points, one per tick
CMD_VEL_NOMINAL_TOPIC = 'safety_layer/cmd_vel_nominal'  # debug: pure-pursuit command before the CBF filter
FILTERING_ACTIVE_TOPIC = 'safety_layer/filtering_active'  # debug: True when the CBF filter changed v or omega
REVERSING_TOPIC = 'safety_layer/reversing'  # debug: True while the reverse recovery is driving

# --- Filtering-active detection ---
FILTERING_EPS = 1e-3          # per-channel deadband; |v_safe - v_nom| or |omega_safe - omega_nom|
                              # above this (in m/s / rad/s) counts as the filter intervening,
                              # set just above the QP solver's numerical noise floor
