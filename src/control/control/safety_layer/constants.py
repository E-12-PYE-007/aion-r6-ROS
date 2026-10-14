"""Constants for the CBF safety layer. Rover limits and shared topics are in control/constants.py."""

# --- ESDF patch ---
PATCH_SIZE = 22               # cells per side, must be even; fixes the spline structure
PATCH_RESOLUTION = 0.1        # [m/cell], the deployment voxel size
SPLINE_DEGREE = 3
NVBLOX_MAX_DISTANCE_M = 2.0   # nvblox's ESDF clamp; unknown cells are read as this

# --- Robot body (base_link frame, x forward) ---
BODY_LENGTH_M = 0.49          # [m] measured
BODY_WIDTH_M = 0.45           # [m] assumed; check against the real robot

# --- Control loop ---
CONTROL_LOOP_DT = 1 / 15      # [s] must stay <= 0.125 s so no 8 Hz VLA chunk is missed

# --- CBF-QP tuning ---
# One barrier per body point, so the front corners and nose are each kept CBF_BODY_CLEARANCE_M from obstacles.
CBF_BODY_POINTS_M = [(BODY_LENGTH_M / 2, BODY_WIDTH_M / 2), (BODY_LENGTH_M / 2, -BODY_WIDTH_M / 2), (BODY_LENGTH_M / 2, 0.0)]
CBF_BODY_CLEARANCE_M = 0.08   # [m] ESDF distance each body point must keep
CBF_ALPHA = 0.7               # class-K gain in dh/dt >= -alpha * h; larger = later, harder braking
CBF_WEIGHT_V = 1.0            # QP weight on deviation from nominal v
CBF_WEIGHT_OMEGA = 2.0        # QP weight on deviation from nominal omega
CBF_SLACK_WEIGHT = 500.0      # QP weight on constraint slack; keeps the QP feasible inside the clearance

# --- Topics and frames ---
ODOM_FRAME = 'odom'
ESDF_TOPIC = '/nvblox_node/static_map_slice'
SLACK_TOPIC = 'safety_layer/slack'   # debug: constraint slack in use, one per tick
CBF_H_TOPIC = 'cbf/h'                # debug: smallest barrier value over the body points, one per tick
CMD_VEL_NOMINAL_TOPIC = 'safety_layer/cmd_vel_nominal'  # debug: pure-pursuit command before the CBF filter
FILTERING_ACTIVE_TOPIC = 'safety_layer/filtering_active'  # debug: True when the CBF filter changed v or omega

# --- Filtering-active detection ---
FILTERING_EPS = 1e-3          # per-channel deadband; |v_safe - v_nom| or |omega_safe - omega_nom|
                              # above this (in m/s / rad/s) counts as the filter intervening,
                              # set just above the QP solver's numerical noise floor
