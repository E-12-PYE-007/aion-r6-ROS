"""Constants for the CBF safety layer. Rover limits and shared topics are in control/constants.py."""

# --- ESDF patch ---
PATCH_SIZE = 22               # cells per side, must be even; fixes the spline structure
PATCH_RESOLUTION = 0.1        # [m/cell], the deployment voxel size
SPLINE_DEGREE = 3
NVBLOX_MAX_DISTANCE_M = 2.0   # nvblox's ESDF clamp; unknown cells are read as this

# --- Safety ---
SAFETY_MARGIN_M = 0.20        # [m] robot half-width plus clearance; check against the real robot

# --- Control loop ---
CONTROL_LOOP_DT = 1 / 15      # [s] must stay <= 0.125 s so no 8 Hz VLA chunk is missed

# --- CBF-QP tuning ---
LOOKAHEAD_M = 0.4             # [m] barrier is evaluated this far ahead, so the QP can steer as well as brake
CBF_EXTRA_MARGIN_M = 0.15     # [m] added to SAFETY_MARGIN_M; the centre cuts closer than the lookahead point in turns
CBF_ALPHA = 2.0               # class-K gain in dh/dt >= -alpha * h; larger = later, harder braking
CBF_WEIGHT_V = 1.0            # QP weight on deviation from nominal v
CBF_WEIGHT_OMEGA = 0.003      # QP weight on deviation from nominal omega; small so the filter steers round obstacles
CBF_SLACK_WEIGHT = 1000.0     # QP weight on constraint slack; keeps the QP feasible inside the margin

# --- Topics and frames ---
ODOM_FRAME = 'odom'
ESDF_TOPIC = '/nvblox_node/static_map_slice'
SLACK_TOPIC = 'safety_layer/slack'   # debug: constraint slack in use, one per tick
CBF_H_TOPIC = 'cbf/h'                # debug: barrier value at the lookahead point, one per tick
