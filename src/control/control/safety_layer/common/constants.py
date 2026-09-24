"""Constants shared by the CBF and MPC safety layers."""

# --- ESDF patch ---
# Fixes the spline structure, so it must match what EsdfMap produces.
PATCH_SIZE = 22               # cells per side, must be even
PATCH_RESOLUTION = 0.1        # [m/cell], the deployment voxel size
SPLINE_DEGREE = 3
NVBLOX_MAX_DISTANCE_M = 2.0   # nvblox's ESDF clamp; unknown cells are read as this

# --- Safety ---
SAFETY_MARGIN_M = 0.20        # [m] robot half-width plus clearance; check against the real robot

# --- Limits ---
V_MAX = 0.3                   # [m/s]
OMEGA_MAX = 0.3               # [rad/s]

# --- Control loop ---
CONTROL_LOOP_DT = 1 / 15      # [s] must stay <= 0.125 s so no 8 Hz VLA chunk is missed

# --- Pure-pursuit nominal command (obstacle-blind) ---
PP_LOOKAHEAD_DISTANCE_M = 0.3
PP_HERMITE_SAMPLES_PER_SEGMENT = 10
PP_K_CURV = 1.0               # slows down in curves: v = V_MAX / (1 + PP_K_CURV * |curvature|)
N_WAYPOINTS = 8               # fixed by aion_msgs/ActionChunk.msg

# --- Topics and frames ---
ACTION_CHUNK_TOPIC = '/vla/action_chunk'
ODOM_TOPIC = '/odom'          # pose of base_link in the odom frame
ODOM_FRAME = 'odom'
BASE_FRAME = 'base_link'
CMD_VEL_TOPIC = 'cmd_vel'
ESDF_TOPIC = '/nvblox_node/static_map_slice'
SLACK_TOPIC = 'safety_layer/slack'   # debug: constraint slack in use, one per tick
