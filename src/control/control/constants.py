"""Rover limits and topics shared by the control nodes."""

V_MAX = 0.3                         # [m/s] rover speed limit
OMEGA_MAX = 0.3                     # [rad/s] rover yaw rate limit

ACTION_CHUNK_TOPIC = '/vla/action_chunk'
ODOM_TOPIC = '/odometry/filtered'   # local EKF output: pose of base_link in the odom frame
CMD_VEL_TOPIC = 'cmd_vel'
