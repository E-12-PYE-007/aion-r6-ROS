"""Tuning for the CBF-QP safety filter. Limits, ESDF patch and margin are in common/constants.py."""

# The barrier h = ESDF distance - margin is evaluated at a point this far ahead of the robot
# along its heading. At the robot's centre omega has no instantaneous effect on h, so the QP
# could only brake; ahead of it, turning sweeps the point sideways and the QP can steer.
LOOKAHEAD_M = 0.4

# Added to SAFETY_MARGIN_M for the barrier. The robot's centre passes closer to an obstacle
# than the lookahead point does when it turns past it, so without this the centre dips
# inside SAFETY_MARGIN_M.
CBF_EXTRA_MARGIN_M = 0.15

# Class-K gain in dh/dt >= -CBF_ALPHA * h. Larger lets the robot approach the margin faster
# and demands a harder push out once inside; smaller brakes earlier and gentler.
CBF_ALPHA = 2.0

# QP weights on the deviation of [v, omega] from the nominal command. Omega is nearly free:
# the nominal is obstacle-blind and keeps pointing at an obstacle, and with a heavy omega
# weight the filter only brakes and the robot stops in front of it instead of steering round.
CBF_WEIGHT_V = 1.0
CBF_WEIGHT_OMEGA = 0.003

# Slack on the CBF constraint, so the QP stays feasible if the robot is already inside the
# margin.
CBF_SLACK_WEIGHT = 1000.0

CBF_H_TOPIC = 'cbf/h'     # debug: barrier value at the lookahead point, one per tick
