#!/usr/bin/env python3
"""Consolidated Roboclaw calibration/demo test nodes for the Aion R6.

Each mode runs as its own ROS node (own node name, own control loop) --
pick one with a required `mode` argument:

    ros2 run debug roboclaw_tests <mode>

Modes:
    forward_test
        Straight-line forward distance test. Drives forward at a fixed duty
        cycle until encoder-derived distance reaches FWD_TARGET_DISTANCE_M,
        then stops. For empirical distance calibration -- compare measured
        distance to the logged computed distance.

    spin_test
        In-place spin angle calibration test. Spins at a fixed duty cycle
        until encoder-derived angle reaches SPIN_TARGET_ANGLE_DEG. Skid-steer
        point turns scrub the wheels against the ground, so this is used to
        tune EFFECTIVE_TRACK_WIDTH_M to compensate.

    wheel_rev_test
        Single-wheel-revolution calibration test. Drives each wheel
        independently for exactly ENCODER_COUNTS_PER_WHEEL_REV ticks. Mark a
        point on each wheel before running -- if it doesn't land back on its
        mark, ENCODER_COUNTS_PER_WHEEL_REV is wrong for that wheel.

    motion_demo
        Diff-drive duty-cycle motion demo/test. Forward, backward, then spin
        90 degrees each direction, using open-loop duty-cycle commands and
        encoder ticks to know when each leg is done. Plots the resulting
        odometry trajectory at the end.

    velocity_motion_demo
        Same forward/backward/spin sequence as motion_demo, but commands
        wheel velocity (Roboclaw's onboard PID via SpeedM1M2) instead of
        open-loop duty cycle. Plots trajectory and measured-vs-commanded
        velocity, to check the onboard PID is tracking the commanded speed.

All modes publish to set_motor_duty_cycle or set_motor_velocity
(LeftRightFloat32) and subscribe to encoder_counts (LeftRightInt32) --
must match roboclaw_for_motors.py.
"""

import argparse
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rclpy
from aion_msgs.msg import LeftRightFloat32, LeftRightInt32
from rclpy.node import Node

# Shared robot geometry / calibration constants -- previously duplicated
# identically across each standalone script.
WHEEL_RADIUS_M = 0.0804              # Empirically measured effective rolling radius.
ENCODER_COUNTS_PER_WHEEL_REV = 1100  # Ticks per wheel revolution (post-gearbox).
TRACK_WIDTH_M = 0.325                # Physical wheel-to-wheel distance (not used directly).
EFFECTIVE_TRACK_WIDTH_M = 0.5132     # Scrub-corrected value for spin kinematics -- see spin_test.
CONTROL_PERIOD_SEC = 0.05            # State machine tick rate, shared by all modes.

METERS_PER_TICK = (2.0 * math.pi * WHEEL_RADIUS_M) / ENCODER_COUNTS_PER_WHEEL_REV


# ---------------------------------------------------------------------------
# forward_test
# ---------------------------------------------------------------------------

FWD_TARGET_DISTANCE_M = 0.40       # Distance to drive forward.
FWD_TRANSLATE_DUTY_PERCENT = 20.0  # Duty cycle while driving.
FWD_MAX_DURATION_SEC = 15.0        # Safety abort if the target isn't reached in time.


class RoboclawForwardTestNode(Node):
    def __init__(self):
        super().__init__("roboclaw_forward_test")

        self.distance = 0.0
        self.finished = False
        self.start_time = self.get_clock().now()
        self.seq_num = 1

        self.duty_publisher = self.create_publisher(LeftRightFloat32, "set_motor_duty_cycle", 1)
        self.create_subscription(LeftRightInt32, "encoder_counts", self.encoder_callback, 10)
        self.control_timer = self.create_timer(CONTROL_PERIOD_SEC, self.control_loop)

        self.get_logger().info(f"Driving forward {FWD_TARGET_DISTANCE_M} m.")

    def encoder_callback(self, msg):
        dist_left = msg.left * METERS_PER_TICK
        dist_right = msg.right * METERS_PER_TICK
        self.distance += (dist_left + dist_right) / 2.0

    def control_loop(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed >= FWD_MAX_DURATION_SEC:
            self.get_logger().error(
                f"Did not reach {FWD_TARGET_DISTANCE_M} m within {FWD_MAX_DURATION_SEC}s -- aborting "
                f"(computed distance so far: {self.distance:.3f} m)"
            )
            self.finished = True
            return

        if self.distance >= FWD_TARGET_DISTANCE_M:
            self.publish_duty(0.0, 0.0)
            self.get_logger().info(f"Done. Computed distance traveled: {self.distance:.3f} m.")
            self.finished = True
            return

        self.publish_duty(FWD_TRANSLATE_DUTY_PERCENT, FWD_TRANSLATE_DUTY_PERCENT)

    def publish_duty(self, left, right):
        msg = LeftRightFloat32()
        msg.left = float(left)
        msg.right = float(right)
        msg.seq_num = self.seq_num
        self.duty_publisher.publish(msg)
        self.seq_num += 1

    def stop_motors(self):
        self.publish_duty(0.0, 0.0)


# ---------------------------------------------------------------------------
# spin_test
# ---------------------------------------------------------------------------

SPIN_TARGET_ANGLE_DEG = 90.0    # Angle to spin (CCW / positive).
SPIN_TEST_DUTY_PERCENT = 20.0  # Duty cycle while spinning.
SPIN_MAX_DURATION_SEC = 15.0   # Safety abort if the target isn't reached in time.

SPIN_TARGET_ANGLE_RAD = math.radians(SPIN_TARGET_ANGLE_DEG)


class RoboclawSpinTestNode(Node):
    def __init__(self):
        super().__init__("roboclaw_spin_test")

        self.angle = 0.0
        self.finished = False
        self.start_time = self.get_clock().now()
        self.seq_num = 1

        self.duty_publisher = self.create_publisher(LeftRightFloat32, "set_motor_duty_cycle", 1)
        self.create_subscription(LeftRightInt32, "encoder_counts", self.encoder_callback, 10)
        self.control_timer = self.create_timer(CONTROL_PERIOD_SEC, self.control_loop)

        self.get_logger().info(f"Spinning {SPIN_TARGET_ANGLE_DEG} deg (CCW).")

    def encoder_callback(self, msg):
        dist_left = msg.left * METERS_PER_TICK
        dist_right = msg.right * METERS_PER_TICK
        self.angle += (dist_right - dist_left) / EFFECTIVE_TRACK_WIDTH_M

    def control_loop(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed >= SPIN_MAX_DURATION_SEC:
            self.get_logger().error(
                f"Did not reach {SPIN_TARGET_ANGLE_DEG} deg within {SPIN_MAX_DURATION_SEC}s -- aborting "
                f"(computed angle so far: {math.degrees(self.angle):.1f} deg)"
            )
            self.finished = True
            return

        if self.angle >= SPIN_TARGET_ANGLE_RAD:
            self.publish_duty(0.0, 0.0)
            self.get_logger().info(f"Done. Computed angle turned: {math.degrees(self.angle):.1f} deg.")
            self.finished = True
            return

        self.publish_duty(-SPIN_TEST_DUTY_PERCENT, SPIN_TEST_DUTY_PERCENT)

    def publish_duty(self, left, right):
        msg = LeftRightFloat32()
        msg.left = float(left)
        msg.right = float(right)
        msg.seq_num = self.seq_num
        self.duty_publisher.publish(msg)
        self.seq_num += 1

    def stop_motors(self):
        self.publish_duty(0.0, 0.0)


# ---------------------------------------------------------------------------
# wheel_rev_test
# ---------------------------------------------------------------------------

WHEEL_REV_TEST_DUTY_PERCENT = 15.0  # Duty cycle while a wheel is still turning.


class RoboclawWheelRevTestNode(Node):
    def __init__(self):
        super().__init__("roboclaw_wheel_rev_test")

        self.left_ticks = 0
        self.right_ticks = 0
        self.left_done = False
        self.right_done = False
        self.finished = False
        self.seq_num = 1

        self.duty_publisher = self.create_publisher(LeftRightFloat32, "set_motor_duty_cycle", 1)
        self.create_subscription(LeftRightInt32, "encoder_counts", self.encoder_callback, 10)
        self.control_timer = self.create_timer(CONTROL_PERIOD_SEC, self.control_loop)

        self.get_logger().info(
            f"Driving each wheel {ENCODER_COUNTS_PER_WHEEL_REV} ticks -- "
            "mark a point on each wheel before it starts."
        )

    def encoder_callback(self, msg):
        self.left_ticks += msg.left
        self.right_ticks += msg.right
        if self.left_ticks >= ENCODER_COUNTS_PER_WHEEL_REV:
            self.left_done = True
        if self.right_ticks >= ENCODER_COUNTS_PER_WHEEL_REV:
            self.right_done = True

    def control_loop(self):
        if self.left_done and self.right_done:
            self.publish_duty(0.0, 0.0)
            self.get_logger().info(
                f"Done. left_ticks={self.left_ticks}, right_ticks={self.right_ticks} "
                f"(target {ENCODER_COUNTS_PER_WHEEL_REV}). Check both wheels landed back on their mark."
            )
            self.finished = True
            return

        duty_left = 0.0 if self.left_done else WHEEL_REV_TEST_DUTY_PERCENT
        duty_right = 0.0 if self.right_done else WHEEL_REV_TEST_DUTY_PERCENT
        self.publish_duty(duty_left, duty_right)

    def publish_duty(self, left, right):
        msg = LeftRightFloat32()
        msg.left = float(left)
        msg.right = float(right)
        msg.seq_num = self.seq_num
        self.duty_publisher.publish(msg)
        self.seq_num += 1

    def stop_motors(self):
        self.publish_duty(0.0, 0.0)


# ---------------------------------------------------------------------------
# motion_demo
# ---------------------------------------------------------------------------

MOTION_DEMO_TRANSLATE_DUTY_PERCENT = 20.0  # Duty cycle for forward/backward legs.
MOTION_DEMO_SPIN_DUTY_PERCENT = 20.0       # Duty cycle for in-place spin legs.

MOTION_DEMO_FORWARD_DISTANCE_M = 1.0   # Distance to travel on the forward leg.
MOTION_DEMO_BACKWARD_DISTANCE_M = 1.0  # Distance to travel on the backward leg.
MOTION_DEMO_SPIN_ANGLE_DEG = 90.0      # Angle to spin on each spin leg.

MOTION_DEMO_PAUSE_SEC = 1.0               # Settle time between legs with motors stopped.
MOTION_DEMO_MAX_PHASE_DURATION_SEC = 15.0  # Safety abort if a leg doesn't finish in time.

MOTION_DEMO_PLOT_OUTPUT_PATH = "roboclaw_motion_test_trajectory.png"  # Trajectory plot output file.

MOTION_DEMO_SPIN_ANGLE_RAD = math.radians(MOTION_DEMO_SPIN_ANGLE_DEG)

# (duty_left, duty_right) for each leg; "pause" legs command zero duty.
MOTION_DEMO_LEG_DUTY = {
    "forward": (MOTION_DEMO_TRANSLATE_DUTY_PERCENT, MOTION_DEMO_TRANSLATE_DUTY_PERCENT),
    "backward": (-MOTION_DEMO_TRANSLATE_DUTY_PERCENT, -MOTION_DEMO_TRANSLATE_DUTY_PERCENT),
    "spin_ccw": (-MOTION_DEMO_SPIN_DUTY_PERCENT, MOTION_DEMO_SPIN_DUTY_PERCENT),
    "spin_cw": (MOTION_DEMO_SPIN_DUTY_PERCENT, -MOTION_DEMO_SPIN_DUTY_PERCENT),
    "pause": (0.0, 0.0),
}
MOTION_DEMO_LEG_SEQUENCE = ["forward", "pause", "backward", "pause", "spin_ccw", "pause", "spin_cw"]


class RoboclawMotionDemoNode(Node):
    def __init__(self):
        super().__init__("roboclaw_motion_demo")

        self.leg_index = 0
        self.leg_name = MOTION_DEMO_LEG_SEQUENCE[0]
        self.leg_distance = 0.0
        self.leg_angle = 0.0
        self.leg_start_time = self.get_clock().now()
        self.finished = False

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.trajectory = [(0.0, 0.0)]
        self.leg_records = []  # Completed legs: name, trajectory slice, achieved distance/angle.
        self.leg_start_index = 0  # Index into self.trajectory where the current leg began.
        self.angle_series = []  # (leg_index, t_since_leg_start, angle_turned_deg) for spin legs.
        self.seq_num = 1

        self.duty_publisher = self.create_publisher(LeftRightFloat32, "set_motor_duty_cycle", 1)
        self.create_subscription(LeftRightInt32, "encoder_counts", self.encoder_callback, 10)
        self.control_timer = self.create_timer(CONTROL_PERIOD_SEC, self.control_loop)

        self.get_logger().info(f"Starting leg: {self.leg_name}")

    def encoder_callback(self, msg):
        dist_left = msg.left * METERS_PER_TICK
        dist_right = msg.right * METERS_PER_TICK
        dist_center = (dist_left + dist_right) / 2.0
        dtheta = (dist_right - dist_left) / EFFECTIVE_TRACK_WIDTH_M

        theta_mid = self.theta + dtheta / 2.0
        self.x += dist_center * math.cos(theta_mid)
        self.y += dist_center * math.sin(theta_mid)
        self.theta = math.atan2(math.sin(self.theta + dtheta), math.cos(self.theta + dtheta))

        self.leg_distance += dist_center
        self.leg_angle += dtheta
        self.trajectory.append((self.x, self.y))

        t_since_leg_start = (self.get_clock().now() - self.leg_start_time).nanoseconds / 1e9
        self.angle_series.append((self.leg_index, t_since_leg_start, math.degrees(self.leg_angle)))

    def leg_complete(self):
        if self.leg_name == "forward":
            return self.leg_distance >= MOTION_DEMO_FORWARD_DISTANCE_M
        if self.leg_name == "backward":
            return self.leg_distance <= -MOTION_DEMO_BACKWARD_DISTANCE_M
        if self.leg_name == "spin_ccw":
            return self.leg_angle >= MOTION_DEMO_SPIN_ANGLE_RAD
        if self.leg_name == "spin_cw":
            return self.leg_angle <= -MOTION_DEMO_SPIN_ANGLE_RAD
        if self.leg_name == "pause":
            elapsed = (self.get_clock().now() - self.leg_start_time).nanoseconds / 1e9
            return elapsed >= MOTION_DEMO_PAUSE_SEC
        return True

    def advance_leg(self):
        self.leg_records.append({
            "name": self.leg_name,
            "leg_index": self.leg_index,
            "start_index": self.leg_start_index,
            "end_index": len(self.trajectory) - 1,
            "distance": self.leg_distance,
            "angle": self.leg_angle,
        })
        self.leg_start_index = len(self.trajectory) - 1
        self.leg_index += 1
        if self.leg_index >= len(MOTION_DEMO_LEG_SEQUENCE):
            self.finished = True
            return
        self.leg_name = MOTION_DEMO_LEG_SEQUENCE[self.leg_index]
        self.leg_distance = 0.0
        self.leg_angle = 0.0
        self.leg_start_time = self.get_clock().now()
        self.get_logger().info(f"Starting leg: {self.leg_name}")

    def control_loop(self):
        elapsed = (self.get_clock().now() - self.leg_start_time).nanoseconds / 1e9
        if elapsed >= MOTION_DEMO_MAX_PHASE_DURATION_SEC:
            self.get_logger().error(
                f"Leg '{self.leg_name}' did not complete within {MOTION_DEMO_MAX_PHASE_DURATION_SEC}s "
                "-- aborting (check encoder_counts is actually publishing, e.g. DRY_RUN)"
            )
            self.finished = True
            return

        if self.leg_complete():
            self.advance_leg()
            if self.finished:
                return

        duty_left, duty_right = MOTION_DEMO_LEG_DUTY[self.leg_name]
        self.publish_duty(duty_left, duty_right)

    def publish_duty(self, left, right):
        msg = LeftRightFloat32()
        msg.left = float(left)
        msg.right = float(right)
        msg.seq_num = self.seq_num
        self.duty_publisher.publish(msg)
        self.seq_num += 1

    def stop_motors(self):
        self.publish_duty(0.0, 0.0)

    def plot_trajectory(self):
        move_legs = [r for r in self.leg_records if r["name"] != "pause"]
        if not move_legs:
            self.get_logger().warn("No completed legs to plot.")
            return

        ncols = 2
        nrows = math.ceil(len(move_legs) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
        axes = axes.flatten()

        for ax, record in zip(axes, move_legs):
            if record["name"] in ("forward", "backward"):
                xs, ys = zip(*self.trajectory[record["start_index"]:record["end_index"] + 1])
                ax.plot(xs, ys, "-", color="tab:blue")
                ax.plot(xs[0], ys[0], "go", label="start")
                ax.plot(xs[-1], ys[-1], "ro", label="end")
                ax.set_title(f"{record['name']}\ndistance = {record['distance']:.3f} m")
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_aspect("equal", adjustable="datalim")
                ax.legend()
            else:
                samples = [(t, a) for idx, t, a in self.angle_series if idx == record["leg_index"]]
                target_deg = MOTION_DEMO_SPIN_ANGLE_DEG if record["name"] == "spin_ccw" else -MOTION_DEMO_SPIN_ANGLE_DEG
                if samples:
                    ts, angles = zip(*samples)
                    ax.plot(ts, angles, "-", color="tab:blue", label="angle turned")
                ax.axhline(target_deg, color="tab:red", linestyle="--", label="target")
                ax.set_title(f"{record['name']}\nangle = {math.degrees(record['angle']):.1f} deg")
                ax.set_xlabel("time (s)")
                ax.set_ylabel("angle turned (deg)")
                ax.legend()
            ax.grid(True)

        for ax in axes[len(move_legs):]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(MOTION_DEMO_PLOT_OUTPUT_PATH)
        self.get_logger().info(f"Saved trajectory plot to {MOTION_DEMO_PLOT_OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# velocity_motion_demo
# ---------------------------------------------------------------------------

VEL_DEMO_TRANSLATE_SPEED_MPS = 0.15  # Target linear speed for forward/backward legs.
VEL_DEMO_SPIN_SPEED_DEG_S = 30.0     # Target angular speed for in-place spin legs.

VEL_DEMO_FORWARD_DISTANCE_M = 1.0   # Distance to travel on the forward leg.
VEL_DEMO_BACKWARD_DISTANCE_M = 1.0  # Distance to travel on the backward leg.
VEL_DEMO_SPIN_ANGLE_DEG = 90.0      # Angle to spin on each spin leg.

VEL_DEMO_PAUSE_SEC = 1.0                # Settle time between legs with motors stopped.
VEL_DEMO_MAX_PHASE_DURATION_SEC = 15.0  # Safety abort if a leg doesn't finish in time.

VEL_DEMO_TRAJECTORY_PLOT_PATH = "roboclaw_velocity_test_trajectory.png"  # Position plot output file.
VEL_DEMO_VELOCITY_PLOT_PATH = "roboclaw_velocity_test_velocity.png"      # Velocity plot output file.

VEL_DEMO_SPIN_ANGLE_RAD = math.radians(VEL_DEMO_SPIN_ANGLE_DEG)
VEL_DEMO_SPIN_SPEED_RAD_S = math.radians(VEL_DEMO_SPIN_SPEED_DEG_S)

VEL_DEMO_TRANSLATE_QPPS = VEL_DEMO_TRANSLATE_SPEED_MPS / METERS_PER_TICK
VEL_DEMO_SPIN_QPPS = (VEL_DEMO_SPIN_SPEED_RAD_S * EFFECTIVE_TRACK_WIDTH_M / 2.0) / METERS_PER_TICK

# (velocity_left, velocity_right) in encoder counts/sec for each leg.
VEL_DEMO_LEG_VELOCITY = {
    "forward": (VEL_DEMO_TRANSLATE_QPPS, VEL_DEMO_TRANSLATE_QPPS),
    "backward": (-VEL_DEMO_TRANSLATE_QPPS, -VEL_DEMO_TRANSLATE_QPPS),
    "spin_ccw": (-VEL_DEMO_SPIN_QPPS, VEL_DEMO_SPIN_QPPS),
    "spin_cw": (VEL_DEMO_SPIN_QPPS, -VEL_DEMO_SPIN_QPPS),
    "pause": (0.0, 0.0),
}
# Commanded linear (m/s) or angular (rad/s) target, used as the reference line on the velocity plot.
VEL_DEMO_LEG_COMMANDED_RATE = {
    "forward": VEL_DEMO_TRANSLATE_SPEED_MPS,
    "backward": -VEL_DEMO_TRANSLATE_SPEED_MPS,
    "spin_ccw": VEL_DEMO_SPIN_SPEED_RAD_S,
    "spin_cw": -VEL_DEMO_SPIN_SPEED_RAD_S,
}
VEL_DEMO_LEG_SEQUENCE = ["forward", "pause", "backward", "pause", "spin_ccw", "pause", "spin_cw"]


class RoboclawVelocityMotionDemoNode(Node):
    def __init__(self):
        super().__init__("roboclaw_velocity_motion_demo")

        self.leg_index = 0
        self.leg_name = VEL_DEMO_LEG_SEQUENCE[0]
        self.leg_distance = 0.0
        self.leg_angle = 0.0
        self.leg_start_time = self.get_clock().now()
        self.finished = False

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.trajectory = [(0.0, 0.0)]
        self.leg_records = []       # Completed legs: name, trajectory slice, achieved distance/angle.
        self.leg_start_index = 0    # Index into self.trajectory where the current leg began.

        self.velocity_samples = []  # (leg_index, t_since_leg_start, measured_rate).
        self.angle_samples = []     # (leg_index, t_since_leg_start, leg_angle_so_far).
        self.last_encoder_time = None

        self.seq_num = 1

        self.velocity_publisher = self.create_publisher(LeftRightFloat32, "set_motor_velocity", 1)
        self.create_subscription(LeftRightInt32, "encoder_counts", self.encoder_callback, 10)
        self.control_timer = self.create_timer(CONTROL_PERIOD_SEC, self.control_loop)

        self.get_logger().info(f"Starting leg: {self.leg_name}")

    def encoder_callback(self, msg):
        now = self.get_clock().now()
        dist_left = msg.left * METERS_PER_TICK
        dist_right = msg.right * METERS_PER_TICK
        dist_center = (dist_left + dist_right) / 2.0
        dtheta = (dist_right - dist_left) / EFFECTIVE_TRACK_WIDTH_M

        theta_mid = self.theta + dtheta / 2.0
        self.x += dist_center * math.cos(theta_mid)
        self.y += dist_center * math.sin(theta_mid)
        self.theta = math.atan2(math.sin(self.theta + dtheta), math.cos(self.theta + dtheta))

        self.leg_distance += dist_center
        self.leg_angle += dtheta
        self.trajectory.append((self.x, self.y))

        t_since_leg_start = (now - self.leg_start_time).nanoseconds / 1e9
        self.angle_samples.append((self.leg_index, t_since_leg_start, self.leg_angle))

        if self.last_encoder_time is not None:
            dt = (now - self.last_encoder_time).nanoseconds / 1e9
            if dt > 0.0:
                measured_rate = dtheta / dt if self.leg_name.startswith("spin") else dist_center / dt
                self.velocity_samples.append((self.leg_index, t_since_leg_start, measured_rate))
        self.last_encoder_time = now

    def leg_complete(self):
        if self.leg_name == "forward":
            return self.leg_distance >= VEL_DEMO_FORWARD_DISTANCE_M
        if self.leg_name == "backward":
            return self.leg_distance <= -VEL_DEMO_BACKWARD_DISTANCE_M
        if self.leg_name == "spin_ccw":
            return self.leg_angle >= VEL_DEMO_SPIN_ANGLE_RAD
        if self.leg_name == "spin_cw":
            return self.leg_angle <= -VEL_DEMO_SPIN_ANGLE_RAD
        if self.leg_name == "pause":
            elapsed = (self.get_clock().now() - self.leg_start_time).nanoseconds / 1e9
            return elapsed >= VEL_DEMO_PAUSE_SEC
        return True

    def advance_leg(self):
        self.leg_records.append({
            "name": self.leg_name,
            "leg_index": self.leg_index,
            "start_index": self.leg_start_index,
            "end_index": len(self.trajectory) - 1,
            "distance": self.leg_distance,
            "angle": self.leg_angle,
        })
        self.leg_start_index = len(self.trajectory) - 1
        self.leg_index += 1
        if self.leg_index >= len(VEL_DEMO_LEG_SEQUENCE):
            self.finished = True
            return
        self.leg_name = VEL_DEMO_LEG_SEQUENCE[self.leg_index]
        self.leg_distance = 0.0
        self.leg_angle = 0.0
        self.leg_start_time = self.get_clock().now()
        self.get_logger().info(f"Starting leg: {self.leg_name}")

    def control_loop(self):
        elapsed = (self.get_clock().now() - self.leg_start_time).nanoseconds / 1e9
        if elapsed >= VEL_DEMO_MAX_PHASE_DURATION_SEC:
            self.get_logger().error(
                f"Leg '{self.leg_name}' did not complete within {VEL_DEMO_MAX_PHASE_DURATION_SEC}s "
                "-- aborting (check encoder_counts is actually publishing, e.g. DRY_RUN)"
            )
            self.finished = True
            return

        if self.leg_complete():
            self.advance_leg()
            if self.finished:
                return

        velocity_left, velocity_right = VEL_DEMO_LEG_VELOCITY[self.leg_name]
        self.publish_velocity(velocity_left, velocity_right)

    def publish_velocity(self, left, right):
        msg = LeftRightFloat32()
        msg.left = float(left)
        msg.right = float(right)
        msg.seq_num = self.seq_num
        self.velocity_publisher.publish(msg)
        self.seq_num += 1

    def stop_motors(self):
        self.publish_velocity(0.0, 0.0)

    def plot_trajectory(self):
        move_legs = [r for r in self.leg_records if r["name"] != "pause"]
        if not move_legs:
            self.get_logger().warn("No completed legs to plot.")
            return

        ncols = 2
        nrows = math.ceil(len(move_legs) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
        axes = axes.flatten()

        for ax, record in zip(axes, move_legs):
            is_spin = record["name"].startswith("spin")

            if is_spin:
                # Spinning on the spot barely moves x/y, so plot heading angle over
                # time instead of a degenerate trajectory.
                samples = [(t, a) for idx, t, a in self.angle_samples if idx == record["leg_index"]]
                if samples:
                    ts, angles = zip(*samples)
                    ax.plot(ts, [math.degrees(a) for a in angles], "-", color="tab:blue")
                ax.set_xlabel("time (s)")
                ax.set_ylabel("angle (deg)")
                detail = f"angle = {math.degrees(record['angle']):.1f} deg"
            else:
                xs, ys = zip(*self.trajectory[record["start_index"]:record["end_index"] + 1])
                ax.plot(xs, ys, "-", color="tab:blue")
                ax.plot(xs[0], ys[0], "go", label="start")
                ax.plot(xs[-1], ys[-1], "ro", label="end")
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_aspect("equal", adjustable="datalim")
                ax.legend()
                detail = f"distance = {record['distance']:.3f} m"

            ax.set_title(f"{record['name']}\n{detail}")
            ax.grid(True)

        for ax in axes[len(move_legs):]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(VEL_DEMO_TRAJECTORY_PLOT_PATH)
        self.get_logger().info(f"Saved trajectory plot to {VEL_DEMO_TRAJECTORY_PLOT_PATH}")

    def plot_velocity(self):
        move_legs = [r for r in self.leg_records if r["name"] != "pause"]
        if not move_legs:
            return

        ncols = 2
        nrows = math.ceil(len(move_legs) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
        axes = axes.flatten()

        for ax, record in zip(axes, move_legs):
            samples = [(t, rate) for idx, t, rate in self.velocity_samples if idx == record["leg_index"]]
            is_spin = record["name"].startswith("spin")
            commanded = VEL_DEMO_LEG_COMMANDED_RATE[record["name"]]
            commanded_plotted = math.degrees(commanded) if is_spin else commanded

            if samples:
                ts, rates = zip(*samples)
                plotted_rates = [math.degrees(r) for r in rates] if is_spin else list(rates)
                ax.plot(ts, plotted_rates, "-", color="tab:blue", label="measured")
            ax.axhline(commanded_plotted, color="tab:red", linestyle="--", label="commanded")
            ax.set_title(record["name"])
            ax.set_xlabel("time (s)")
            ax.set_ylabel("angular rate (deg/s)" if is_spin else "linear speed (m/s)")
            ax.grid(True)
            ax.legend()

        for ax in axes[len(move_legs):]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(VEL_DEMO_VELOCITY_PLOT_PATH)
        self.get_logger().info(f"Saved velocity plot to {VEL_DEMO_VELOCITY_PLOT_PATH}")


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------

NODE_CLASSES = {
    "forward_test": RoboclawForwardTestNode,
    "spin_test": RoboclawSpinTestNode,
    "wheel_rev_test": RoboclawWheelRevTestNode,
    "motion_demo": RoboclawMotionDemoNode,
    "velocity_motion_demo": RoboclawVelocityMotionDemoNode,
}


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(
        prog="roboclaw_tests",
        description="Run one Roboclaw calibration/demo test node (see module docstring for what each mode does).",
    )
    parser.add_argument("mode", choices=sorted(NODE_CLASSES))
    parsed = parser.parse_args(rclpy.utilities.remove_ros_args(args=sys.argv)[1:])

    node = NODE_CLASSES[parsed.mode]()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_motors()
        if hasattr(node, "plot_trajectory"):
            node.plot_trajectory()
        if hasattr(node, "plot_velocity"):
            node.plot_velocity()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
