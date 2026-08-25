#!/usr/bin/env python3
"""Diff-drive velocity-command motion demo/test for the Roboclaw interface.

Same forward/backward/spin-90-each-direction test as roboclaw_motion_demo.py,
but commands wheel velocity (Roboclaw's onboard PID via SpeedM1M2) instead of
open-loop duty cycle. Encoder ticks are used both for position (to know when
each leg is done) and for measured velocity (to check the PID is actually
tracking the commanded speed).

Publishes set_motor_velocity (LeftRightFloat32, encoder counts/sec),
subscribes to encoder_counts (LeftRightInt32) -- must match
roboclaw_for_motors.py.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rclpy
from aion_msgs.msg import LeftRightFloat32, LeftRightInt32
from rclpy.node import Node

TRACK_WIDTH_M = 0.325                # Must match roboclaw_motion_demo.py (not used directly).
EFFECTIVE_TRACK_WIDTH_M = 0.5132      # Scrub-corrected value for spin kinematics -- see roboclaw_spin_test.py.
WHEEL_RADIUS_M = 0.0804              # Must match roboclaw_motion_demo.py.
ENCODER_COUNTS_PER_WHEEL_REV = 1100  # Must match roboclaw_motion_demo.py.

TRANSLATE_SPEED_MPS = 0.15   # Target linear speed for forward/backward legs.
SPIN_SPEED_DEG_S = 30.0      # Target angular speed for in-place spin legs.

FORWARD_DISTANCE_M = 1.0     # Distance to travel on the forward leg.
BACKWARD_DISTANCE_M = 1.0    # Distance to travel on the backward leg.
SPIN_ANGLE_DEG = 90.0        # Angle to spin on each spin leg.

PAUSE_SEC = 1.0                # Settle time between legs with motors stopped.
MAX_PHASE_DURATION_SEC = 15.0  # Safety abort if a leg doesn't finish in time.
CONTROL_PERIOD_SEC = 0.05      # State machine tick rate.

TRAJECTORY_PLOT_PATH = "roboclaw_velocity_test_trajectory.png"  # Position plot output file.
VELOCITY_PLOT_PATH = "roboclaw_velocity_test_velocity.png"      # Velocity plot output file.

METERS_PER_TICK = (2.0 * math.pi * WHEEL_RADIUS_M) / ENCODER_COUNTS_PER_WHEEL_REV
SPIN_ANGLE_RAD = math.radians(SPIN_ANGLE_DEG)
SPIN_SPEED_RAD_S = math.radians(SPIN_SPEED_DEG_S)

TRANSLATE_QPPS = TRANSLATE_SPEED_MPS / METERS_PER_TICK
SPIN_QPPS = (SPIN_SPEED_RAD_S * EFFECTIVE_TRACK_WIDTH_M / 2.0) / METERS_PER_TICK

# (velocity_left, velocity_right) in encoder counts/sec for each leg.
LEG_VELOCITY = {
    "forward": (TRANSLATE_QPPS, TRANSLATE_QPPS),
    "backward": (-TRANSLATE_QPPS, -TRANSLATE_QPPS),
    "spin_ccw": (-SPIN_QPPS, SPIN_QPPS),
    "spin_cw": (SPIN_QPPS, -SPIN_QPPS),
    "pause": (0.0, 0.0),
}
# Commanded linear (m/s) or angular (rad/s) target, used as the reference line on the velocity plot.
LEG_COMMANDED_RATE = {
    "forward": TRANSLATE_SPEED_MPS,
    "backward": -TRANSLATE_SPEED_MPS,
    "spin_ccw": SPIN_SPEED_RAD_S,
    "spin_cw": -SPIN_SPEED_RAD_S,
}
LEG_SEQUENCE = ["forward", "pause", "backward", "pause", "spin_ccw", "pause", "spin_cw"]


class RoboclawVelocityMotionDemoNode(Node):
    def __init__(self):
        super().__init__("roboclaw_velocity_motion_demo")

        self.leg_index = 0
        self.leg_name = LEG_SEQUENCE[0]
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
            return self.leg_distance >= FORWARD_DISTANCE_M
        if self.leg_name == "backward":
            return self.leg_distance <= -BACKWARD_DISTANCE_M
        if self.leg_name == "spin_ccw":
            return self.leg_angle >= SPIN_ANGLE_RAD
        if self.leg_name == "spin_cw":
            return self.leg_angle <= -SPIN_ANGLE_RAD
        if self.leg_name == "pause":
            elapsed = (self.get_clock().now() - self.leg_start_time).nanoseconds / 1e9
            return elapsed >= PAUSE_SEC
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
        if self.leg_index >= len(LEG_SEQUENCE):
            self.finished = True
            return
        self.leg_name = LEG_SEQUENCE[self.leg_index]
        self.leg_distance = 0.0
        self.leg_angle = 0.0
        self.leg_start_time = self.get_clock().now()
        self.get_logger().info(f"Starting leg: {self.leg_name}")

    def control_loop(self):
        elapsed = (self.get_clock().now() - self.leg_start_time).nanoseconds / 1e9
        if elapsed >= MAX_PHASE_DURATION_SEC:
            self.get_logger().error(
                f"Leg '{self.leg_name}' did not complete within {MAX_PHASE_DURATION_SEC}s "
                "-- aborting (check encoder_counts is actually publishing, e.g. DRY_RUN)"
            )
            self.finished = True
            return

        if self.leg_complete():
            self.advance_leg()
            if self.finished:
                return

        velocity_left, velocity_right = LEG_VELOCITY[self.leg_name]
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
        fig.savefig(TRAJECTORY_PLOT_PATH)
        self.get_logger().info(f"Saved trajectory plot to {TRAJECTORY_PLOT_PATH}")

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
            commanded = LEG_COMMANDED_RATE[record["name"]]
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
        fig.savefig(VELOCITY_PLOT_PATH)
        self.get_logger().info(f"Saved velocity plot to {VELOCITY_PLOT_PATH}")


def main(args=None):
    rclpy.init(args=args)
    node = RoboclawVelocityMotionDemoNode()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_motors()
        node.plot_trajectory()
        node.plot_velocity()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
