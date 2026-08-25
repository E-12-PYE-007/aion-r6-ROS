#!/usr/bin/env python3
"""Diff-drive motion demo/test for the Roboclaw interface.

Drives the robot forward, backward, then spins 90 degrees each direction,
using encoder ticks (via roboclaw_for_motors.py) to know when each leg is
done. Plots the resulting odometry trajectory at the end.

Publishes set_motor_duty_cycle (LeftRightFloat32), subscribes to
encoder_counts (LeftRightInt32) -- must match roboclaw_for_motors.py.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rclpy
from aion_msgs.msg import LeftRightFloat32, LeftRightInt32
from rclpy.node import Node

TRACK_WIDTH_M = 0.325                    # Distance between left/right wheel contact patches (not used directly).
EFFECTIVE_TRACK_WIDTH_M = 0.5132          # Scrub-corrected value for spin kinematics -- see roboclaw_spin_test.py.
WHEEL_RADIUS_M = 0.0804                 # Empirically measured effective rolling radius (was diameter/2 mixup).
ENCODER_COUNTS_PER_WHEEL_REV = 1100     # Ticks per wheel revolution (post-gearbox).

TRANSLATE_DUTY_PERCENT = 20.0           # Duty cycle for forward/backward legs.
SPIN_DUTY_PERCENT = 20.0                # Duty cycle for in-place spin legs.

FORWARD_DISTANCE_M = 1.0                # Distance to travel on the forward leg.
BACKWARD_DISTANCE_M = 1.0               # Distance to travel on the backward leg.
SPIN_ANGLE_DEG = 90.0                   # Angle to spin on each spin leg.

PAUSE_SEC = 1.0                         # Settle time between legs with motors stopped.
MAX_PHASE_DURATION_SEC = 15.0           # Safety abort if a leg doesn't finish in time.
CONTROL_PERIOD_SEC = 0.05               # State machine tick rate.

PLOT_OUTPUT_PATH = "roboclaw_motion_test_trajectory.png"  # Trajectory plot output file.

METERS_PER_TICK = (2.0 * math.pi * WHEEL_RADIUS_M) / ENCODER_COUNTS_PER_WHEEL_REV
SPIN_ANGLE_RAD = math.radians(SPIN_ANGLE_DEG)

# (duty_left, duty_right) for each leg; "pause" legs command zero duty.
LEG_DUTY = {
    "forward": (TRANSLATE_DUTY_PERCENT, TRANSLATE_DUTY_PERCENT),
    "backward": (-TRANSLATE_DUTY_PERCENT, -TRANSLATE_DUTY_PERCENT),
    "spin_ccw": (-SPIN_DUTY_PERCENT, SPIN_DUTY_PERCENT),
    "spin_cw": (SPIN_DUTY_PERCENT, -SPIN_DUTY_PERCENT),
    "pause": (0.0, 0.0),
}
LEG_SEQUENCE = ["forward", "pause", "backward", "pause", "spin_ccw", "pause", "spin_cw"]


class RoboclawMotionDemoNode(Node):
    def __init__(self):
        super().__init__("roboclaw_motion_demo")

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

        duty_left, duty_right = LEG_DUTY[self.leg_name]
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
                target_deg = SPIN_ANGLE_DEG if record["name"] == "spin_ccw" else -SPIN_ANGLE_DEG
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
        fig.savefig(PLOT_OUTPUT_PATH)
        self.get_logger().info(f"Saved trajectory plot to {PLOT_OUTPUT_PATH}")


def main(args=None):
    rclpy.init(args=args)
    node = RoboclawMotionDemoNode()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_motors()
        node.plot_trajectory()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
