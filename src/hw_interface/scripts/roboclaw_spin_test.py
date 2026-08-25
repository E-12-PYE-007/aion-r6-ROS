#!/usr/bin/env python3
"""In-place spin angle calibration test for the Roboclaw interface.

Spins the robot at a fixed duty cycle until the encoder-derived angle
reaches TARGET_ANGLE_DEG, then stops. Meant for empirical calibration --
measure the actual angle turned (protractor / marked reference lines) and
compare to TARGET_ANGLE_DEG and the logged computed angle. Skid-steer point
turns scrub the wheels against the ground, so the physically measured
TRACK_WIDTH_M is expected to under-predict real rotation; this test is for
tuning a separate effective track width to compensate.

Publishes set_motor_duty_cycle (LeftRightFloat32), subscribes to
encoder_counts (LeftRightInt32) -- must match roboclaw_for_motors.py.
"""

import math

import rclpy
from aion_msgs.msg import LeftRightFloat32, LeftRightInt32
from rclpy.node import Node

WHEEL_RADIUS_M = 0.0804              # Must match roboclaw_motion_demo.py.
ENCODER_COUNTS_PER_WHEEL_REV = 1100  # Must match roboclaw_motion_demo.py.
TRACK_WIDTH_M = 0.325                # Physical wheel-to-wheel distance (not used directly -- see below).
EFFECTIVE_TRACK_WIDTH_M = 0.5132     # Scrub-corrected value from this test: computed 90 deg, measured 57 deg.

TARGET_ANGLE_DEG = 90.0    # Angle to spin (CCW / positive).
SPIN_DUTY_PERCENT = 20.0   # Duty cycle while spinning.

MAX_DURATION_SEC = 15.0    # Safety abort if the target isn't reached in time.
CONTROL_PERIOD_SEC = 0.05  # State machine tick rate.

METERS_PER_TICK = (2.0 * math.pi * WHEEL_RADIUS_M) / ENCODER_COUNTS_PER_WHEEL_REV
TARGET_ANGLE_RAD = math.radians(TARGET_ANGLE_DEG)


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

        self.get_logger().info(f"Spinning {TARGET_ANGLE_DEG} deg (CCW).")

    def encoder_callback(self, msg):
        dist_left = msg.left * METERS_PER_TICK
        dist_right = msg.right * METERS_PER_TICK
        self.angle += (dist_right - dist_left) / EFFECTIVE_TRACK_WIDTH_M

    def control_loop(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed >= MAX_DURATION_SEC:
            self.get_logger().error(
                f"Did not reach {TARGET_ANGLE_DEG} deg within {MAX_DURATION_SEC}s -- aborting "
                f"(computed angle so far: {math.degrees(self.angle):.1f} deg)"
            )
            self.finished = True
            return

        if self.angle >= TARGET_ANGLE_RAD:
            self.publish_duty(0.0, 0.0)
            self.get_logger().info(f"Done. Computed angle turned: {math.degrees(self.angle):.1f} deg.")
            self.finished = True
            return

        self.publish_duty(-SPIN_DUTY_PERCENT, SPIN_DUTY_PERCENT)

    def publish_duty(self, left, right):
        msg = LeftRightFloat32()
        msg.left = float(left)
        msg.right = float(right)
        msg.seq_num = self.seq_num
        self.duty_publisher.publish(msg)
        self.seq_num += 1

    def stop_motors(self):
        self.publish_duty(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = RoboclawSpinTestNode()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_motors()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
