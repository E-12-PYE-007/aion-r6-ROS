#!/usr/bin/env python3
"""Straight-line forward distance test for the Roboclaw interface.

Drives forward at a fixed duty cycle until the encoder-derived distance
reaches TARGET_DISTANCE_M, then stops. Meant for empirical calibration --
measure the actual distance traveled and compare to TARGET_DISTANCE_M and
the logged computed distance.

Publishes set_motor_duty_cycle (LeftRightFloat32), subscribes to
encoder_counts (LeftRightInt32) -- must match roboclaw_for_motors.py.
"""

import math

import rclpy
from aion_msgs.msg import LeftRightFloat32, LeftRightInt32
from rclpy.node import Node

WHEEL_RADIUS_M = 0.0804              # Must match roboclaw_motion_demo.py.
ENCODER_COUNTS_PER_WHEEL_REV = 1100  # Must match roboclaw_motion_demo.py.

TARGET_DISTANCE_M = 0.40      # Distance to drive forward.
TRANSLATE_DUTY_PERCENT = 20.0 # Duty cycle while driving.

MAX_DURATION_SEC = 15.0    # Safety abort if the target isn't reached in time.
CONTROL_PERIOD_SEC = 0.05  # State machine tick rate.

METERS_PER_TICK = (2.0 * math.pi * WHEEL_RADIUS_M) / ENCODER_COUNTS_PER_WHEEL_REV


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

        self.get_logger().info(f"Driving forward {TARGET_DISTANCE_M} m.")

    def encoder_callback(self, msg):
        dist_left = msg.left * METERS_PER_TICK
        dist_right = msg.right * METERS_PER_TICK
        self.distance += (dist_left + dist_right) / 2.0

    def control_loop(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed >= MAX_DURATION_SEC:
            self.get_logger().error(
                f"Did not reach {TARGET_DISTANCE_M} m within {MAX_DURATION_SEC}s -- aborting "
                f"(computed distance so far: {self.distance:.3f} m)"
            )
            self.finished = True
            return

        if self.distance >= TARGET_DISTANCE_M:
            self.publish_duty(0.0, 0.0)
            self.get_logger().info(f"Done. Computed distance traveled: {self.distance:.3f} m.")
            self.finished = True
            return

        self.publish_duty(TRANSLATE_DUTY_PERCENT, TRANSLATE_DUTY_PERCENT)

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
    node = RoboclawForwardTestNode()
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
