#!/usr/bin/env python3
"""Single-wheel-revolution calibration test for the Roboclaw interface.

Drives each wheel independently for exactly ENCODER_COUNTS_PER_WHEEL_REV
ticks, then stops. Mark a point on each wheel before running it -- if a
wheel doesn't land back on its mark, ENCODER_COUNTS_PER_WHEEL_REV is wrong
for that wheel (the real counts-per-revolution differs from the configured
value).

Publishes set_motor_duty_cycle (LeftRightFloat32), subscribes to
encoder_counts (LeftRightInt32) -- must match roboclaw_for_motors.py.
"""

import rclpy
from aion_msgs.msg import LeftRightFloat32, LeftRightInt32
from rclpy.node import Node

ENCODER_COUNTS_PER_WHEEL_REV = 1100  # Must match roboclaw_motion_demo.py.
TEST_DUTY_PERCENT = 15.0             # Duty cycle while a wheel is still turning.
CONTROL_PERIOD_SEC = 0.05            # State machine tick rate.


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

        duty_left = 0.0 if self.left_done else TEST_DUTY_PERCENT
        duty_right = 0.0 if self.right_done else TEST_DUTY_PERCENT
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


def main(args=None):
    rclpy.init(args=args)
    node = RoboclawWheelRevTestNode()
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
