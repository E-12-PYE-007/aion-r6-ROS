#!/usr/bin/env python3
"""cmd_vel-to-Roboclaw adapter for the Aion R6.

Subscribes to cmd_vel (geometry_msgs/Twist) and converts body velocity into
per-wheel Roboclaw velocity commands, published as custom_msgs/LeftRightFloat32
on set_motor_velocity. If no cmd_vel arrives for CMD_VEL_TIMEOUT_S, it commands zero
wheel velocity until cmd_vel resumes.
"""

import math

import rclpy
from custom_msgs.msg import LeftRightFloat32
from geometry_msgs.msg import Twist
from rclpy.node import Node

WHEEL_RADIUS_M = 0.0804              # Must match roboclaw_tests.py.
ENCODER_COUNTS_PER_WHEEL_REV = 1100  # Must match roboclaw_tests.py.
EFFECTIVE_TRACK_WIDTH_M = 0.75     # Must match roboclaw_tests.py.

CMD_VEL_TIMEOUT_S = 0.5              # Controllers publish at 15-30 Hz; this is several missed ticks.
WATCHDOG_PERIOD_S = 0.1

METERS_PER_TICK = (2.0 * math.pi * WHEEL_RADIUS_M) / ENCODER_COUNTS_PER_WHEEL_REV


class CmdVelToRoboclawNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_roboclaw')

        self.seq_num = 1
        self.last_cmd_time = None
        self.timed_out = False

        self.velocity_publisher = self.create_publisher(LeftRightFloat32, 'set_motor_velocity', 1)
        self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.create_timer(WATCHDOG_PERIOD_S, self.watchdog)

    def cmd_vel_callback(self, msg):
        linear_vel = msg.linear.x
        angular_vel = msg.angular.z

        left_mps = linear_vel - angular_vel * EFFECTIVE_TRACK_WIDTH_M / 2.0
        right_mps = linear_vel + angular_vel * EFFECTIVE_TRACK_WIDTH_M / 2.0

        self.last_cmd_time = self.get_clock().now()
        self.timed_out = False
        self.publish_wheels(left_mps, right_mps)

    def watchdog(self):
        if self.last_cmd_time is None:
            return
        age_s = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
        if age_s <= CMD_VEL_TIMEOUT_S:
            return
        if not self.timed_out:
            self.get_logger().warn(f'No cmd_vel for {CMD_VEL_TIMEOUT_S} s, stopping the wheels')
            self.timed_out = True
        self.publish_wheels(0.0, 0.0)

    def publish_wheels(self, left_mps, right_mps):
        out = LeftRightFloat32()
        out.left = left_mps / METERS_PER_TICK
        out.right = right_mps / METERS_PER_TICK
        out.seq_num = self.seq_num
        self.velocity_publisher.publish(out)
        self.seq_num += 1


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToRoboclawNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
