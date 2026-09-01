#!/usr/bin/env python3
"""cmd_vel-to-Roboclaw adapter for the Aion R6.

Subscribes to cmd_vel (geometry_msgs/Twist) and converts body velocity into
per-wheel Roboclaw velocity commands, published as aion_msgs/LeftRightFloat32
on set_motor_velocity.
"""

import math

import rclpy
from aion_msgs.msg import LeftRightFloat32
from geometry_msgs.msg import Twist
from rclpy.node import Node

WHEEL_RADIUS_M = 0.0804              # Must match roboclaw_tests.py.
ENCODER_COUNTS_PER_WHEEL_REV = 1100  # Must match roboclaw_tests.py.
EFFECTIVE_TRACK_WIDTH_M = 0.5132     # Must match roboclaw_tests.py.

METERS_PER_TICK = (2.0 * math.pi * WHEEL_RADIUS_M) / ENCODER_COUNTS_PER_WHEEL_REV


class CmdVelToRoboclawNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_roboclaw')

        self.seq_num = 1

        self.velocity_publisher = self.create_publisher(LeftRightFloat32, 'set_motor_velocity', 1)
        self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)

    def cmd_vel_callback(self, msg):
        linear_vel = msg.linear.x
        angular_vel = msg.angular.z

        left_mps = linear_vel - angular_vel * EFFECTIVE_TRACK_WIDTH_M / 2.0
        right_mps = linear_vel + angular_vel * EFFECTIVE_TRACK_WIDTH_M / 2.0

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
