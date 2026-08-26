#!/usr/bin/env python3
"""Wheel-odometry node for the Aion R6.

Subscribes to encoder_counts (aion_msgs/LeftRightInt32) and integrates tick
deltas into a pose + velocity estimate, published as nav_msgs/Odometry on
/odometry/wheel.
"""

import math

import rclpy
from aion_msgs.msg import LeftRightInt32
from nav_msgs.msg import Odometry
from rclpy.node import Node

WHEEL_RADIUS_M = 0.0804              # Must match roboclaw_tests.py.
ENCODER_COUNTS_PER_WHEEL_REV = 1100  # Must match roboclaw_tests.py.
EFFECTIVE_TRACK_WIDTH_M = 0.5132     # Must match roboclaw_tests.py.

METERS_PER_TICK = (2.0 * math.pi * WHEEL_RADIUS_M) / ENCODER_COUNTS_PER_WHEEL_REV

ODOM_TOPIC = '/odometry/wheel'  # Raw wheel-odometry source; the EKF's fused output takes over this role later.


class EncoderLocalisationNode(Node):
    def __init__(self):
        super().__init__('encoder_localisation')

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_stamp = None  # None until the first encoder message establishes a baseline

        self.odom_publisher = self.create_publisher(Odometry, ODOM_TOPIC, 10)
        self.create_subscription(LeftRightInt32, 'encoder_counts', self.encoder_callback, 10)

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

        linear_vel = 0.0
        angular_vel = 0.0
        if self.last_stamp is not None:
            dt = (now - self.last_stamp).nanoseconds / 1e9
            if dt > 0.0:
                linear_vel = dist_center / dt
                angular_vel = dtheta / dt
        self.last_stamp = now

        self.publish_odometry(now, linear_vel, angular_vel)

    def publish_odometry(self, stamp, linear_vel, angular_vel):
        odom = Odometry()
        odom.header.stamp = stamp.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = math.sin(self.theta / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.theta / 2.0)

        odom.twist.twist.linear.x = linear_vel
        odom.twist.twist.angular.z = angular_vel

        self.odom_publisher.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = EncoderLocalisationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
