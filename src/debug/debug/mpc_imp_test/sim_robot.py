#!/usr/bin/env python3
"""Barebones simulated robot for testing mpc_path_follower - not a real robot model.

Integrates cmd_vel (geometry_msgs/Twist) through the same forward-Euler unicycle model
UnicycleMPC uses internally, republishing the result as nav_msgs/Odometry on /odom and
broadcasting the matching odom -> base_link transform on /tf - standing in for both jobs
the EKF will eventually do. Delete once real hardware or a full simulator exists.
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

ODOM_TOPIC = '/odom'
CMD_VEL_TOPIC = 'cmd_vel'
SIM_RATE_HZ = 50.0


def quaternion_from_yaw(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)  # (qz, qw); qx = qy = 0 for a planar pose


class SimRobotNode(Node):
    def __init__(self):
        super().__init__('sim_robot')

        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_theta', 0.0)
        self._pose = np.array([
            float(self.get_parameter('initial_x').value),
            float(self.get_parameter('initial_y').value),
            float(self.get_parameter('initial_theta').value),
        ])
        self._cmd = np.array([0.0, 0.0])  # (v, omega), last received cmd_vel

        self.create_subscription(Twist, CMD_VEL_TOPIC, self._cmd_vel_callback, 10)
        self._odom_publisher = self.create_publisher(Odometry, ODOM_TOPIC, 10)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(1.0 / SIM_RATE_HZ, self._step)

    def _cmd_vel_callback(self, msg):
        self._cmd = np.array([msg.linear.x, msg.angular.z])

    def _step(self):
        dt = 1.0 / SIM_RATE_HZ
        v, omega = self._cmd
        theta = self._pose[2]
        self._pose = self._pose + dt * np.array([v * np.cos(theta), v * np.sin(theta), omega])

        stamp = self.get_clock().now().to_msg()
        qz, qw = quaternion_from_yaw(self._pose[2])

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = float(self._pose[0])
        odom.pose.pose.position.y = float(self._pose[1])
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self._odom_publisher.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = float(self._pose[0])
        tf.transform.translation.y = float(self._pose[1])
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = SimRobotNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
