#!/usr/bin/env python3
"""Pure-pursuit controller node for the Aion R6, without the safety layer.

Subscribes: /odometry/filtered, /vla/action_chunk
Publishes:  cmd_vel
"""
import rclpy
from aion_msgs.msg import ActionChunk
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from .constants import ACTION_CHUNK_TOPIC, ODOM_TOPIC, CMD_VEL_TOPIC
from .frames import yaw_from_quaternion
from .pure_pursuit import PurePursuit

CONTROL_PERIOD_SEC = 1.0 / 30.0   # 30 Hz control rate


class PurePursuitControllerNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')

        self._current_pose = None          # (x, y, theta), latest odometry sample
        self._current_action_chunk = None
        self._pure_pursuit = PurePursuit()

        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, 10)
        self.create_subscription(ActionChunk, ACTION_CHUNK_TOPIC, self.action_chunk_callback, 10)
        self._cmd_vel_publisher = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)

        self.create_timer(CONTROL_PERIOD_SEC, self.control_loop)

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        self._current_pose = (position.x, position.y, yaw_from_quaternion(msg.pose.pose.orientation))

    def action_chunk_callback(self, msg):
        self._current_action_chunk = msg

    def control_loop(self):
        if self._current_pose is None or self._current_action_chunk is None:
            self.get_logger().warn('No pose/action chunk received yet, skipping tick')
            return

        linear_vel, angular_vel = self._pure_pursuit.nominal(self._current_action_chunk, self._current_pose)

        cmd = Twist()
        cmd.linear.x = float(linear_vel)
        cmd.angular.z = float(angular_vel)
        self._cmd_vel_publisher.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
