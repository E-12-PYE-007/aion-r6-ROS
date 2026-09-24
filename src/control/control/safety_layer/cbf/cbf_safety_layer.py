"""
CBF safety layer for the Aion R6: takes VLA action chunks and publishes cmd_vel.

A pure-pursuit controller turns the latest chunk into a nominal (v, omega), and a CBF-QP
filter (cbf_qp.py) changes it as little as needed to keep the robot outside the ESDF safety
margin.

Subscribes: /odometry/filtered, /vla/action_chunk, /nvblox_node/static_map_slice
Publishes:  cmd_vel, safety_layer/slack, cbf/h
Parameters: patch_size, patch_resolution (ESDF patch geometry; defaults in common/constants.py),
            print_solve_time (log the QP solve time every tick as 'solve_time_ms=<value>')
"""
import time

import numpy as np
import rclpy
from aion_msgs.msg import ActionChunk
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nvblox_msgs.msg import DistanceMapSlice
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from ..common.constants import (
    ACTION_CHUNK_TOPIC, ODOM_TOPIC, CMD_VEL_TOPIC, ESDF_TOPIC, SLACK_TOPIC, CONTROL_LOOP_DT,
    PATCH_SIZE, PATCH_RESOLUTION,
)
from ..common.esdf_map import EsdfMap
from ..common.frames import yaw_from_quaternion
from ..common.pure_pursuit import PurePursuit
from .cbf_qp import CbfFilter
from .constants import CBF_H_TOPIC


class CbfSafetyLayerNode(Node):
    def __init__(self):
        super().__init__('cbf_safety_layer')

        self._current_pose = None          # (x, y, theta), latest odometry sample
        self._current_action_chunk = None

        self.declare_parameter('patch_size', PATCH_SIZE)
        self.declare_parameter('patch_resolution', PATCH_RESOLUTION)
        self.declare_parameter('print_solve_time', True)
        self._print_solve_time = bool(self.get_parameter('print_solve_time').value)
        patch_size = int(self.get_parameter('patch_size').value)
        patch_resolution = float(self.get_parameter('patch_resolution').value)
        self._cbf = CbfFilter(patch_size=patch_size, patch_resolution=patch_resolution)
        self._esdf_map = EsdfMap(patch_size)
        self._pure_pursuit = PurePursuit()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, 10)
        self.create_subscription(ActionChunk, ACTION_CHUNK_TOPIC, self.action_chunk_callback, 10)
        self.create_subscription(DistanceMapSlice, ESDF_TOPIC, self.esdf_callback, 10)
        self._cmd_vel_publisher = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self._h_publisher = self.create_publisher(Float32, CBF_H_TOPIC, 10)
        self._slack_publisher = self.create_publisher(Float32, SLACK_TOPIC, 10)

        self.create_timer(CONTROL_LOOP_DT, self.control_loop)

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        self._current_pose = np.array([p.x, p.y, yaw_from_quaternion(msg.pose.pose.orientation)])

    def esdf_callback(self, msg):
        self._esdf_map.store(msg)

    def action_chunk_callback(self, msg):
        self._current_action_chunk = msg

    def control_loop(self):
        if self._current_pose is None or self._current_action_chunk is None:
            self.get_logger().warn('No pose/action chunk received yet, skipping tick')
            return
        x0 = self._current_pose

        try:
            esdf_result = self._esdf_map.patch_at(self._tf_buffer, x0)
        except TransformException as ex:
            self.get_logger().warn(f'Could not look up odom -> {self._esdf_map.frame_id}: {ex}')
            esdf_result = None
        if esdf_result is None:
            self.get_logger().warn('No ESDF/transform available yet, skipping tick')
            return
        psi, esdf_patch = esdf_result

        v_nom, omega_nom = self._pure_pursuit.nominal(self._current_action_chunk, x0)
        solve_start = time.perf_counter()
        u_safe, h, slack = self._cbf.filter(x0, psi, esdf_patch, [v_nom, omega_nom])
        if self._print_solve_time:
            self.get_logger().info(f'solve_time_ms={(time.perf_counter() - solve_start) * 1e3:.3f}')

        self._h_publisher.publish(Float32(data=h))
        self._slack_publisher.publish(Float32(data=slack))
        if slack > 1e-4:
            self.get_logger().warn(f'CBF constraint slack in use: {slack:.4f} (h={h:.4f})')

        cmd = Twist()
        cmd.linear.x = float(u_safe[0])
        cmd.angular.z = float(u_safe[1])
        self._cmd_vel_publisher.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = CbfSafetyLayerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:   # a second SIGINT (terminal + launch) during shutdown
            pass


if __name__ == '__main__':
    main()
