"""
CBF safety layer for the Aion R6: takes VLA action chunks and publishes cmd_vel.

A pure-pursuit controller turns the latest chunk into a nominal (v, omega), and a CBF-QP
filter (cbf_qp.py) changes it as little as needed to keep the robot outside the ESDF safety
margin.

Subscribes: /odometry/local, /agvla/action_chunk, /nvblox_node/static_map_slice
Publishes:  cmd_vel, safety_layer/slack, cbf/h, safety_layer/reversing (True while the reverse recovery drives),
            safety_layer/cmd_vel_nominal (pure-pursuit command before the filter; diff against
            cmd_vel to see when and how much the CBF filter intervenes),
            safety_layer/filtering_active (True when the filter changed v or omega)
Parameters: patch_size, patch_resolution (ESDF patch geometry; defaults in safety_layer/constants.py),
            print_solve_time (log the QP solve time every tick as 'solve_time_ms=<value>')
"""
import math
import time

import numpy as np
import rclpy
from custom_msgs.msg import ActionChunk
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nvblox_msgs.msg import DistanceMapSlice
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from ..constants import ACTION_CHUNK_TOPIC, ODOM_TOPIC, CMD_VEL_TOPIC
from ..frames import yaw_from_quaternion
from ..pure_pursuit import PurePursuit
from .cbf_qp import CbfFilter
from .constants import (
    ESDF_TOPIC, SLACK_TOPIC, CBF_H_TOPIC, CMD_VEL_NOMINAL_TOPIC, FILTERING_ACTIVE_TOPIC,
    FILTERING_EPS, CONTROL_LOOP_DT, PATCH_SIZE, PATCH_RESOLUTION, REVERSING_TOPIC,
    REVERSE_GRAD_MIN, ODOM_READY_TICKS, ODOM_YAW_JUMP_RAD,
)
from .esdf_map import EsdfMap
from .reverse_recovery import ReverseRecovery


class CbfSafetyLayerNode(Node):
    def __init__(self):
        super().__init__('cbf_safety_layer')

        self._current_pose = None          # (x, y, theta), latest odometry sample
        self._current_action_chunk = None
        self._odom_ready = False           # True once yaw has been stable for ODOM_READY_TICKS
        self._odom_stable_ticks = 0

        self.declare_parameter('patch_size', PATCH_SIZE)
        self.declare_parameter('patch_resolution', PATCH_RESOLUTION)
        self.declare_parameter('print_solve_time', False)
        self._print_solve_time = bool(self.get_parameter('print_solve_time').value)
        patch_size = int(self.get_parameter('patch_size').value)
        patch_resolution = float(self.get_parameter('patch_resolution').value)
        self._cbf = CbfFilter(patch_size=patch_size, patch_resolution=patch_resolution)
        self._esdf_map = EsdfMap(patch_size)
        self._pure_pursuit = PurePursuit()
        self._reverse_recovery = ReverseRecovery()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, 10)
        self.create_subscription(ActionChunk, ACTION_CHUNK_TOPIC, self.action_chunk_callback, 10)
        self.create_subscription(DistanceMapSlice, ESDF_TOPIC, self.esdf_callback, 10)
        self._cmd_vel_publisher = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self._h_publisher = self.create_publisher(Float32, CBF_H_TOPIC, 10)
        self._slack_publisher = self.create_publisher(Float32, SLACK_TOPIC, 10)
        self._nominal_publisher = self.create_publisher(Twist, CMD_VEL_NOMINAL_TOPIC, 10)
        self._filtering_active_publisher = self.create_publisher(Bool, FILTERING_ACTIVE_TOPIC, 10)
        self._reversing_publisher = self.create_publisher(Bool, REVERSING_TOPIC, 10)

        self.create_timer(CONTROL_LOOP_DT, self.control_loop)

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        if not self._odom_ready and self._current_pose is not None:
            delta = abs(math.atan2(math.sin(yaw - self._current_pose[2]),
                                   math.cos(yaw - self._current_pose[2])))
            if delta < ODOM_YAW_JUMP_RAD:
                self._odom_stable_ticks += 1
            else:
                self._odom_stable_ticks = 0
            if self._odom_stable_ticks >= ODOM_READY_TICKS:
                self._odom_ready = True
                self.get_logger().info('Odometry yaw converged — CBF filter active')
        self._current_pose = np.array([p.x, p.y, yaw])

    def esdf_callback(self, msg):
        self._esdf_map.store(msg)

    def action_chunk_callback(self, msg):
        self._current_action_chunk = msg

    def control_loop(self):
        if self._current_pose is None or self._current_action_chunk is None:
            self.get_logger().warn('No pose/action chunk received yet, skipping tick')
            return
        x0 = self._current_pose

        if not self._odom_ready:
            v_nom, omega_nom = self._pure_pursuit.nominal(self._current_action_chunk, x0)
            cmd = Twist()
            cmd.linear.x = float(v_nom)
            cmd.angular.z = float(omega_nom)
            self._cmd_vel_publisher.publish(cmd)
            self._filtering_active_publisher.publish(Bool(data=False))
            self._reversing_publisher.publish(Bool(data=False))
            return

        try:
            esdf_result = self._esdf_map.patch_at(self._tf_buffer, x0)
        except TransformException as ex:
            self.get_logger().warn(f'Could not look up odom -> {self._esdf_map.frame_id}: {ex}')
            esdf_result = None
        if esdf_result is None:
            self.get_logger().warn('No ESDF/transform available yet, skipping tick')
            return
        psi, esdf_patch, patch_offset = esdf_result

        v_nom, omega_nom = self._pure_pursuit.nominal(self._current_action_chunk, x0)
        solve_start = time.perf_counter()
        u_safe, h, slack, escape_dir = self._cbf.filter(x0, psi, patch_offset, esdf_patch, [v_nom, omega_nom])
        if self._print_solve_time:
            self.get_logger().info(f'solve_time_ms={(time.perf_counter() - solve_start) * 1e3:.3f}')

        self._h_publisher.publish(Float32(data=h))
        self._slack_publisher.publish(Float32(data=slack))
        if slack > 1e-4:
            self.get_logger().warn(f'CBF constraint slack in use: {slack:.4f} (h={h:.4f})')

        nominal = Twist()
        nominal.linear.x = float(v_nom)
        nominal.angular.z = float(omega_nom)
        self._nominal_publisher.publish(nominal)
        # Filter is intervening if it changed either channel beyond the solver noise floor.
        filtering_active = (abs(u_safe[0] - v_nom) > FILTERING_EPS
                            or abs(u_safe[1] - omega_nom) > FILTERING_EPS)
        self._filtering_active_publisher.publish(Bool(data=bool(filtering_active)))

        was_reversing = self._reverse_recovery.reversing
        # Heading away from the binding obstacle (odom frame) for the in-place turn to face toward;
        # None when the ESDF gradient is too weak to trust, so the turn phase is skipped.
        away_yaw = None
        if np.hypot(escape_dir[0], escape_dir[1]) > REVERSE_GRAD_MIN:
            away_yaw = float(np.arctan2(escape_dir[1], escape_dir[0]))
        reverse_cmd = self._reverse_recovery.update(x0, slack, [v_nom, omega_nom], u_safe, away_yaw)
        if reverse_cmd is not None and not was_reversing:
            self.get_logger().warn('Robot stuck (slack high or stalled at clearance boundary), turning in place then backing up')
        elif reverse_cmd is None and was_reversing:
            self.get_logger().info('Recovery finished, resuming the filter')
        self._reversing_publisher.publish(Bool(data=reverse_cmd is not None))
        v_out, omega_out = u_safe if reverse_cmd is None else reverse_cmd

        cmd = Twist()
        cmd.linear.x = float(v_out)
        cmd.angular.z = float(omega_out)
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
