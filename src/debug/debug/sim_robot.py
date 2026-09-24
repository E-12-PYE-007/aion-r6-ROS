#!/usr/bin/env python3
"""Bench-test robot: integrates cmd_vel through a unicycle model, republishes the result as
/odom and the odom -> base_link transform, and logs every step to a CSV file.

Subscribes: cmd_vel (Twist), /vla/action_chunk (ActionChunk), safety_layer/slack (Float32).
Publishes:  /odom (Odometry), /tf (odom -> base_link).

Start pose is DEFAULT_START plus the dx / dy / dtheta parameters (a delta in the odom frame).
The CSV is flushed after every row, so a killed run keeps everything logged so far. Format
and columns: src/safety_layer_design.html, "Log format".

Parameters (all optional):
  dx, dy, dtheta   start-pose delta from DEFAULT_START [m, m, rad]
  goal_distance    written to the log header only, for the plotter [m]
  esdf_bag         written to the log header only, for the plotter
  controller       label written to the log header ('cbf', 'mpc', ...)
  output_path      CSV path; default sim_logs/run_<timestamp>.csv
  max_duration     stop after this many seconds of ROS time; 0 runs until killed (default 30)
"""
import math
import os
import time

import numpy as np
import rclpy
from aion_msgs.msg import ActionChunk
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster

ODOM_TOPIC = '/odom'
CMD_VEL_TOPIC = 'cmd_vel'
ACTION_CHUNK_TOPIC = '/vla/action_chunk'
SLACK_TOPIC = 'safety_layer/slack'

SIM_RATE_HZ = 50.0
MAX_STEP_S = 0.1  # a stalled timer never integrates more than this in one step
DEFAULT_MAX_DURATION_S = 30.0

# Head-on at the single obstacle in esdf_single_obs: the ESDF cone's apex, heading down its axis.
DEFAULT_START = (0.25, 0.0, 0.0)
DEFAULT_ESDF_BAG = (
    '/home/djstrahan/aion-r6-vla-master/aion-r6-ROS/safety_layer_bench_testing/'
    'rosbags/esdf_single_obs/esdf_single_obs_0.db3'
)

N_WAYPOINTS = 8  # fixed by aion_msgs/ActionChunk.msg (Pose2D[8] relative_poses)


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def csv_columns():
    chunk_cols = [f'c{i}{axis}' for i in range(N_WAYPOINTS) for axis in ('x', 'y', 'th')]
    return ['t', 'x', 'y', 'theta', 'v', 'omega', 'slack', 'seq_num'] + chunk_cols


class SimRobotNode(Node):
    def __init__(self):
        super().__init__('sim_robot')

        self.declare_parameter('dx', 0.0)
        self.declare_parameter('dy', 0.0)
        self.declare_parameter('dtheta', 0.0)
        self.declare_parameter('goal_distance', 6.0)
        self.declare_parameter('esdf_bag', DEFAULT_ESDF_BAG)
        self.declare_parameter('controller', 'unknown')
        self.declare_parameter('output_path', '')
        self.declare_parameter('max_duration', DEFAULT_MAX_DURATION_S)

        delta = np.array([float(self.get_parameter(n).value) for n in ('dx', 'dy', 'dtheta')])
        self._pose = np.array(DEFAULT_START) + delta
        self._pose[2] = wrap_to_pi(self._pose[2])
        self._max_duration = float(self.get_parameter('max_duration').value)

        self._cmd = (0.0, 0.0)        # (v, omega), last received cmd_vel
        self._slack = 0.0             # last received safety_layer/slack
        self._seq_num = -1            # latest action chunk, exactly as published
        self._chunk = [0.0] * (3 * N_WAYPOINTS)

        self.create_subscription(Twist, CMD_VEL_TOPIC, self._cmd_vel_callback, 10)
        self.create_subscription(ActionChunk, ACTION_CHUNK_TOPIC, self._chunk_callback, 10)
        self.create_subscription(Float32, SLACK_TOPIC, self._slack_callback, 10)
        self._odom_publisher = self.create_publisher(Odometry, ODOM_TOPIC, 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._log_file = self._open_log(delta)
        self._t0 = None
        self._last_time = None
        self.finished = False
        self.create_timer(1.0 / SIM_RATE_HZ, self._step)

    def _open_log(self, delta):
        path = str(self.get_parameter('output_path').value)
        if not path:
            path = os.path.join('sim_logs', time.strftime('run_%Y%m%d_%H%M%S.csv'))
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        f = open(path, 'w')
        f.write(f'# controller: {self.get_parameter("controller").value}\n')
        f.write(f'# esdf_bag: {self.get_parameter("esdf_bag").value}\n')
        f.write(f'# default_start: {DEFAULT_START[0]} {DEFAULT_START[1]} {DEFAULT_START[2]}\n')
        f.write(f'# start_delta: {delta[0]} {delta[1]} {delta[2]}\n')
        f.write(f'# start: {self._pose[0]} {self._pose[1]} {self._pose[2]}\n')
        f.write(f'# goal_distance: {self.get_parameter("goal_distance").value}\n')
        f.write(','.join(csv_columns()) + '\n')
        f.flush()
        self.get_logger().info(f'Logging to {path}; start pose {self._pose.round(4).tolist()}')
        return f

    def _cmd_vel_callback(self, msg):
        self._cmd = (msg.linear.x, msg.angular.z)

    def _slack_callback(self, msg):
        self._slack = float(msg.data)

    def _chunk_callback(self, msg):
        self._seq_num = int(msg.seq_num)
        flat = []
        for pose in msg.relative_poses:
            flat += [pose.x, pose.y, pose.theta]
        self._chunk = flat

    def _step(self):
        if self.finished:
            return

        now = self.get_clock().now()
        if self._t0 is None:
            self._t0 = self._last_time = now
        t = (now - self._t0).nanoseconds * 1e-9
        dt = min((now - self._last_time).nanoseconds * 1e-9, MAX_STEP_S)
        self._last_time = now

        v, omega = self._cmd
        theta = self._pose[2]
        self._pose = self._pose + dt * np.array([v * math.cos(theta), v * math.sin(theta), omega])
        self._pose[2] = wrap_to_pi(self._pose[2])

        self._publish_odom(now.to_msg())
        self._write_row(t)

        if self._max_duration > 0.0 and t >= self._max_duration:
            self._finish(f'reached max_duration ({self._max_duration} s)')

    def _finish(self, reason):
        self.finished = True
        self.close_log()
        self.get_logger().info(f'Stopping: {reason}. Log closed.')

    def _publish_odom(self, stamp):
        qz, qw = math.sin(self._pose[2] / 2.0), math.cos(self._pose[2] / 2.0)  # qx = qy = 0 for a planar pose

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = float(self._pose[0])
        odom.pose.pose.position.y = float(self._pose[1])
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(self._cmd[0])
        odom.twist.twist.angular.z = float(self._cmd[1])
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

    def _write_row(self, t):
        v, omega = self._cmd
        values = [t, self._pose[0], self._pose[1], self._pose[2], v, omega, self._slack]
        row = ','.join(f'{value:.5f}' for value in values)
        row += f',{self._seq_num},' + ','.join(f'{value:.5f}' for value in self._chunk)
        self._log_file.write(row + '\n')
        self._log_file.flush()

    def close_log(self):
        if not self._log_file.closed:
            self._log_file.close()


def main(args=None):
    rclpy.init(args=args)
    node = SimRobotNode()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close_log()
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:   # a second SIGINT (terminal + launch) during shutdown
            pass


if __name__ == '__main__':
    main()
