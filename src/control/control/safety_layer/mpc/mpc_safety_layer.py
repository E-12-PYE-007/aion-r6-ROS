"""
MPC safety layer for the Aion R6: takes VLA action chunks and publishes cmd_vel.

Each chunk is resampled to the solver's time step and tracked by an NMPC (solver_setup.py)
that also keeps the robot outside the ESDF safety margin.

Subscribes: /odometry/filtered, /vla/action_chunk, /nvblox_node/static_map_slice
Publishes:  cmd_vel, safety_layer/slack (largest margin slack over the horizon)
Parameters: patch_size, patch_resolution (ESDF patch geometry; defaults in common/constants.py),
            print_solve_time (log the NLP solve time every tick as 'solve_time_ms=<value>')
"""
import math
import time

import numpy as np
import rclpy
from aion_msgs.msg import ActionChunk
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nvblox_msgs.msg import DistanceMapSlice
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from ..common.constants import (
    ACTION_CHUNK_TOPIC, ODOM_TOPIC, ODOM_FRAME, BASE_FRAME, CMD_VEL_TOPIC, ESDF_TOPIC, SLACK_TOPIC,
    CONTROL_LOOP_DT, PATCH_SIZE, PATCH_RESOLUTION, N_WAYPOINTS,
)
from ..common.esdf_map import EsdfMap
from ..common.frames import yaw_from_quaternion, wrap_to_pi, transform_pose_2d
from .constants import WAYPOINT_DT, MAX_CONSECUTIVE_SOLVE_FAILURES
from .solver_setup import UnicycleMPC


def interpolate_path(times, poses, query_times):
    """Linear interpolation for x and y; sin/cos interpolation for theta so it wraps correctly."""
    x = np.interp(query_times, times, poses[:, 0])
    y = np.interp(query_times, times, poses[:, 1])
    theta = np.arctan2(
        np.interp(query_times, times, np.sin(poses[:, 2])),
        np.interp(query_times, times, np.cos(poses[:, 2])),
    )
    return np.stack([x, y, theta], axis=1)


class MpcSafetyLayerNode(Node):
    def __init__(self):
        super().__init__('mpc_safety_layer')

        self._current_pose = None          # (x, y, theta), latest odometry sample
        self._chunk_t0 = None              # clock time the current chunk was received at
        self._consecutive_solve_failures = 0
        self._interpolated_path = None     # (n, 3) odom-frame poses, resampled to the solver's dt

        self.declare_parameter('patch_size', PATCH_SIZE)
        self.declare_parameter('patch_resolution', PATCH_RESOLUTION)
        self.declare_parameter('print_solve_time', True)
        self._print_solve_time = bool(self.get_parameter('print_solve_time').value)
        patch_size = int(self.get_parameter('patch_size').value)
        patch_resolution = float(self.get_parameter('patch_resolution').value)
        self._mpc = UnicycleMPC(patch_size=patch_size, patch_resolution=patch_resolution)
        self._esdf_map = EsdfMap(patch_size)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, 10)
        self.create_subscription(ActionChunk, ACTION_CHUNK_TOPIC, self.action_chunk_callback, 10)
        self.create_subscription(DistanceMapSlice, ESDF_TOPIC, self.esdf_callback, 10)
        self._cmd_vel_publisher = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self._slack_publisher = self.create_publisher(Float32, SLACK_TOPIC, 10)

        self.create_timer(CONTROL_LOOP_DT, self.control_loop)

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        self._current_pose = np.array([p.x, p.y, yaw_from_quaternion(msg.pose.pose.orientation)])

    def esdf_callback(self, msg):
        self._esdf_map.store(msg)

    def action_chunk_callback(self, msg):
        """Transform the chunk into the odom frame and resample it at the solver's time step."""
        try:
            anchor_tf = self._tf_buffer.lookup_transform(ODOM_FRAME, BASE_FRAME, Time())
        except TransformException as ex:
            self.get_logger().warn(f'Could not look up {ODOM_FRAME} -> {BASE_FRAME}: {ex}')
            return

        self._chunk_t0 = self.get_clock().now().nanoseconds * 1e-9

        # The chunk holds 8 future deltas; prepend the identity so the path starts at the anchor.
        relative_poses = [(0.0, 0.0, 0.0)] + [(p.x, p.y, p.theta) for p in msg.relative_poses]
        world_path = np.array([transform_pose_2d(x, y, theta, anchor_tf) for x, y, theta in relative_poses])

        chunk_times = np.arange(N_WAYPOINTS + 1) * WAYPOINT_DT
        fine_times = np.arange(0.0, chunk_times[-1] + 1e-9, self._mpc.dt)
        self._interpolated_path = interpolate_path(chunk_times, world_path, fine_times)

    def build_xref(self):
        """(x0, x_ref) for this tick, or None until a pose and a chunk have arrived. x_ref is
        (N+1, 3): the resampled chunk indexed from the current time. Beyond the chunk's span
        the position extrapolates linearly from its last step instead of holding the final
        sample, so the terminal target never sits still on an obstacle."""
        if self._current_pose is None or self._interpolated_path is None:
            return None

        x0 = self._current_pose

        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self._chunk_t0
        idx0 = max(round(elapsed / self._mpc.dt), 0)

        last_idx = len(self._interpolated_path) - 1
        raw_idxs = idx0 + np.arange(self._mpc.N + 1)
        x_ref = self._interpolated_path[np.clip(raw_idxs, 0, last_idx)].copy()

        overflow = raw_idxs > last_idx
        if np.any(overflow) and last_idx >= 1:
            last, prev = self._interpolated_path[last_idx], self._interpolated_path[last_idx - 1]
            step = last[:2] - prev[:2]
            # Heading from the step itself; extrapolating raw angles would not wrap.
            heading = math.atan2(step[1], step[0]) if np.linalg.norm(step) > 1e-9 else last[2]
            steps_beyond = (raw_idxs[overflow] - last_idx).astype(float)
            x_ref[overflow, 0] = last[0] + steps_beyond * step[0]
            x_ref[overflow, 1] = last[1] + steps_beyond * step[1]
            x_ref[overflow, 2] = heading

        return x0, x_ref

    def build_uref(self, x_ref):
        """(N, 2) [v, omega] from finite differences of x_ref; the chunk gives only poses."""
        dt = self._mpc.dt
        dx = x_ref[1:, 0] - x_ref[:-1, 0]
        dy = x_ref[1:, 1] - x_ref[:-1, 1]
        dtheta = wrap_to_pi(x_ref[1:, 2] - x_ref[:-1, 2])
        heading = x_ref[:-1, 2]
        v = (dx * np.cos(heading) + dy * np.sin(heading)) / dt
        omega = dtheta / dt
        return np.stack([v, omega], axis=1)

    def control_loop(self):
        result = self.build_xref()
        if result is None:
            self.get_logger().warn('No pose/action chunk received yet, skipping tick')
            return
        x0, x_ref = result

        try:
            esdf_result = self._esdf_map.patch_at(self._tf_buffer, x0)
        except TransformException as ex:
            self.get_logger().warn(f'Could not look up odom -> {self._esdf_map.frame_id}: {ex}')
            esdf_result = None
        if esdf_result is None:
            self.get_logger().warn('No ESDF/transform available yet, skipping tick')
            return
        psi, esdf_patch = esdf_result

        u_ref = self.build_uref(x_ref)

        solve_start = time.perf_counter()
        u0, _, solved_ok = self._mpc.solve(x0, x_ref, u_ref, psi, esdf_patch)
        if self._print_solve_time:
            self.get_logger().info(f'solve_time_ms={(time.perf_counter() - solve_start) * 1e3:.3f}')

        slacks = self._mpc.extract_slacks(self._mpc._w_guess)
        self._slack_publisher.publish(Float32(data=float(slacks.max())))
        if slacks.max() > 1e-4:
            self.get_logger().warn(f'obstacle slack in use: max S_k={slacks.max():.4f} m over horizon')

        if solved_ok:
            self._consecutive_solve_failures = 0
        else:
            self._consecutive_solve_failures += 1
            self.get_logger().warn(
                f'MPC solver did not report success ({self._consecutive_solve_failures} in a row)')
            self._mpc.reset()   # keep a bad iterate out of the next warm start

        cmd = Twist()
        if self._consecutive_solve_failures < MAX_CONSECUTIVE_SOLVE_FAILURES:
            cmd.linear.x = float(u0[0])
            cmd.angular.z = float(u0[1])
        else:
            self.get_logger().warn('Too many consecutive solver failures, stopping')
        self._cmd_vel_publisher.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MpcSafetyLayerNode()
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
