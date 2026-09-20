"""
MPC Path follower for Aion R6.
    Consumes VLA action chunks, resamples to compute reference poses
    and inputs, then calls the solver and publishes output to cmd_vel
"""
import math

import numpy as np
import rclpy
from aion_msgs.msg import ActionChunk
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from nvblox_msgs.msg import DistanceMapSlice
from rclpy.node import Node
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from .constants import (
    ACTION_CHUNK_TOPIC, ODOM_TOPIC, ODOM_FRAME, BASE_FRAME, CMD_VEL_TOPIC, ESDF_TOPIC,
    WAYPOINT_DT, MAX_CONSECUTIVE_SOLVE_FAILURES, CONTROL_LOOP_DT, PATCH_RESOLUTION,
)
from .esdf_map import EsdfMap
from .solver_setup import UnicycleMPC

N_WAYPOINTS = 8  # fixed by aion_msgs/ActionChunk.msg (Pose2D[8] relative_poses) - not a tuning knob


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def transform_pose_2d(x, y, theta, tf):
    """Apply a TransformStamped to a planar (x, y, theta) pose via tf2's own Pose transform."""
    pose_in = Pose()
    pose_in.position.x = x
    pose_in.position.y = y
    pose_in.orientation.z = math.sin(theta / 2.0)
    pose_in.orientation.w = math.cos(theta / 2.0)

    pose_out = do_transform_pose(pose_in, tf)

    return pose_out.position.x, pose_out.position.y, yaw_from_quaternion(pose_out.orientation)


def interpolate_path(times, poses, query_times):
    """Linear interpolation for x/y; angle-aware (sin/cos) interpolation for theta, so a
    heading crossing due-south doesn't jump from -179deg to +179deg mid-interpolation.
    """
    x = np.interp(query_times, times, poses[:, 0])
    y = np.interp(query_times, times, poses[:, 1])
    theta = np.arctan2(
        np.interp(query_times, times, np.sin(poses[:, 2])),
        np.interp(query_times, times, np.cos(poses[:, 2])),
    )
    return np.stack([x, y, theta], axis=1)


class MpcPathFollowerNode(Node):
    def __init__(self):
        super().__init__('mpc_path_follower')

        self._current_pose = None          # (x, y, theta), latest /odom sample
        self._chunk_t0 = None              # clock time the current chunk was received at
        self._consecutive_solve_failures = 0
        self._interpolated_path = None     # (n, 3) odom-frame poses, resampled to mpc.dt

        # Overridable so a test can match a real ESDF source's actual resolution
        # (e.g. a recorded bag) without changing the production default in constants.py.
        self.declare_parameter('patch_resolution', PATCH_RESOLUTION)
        patch_resolution = float(self.get_parameter('patch_resolution').value)
        self._mpc = UnicycleMPC(patch_resolution=patch_resolution)
        self._esdf_map = EsdfMap()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._odom_subscription = self.create_subscription(
            Odometry, ODOM_TOPIC, self.odom_callback, 10)
        self._action_chunk_subscription = self.create_subscription(
            ActionChunk, ACTION_CHUNK_TOPIC, self.action_chunk_callback, 10)
        self._esdf_subscription = self.create_subscription(
            DistanceMapSlice, ESDF_TOPIC, self.esdf_callback, 10)
        self._cmd_vel_publisher = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)

        self._control_timer = self.create_timer(CONTROL_LOOP_DT, self.control_loop)

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        self._current_pose = np.array([p.x, p.y, yaw_from_quaternion(msg.pose.pose.orientation)])

    def esdf_callback(self, msg):
        self._esdf_map.store(msg)

    # On reception of an action chunk, transforms into world coordinates (odom) and interpolates
    # to provide sampling at the same dt as the MPC.
    def action_chunk_callback(self, msg):
        try:
            anchor_tf = self._tf_buffer.lookup_transform(ODOM_FRAME, BASE_FRAME, Time())
        except TransformException as ex:
            self.get_logger().warn(f'Could not look up {ODOM_FRAME} -> {BASE_FRAME}: {ex}')
            return

        self._chunk_t0 = self.get_clock().now().nanoseconds * 1e-9

        # The wire format is 8 future deltas, none of them "now" - prepend the identity
        # pose so the path has a t=0 sample: the anchor itself.
        relative_poses = [(0.0, 0.0, 0.0)] + [(p.x, p.y, p.theta) for p in msg.relative_poses]
        world_path = np.array([transform_pose_2d(x, y, theta, anchor_tf) for x, y, theta in relative_poses])

        chunk_times = np.arange(N_WAYPOINTS + 1) * WAYPOINT_DT  # 0, dt, ..., N_WAYPOINTS*dt
        fine_times = np.arange(0.0, chunk_times[-1] + 1e-9, self._mpc.dt)
        self._interpolated_path = interpolate_path(chunk_times, world_path, fine_times)

    def build_xref(self):
        """Returns (x0, x_ref) for the current tick, or None if there's nothing to solve against yet.
        x_ref is (N+1, 3): the interpolated chunk path, indexed from wherever "now" falls in its
        own timeline. Past the chunk's own span, the last sample is held rather than extrapolated.
        """
        if self._current_pose is None or self._interpolated_path is None:
            return None

        x0 = self._current_pose

        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self._chunk_t0
        idx0 = max(round(elapsed / self._mpc.dt), 0)

        last_idx = len(self._interpolated_path) - 1
        idxs = np.clip(idx0 + np.arange(self._mpc.N + 1), 0, last_idx)
        x_ref = self._interpolated_path[idxs]

        return x0, x_ref

    def build_esdf_inputs(self, x0):
        """(psi, esdf_patch) for the obstacle constraint, or None if the ESDF or
        transform isn't available yet. Looked up against esdf_map.frame_id rather
        than a hardcoded 'map', since these test bags have no separate map frame."""
        if self._esdf_map.vals is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(self._esdf_map.frame_id, ODOM_FRAME, Time())
        except TransformException as ex:
            self.get_logger().warn(f'Could not look up {ODOM_FRAME} -> {self._esdf_map.frame_id}: {ex}')
            return None

        x_esdf, y_esdf, _ = transform_pose_2d(x0[0], x0[1], x0[2], tf)
        psi = yaw_from_quaternion(tf.transform.rotation)
        return psi, self._esdf_map.get_patch([x_esdf, y_esdf])

    def build_uref(self, x_ref):
        """(N, 2) [v, omega] recovered by finite-differencing consecutive x_ref samples -
        the chunk only ever gives poses, the solver's cost function also wants inputs.
        """
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

        esdf_result = self.build_esdf_inputs(x0)
        if esdf_result is None:
            self.get_logger().warn('No ESDF/transform available yet, skipping tick')
            return
        psi, esdf_patch = esdf_result

        u_ref = self.build_uref(x_ref)

        u0, _, solved_ok = self._mpc.solve(x0, x_ref, u_ref, psi, esdf_patch)

        if solved_ok:
            self._consecutive_solve_failures = 0
        else:
            self._consecutive_solve_failures += 1
            self.get_logger().warn(
                f'MPC solver did not report success ({self._consecutive_solve_failures} in a row)')
            # Don't let a bad iterate poison the next tick's warm start.
            self._mpc.reset()

        cmd = Twist()
        if self._consecutive_solve_failures < MAX_CONSECUTIVE_SOLVE_FAILURES:
            cmd.linear.x = float(u0[0])
            cmd.angular.z = float(u0[1])
        else:
            self.get_logger().warn('Too many consecutive solver failures, stopping')
        self._cmd_vel_publisher.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MpcPathFollowerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
