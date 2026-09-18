#!/usr/bin/env python3
"""Plots mpc_path_follower's closed loop: actual robot path, the current action chunk
(converted to world frame), and the static chicane reference shape. Renders an mp4 on
shutdown - buffers everything while running, no live encoding, so a ros2 launch bringing
this up needs a generous shutdown grace period for anything but a short run (the render
itself can take longer than launch's default SIGINT/SIGTERM timeouts allow).

Does NOT draw the MPC's predicted horizon: that's internal to mpc_path_follower and isn't
published on any topic, so reproducing it here would need a new message, not just consuming
what's already on the bus.
"""
import bisect
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from aion_msgs.msg import ActionChunk

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon

from .chicane_path import ChicanePath

ODOM_TOPIC = '/odom'
ACTION_CHUNK_TOPIC = '/vla/action_chunk'
CMD_VEL_TOPIC = 'cmd_vel'


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def relative_poses_to_world(rel_poses, anchor_pose):
    """(n, 3) poses relative to anchor_pose -> (n, 3) poses in world (odom) frame."""
    rel_poses = np.asarray(rel_poses, dtype=float)
    c, s = np.cos(anchor_pose[2]), np.sin(anchor_pose[2])
    world_x = anchor_pose[0] + rel_poses[:, 0] * c - rel_poses[:, 1] * s
    world_y = anchor_pose[1] + rel_poses[:, 0] * s + rel_poses[:, 1] * c
    world_theta = wrap_to_pi(anchor_pose[2] + rel_poses[:, 2])
    return np.stack([world_x, world_y, world_theta], axis=1)


def robot_triangle(pose, size=0.12):
    x, y, theta = pose
    pts = np.array([[size, 0.0], [-0.6 * size, 0.5 * size], [-0.6 * size, -0.5 * size]])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = pts @ rot.T
    pts[:, 0] += x
    pts[:, 1] += y
    return pts


class MpcPlotterNode(Node):
    def __init__(self):
        super().__init__('mpc_plotter')
        self.declare_parameter('output_path', 'mpc_plotter.mp4')
        self.declare_parameter('show_reference_path', True)

        self._pose_times = []
        self._poses = []
        self._chunk_times = []
        self._chunks = []       # world-frame waypoints, one entry per received chunk
        self._cmd_times = []
        self._cmds = []         # (v, omega)
        self._last_seq = None

        self.create_subscription(Odometry, ODOM_TOPIC, self._odom_callback, 10)
        self.create_subscription(ActionChunk, ACTION_CHUNK_TOPIC, self._chunk_callback, 10)
        self.create_subscription(Twist, CMD_VEL_TOPIC, self._cmd_vel_callback, 10)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _odom_callback(self, msg):
        p = msg.pose.pose.position
        self._pose_times.append(self._now())
        self._poses.append(np.array([p.x, p.y, yaw_from_quaternion(msg.pose.pose.orientation)]))

    def _chunk_callback(self, msg):
        if msg.seq_num == self._last_seq or not self._poses:
            return
        self._last_seq = msg.seq_num
        current_pose = self._poses[-1]
        rel_poses = np.array([[p.x, p.y, p.theta] for p in msg.relative_poses])
        self._chunk_times.append(self._now())
        self._chunks.append(relative_poses_to_world(rel_poses, current_pose))

    def _cmd_vel_callback(self, msg):
        self._cmd_times.append(self._now())
        self._cmds.append((msg.linear.x, msg.angular.z))

    def save_animation(self):
        if len(self._poses) < 2:
            self.get_logger().warn('Not enough odometry data collected; skipping animation')
            return

        poses = np.array(self._poses)
        pose_times = np.array(self._pose_times) - self._pose_times[0]
        dt = float(np.median(np.diff(pose_times))) if len(pose_times) > 1 else 1.0 / 15.0

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_aspect('equal')
        ax.grid(True, linewidth=0.3)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        margin = 0.3
        ax.set_xlim(poses[:, 0].min() - margin, poses[:, 0].max() + margin)
        ax.set_ylim(poses[:, 1].min() - margin, poses[:, 1].max() + margin)

        if self.get_parameter('show_reference_path').value:
            path = ChicanePath()
            s_samples = np.linspace(0, path.total_length, 300)
            ref_xy = np.array([path.pose_at_arclength(s)[:2] for s in s_samples])
            ax.plot(ref_xy[:, 0], ref_xy[:, 1], '--', color='gray', label='Reference path')

        ax.plot(poses[0, 0], poses[0, 1], 'go', label='Start')
        actual_line, = ax.plot([], [], '-', color='tab:blue', label='Actual path')
        chunk_scatter, = ax.plot([], [], 'x', color='tab:purple', ms=6, label='Current chunk (world frame)')
        robot_patch = Polygon(robot_triangle(poses[0]), closed=True, color='tab:red', zorder=5)
        ax.add_patch(robot_patch)
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')
        ax.legend(loc='upper right', fontsize=8)

        chunk_times_rel = np.array(self._chunk_times) - self._pose_times[0] if self._chunk_times else np.array([])
        cmd_times_rel = np.array(self._cmd_times) - self._pose_times[0] if self._cmd_times else np.array([])

        def update(frame):
            t = pose_times[frame]
            actual_line.set_data(poses[:frame + 1, 0], poses[:frame + 1, 1])
            robot_patch.set_xy(robot_triangle(poses[frame]))

            if len(chunk_times_rel):
                idx = min(bisect.bisect_right(chunk_times_rel, t) - 1, len(self._chunks) - 1)
                if idx >= 0:
                    chunk_wps = self._chunks[idx]
                    chunk_scatter.set_data(chunk_wps[:, 0], chunk_wps[:, 1])

            cmd_str = ''
            if len(cmd_times_rel):
                idx = min(bisect.bisect_right(cmd_times_rel, t) - 1, len(self._cmds) - 1)
                if idx >= 0:
                    v, omega = self._cmds[idx]
                    cmd_str = f"\nv={v:+.2f} m/s, omega={omega:+.2f} rad/s"
            info_text.set_text(f"t = {t:5.2f} s{cmd_str}")

            return actual_line, chunk_scatter, robot_patch, info_text

        anim = animation.FuncAnimation(fig, update, frames=len(poses), interval=dt * 1000, blit=True)

        output_path = self.get_parameter('output_path').value
        writer = 'ffmpeg' if animation.writers.is_available('ffmpeg') else 'pillow'
        anim.save(output_path, writer=writer, fps=round(1.0 / dt) if dt > 0 else 15)
        self.get_logger().info(f'Saved animation to {output_path}')

    def destroy_node(self):
        self.save_animation()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MpcPlotterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
