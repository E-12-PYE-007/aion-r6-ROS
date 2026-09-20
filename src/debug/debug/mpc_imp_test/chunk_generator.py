#!/usr/bin/env python3
"""Fake VLA for testing mpc_path_follower: publishes action chunks that point from the
robot's current position back toward a straight reference line.

Each chunk's waypoints are spaced by fixed DISTANCE (WAYPOINT_SPACING_M), not time - they
run from the robot's current position to a point on the reference line exactly LOOKAHEAD_M
away (by straight-line norm, picked ahead along the line's own direction, not the closest
point). WAYPOINT_SPACING_M * N_WAYPOINTS == LOOKAHEAD_M by construction (0.1m * 8 = 0.8m),
so the final waypoint always lands exactly on the reference line when one exists at that
distance (see StraightPath.lookahead_point for the fallback when the robot has drifted
further than LOOKAHEAD_M off the line).

All 8 relative poses are future deltas - never the current pose itself, matching
aion_msgs/ActionChunk's wire format.
"""
import math

import numpy as np
import rclpy
from aion_msgs.msg import ActionChunk
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Header

from .straight_path import StraightPath, PATH_ORIGIN, PATH_HEADING

ODOM_TOPIC = '/odom'
ACTION_CHUNK_TOPIC = '/vla/action_chunk'

N_WAYPOINTS = 8            # fixed by aion_msgs/ActionChunk.msg (Pose2D[8] relative_poses)
WAYPOINT_SPACING_M = 0.1   # distance between consecutive chunk waypoints
LOOKAHEAD_M = N_WAYPOINTS * WAYPOINT_SPACING_M  # 0.8m - point on the line the chunk aims at
CHUNK_RATE_HZ = 8.0


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pose_to_relative(pose, anchor_pose):
    """World-frame pose -> pose relative to anchor_pose (anchor's own local frame)."""
    dx, dy = pose[0] - anchor_pose[0], pose[1] - anchor_pose[1]
    c, s = math.cos(anchor_pose[2]), math.sin(anchor_pose[2])
    rel_x = dx * c + dy * s
    rel_y = -dx * s + dy * c
    rel_theta = wrap_to_pi(pose[2] - anchor_pose[2])
    return rel_x, rel_y, rel_theta


class ChunkGeneratorNode(Node):
    def __init__(self):
        super().__init__('chunk_generator')

        self._current_pose = None
        self._path = StraightPath(PATH_ORIGIN, PATH_HEADING)
        self._seq_num = 0

        self.create_subscription(Odometry, ODOM_TOPIC, self._odom_callback, 10)
        self._publisher = self.create_publisher(ActionChunk, ACTION_CHUNK_TOPIC, 10)
        self.create_timer(1.0 / CHUNK_RATE_HZ, self._publish_chunk)

    def _odom_callback(self, msg):
        p = msg.pose.pose.position
        self._current_pose = np.array([p.x, p.y, yaw_from_quaternion(msg.pose.pose.orientation)])

    def _publish_chunk(self):
        if self._current_pose is None:
            self.get_logger().warn('No odometry yet, skipping chunk publish')
            return

        target_xy = self._path.lookahead_point(self._current_pose[:2], LOOKAHEAD_M)
        seg = target_xy - self._current_pose[:2]
        seg_dist = np.linalg.norm(seg)
        seg_dir = seg / seg_dist if seg_dist > 1e-9 else np.array([1.0, 0.0])
        heading = math.atan2(seg_dir[1], seg_dir[0])

        relative_poses = []
        for i in range(1, N_WAYPOINTS + 1):
            world_xy = self._current_pose[:2] + i * WAYPOINT_SPACING_M * seg_dir
            world_pose = np.array([world_xy[0], world_xy[1], heading])
            rel_x, rel_y, rel_theta = pose_to_relative(world_pose, self._current_pose)
            relative_poses.append(Pose2D(x=rel_x, y=rel_y, theta=rel_theta))

        msg = ActionChunk()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.seq_num = self._seq_num
        msg.relative_poses = relative_poses
        self._publisher.publish(msg)
        self._seq_num += 1


def main(args=None):
    rclpy.init(args=args)
    node = ChunkGeneratorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
