#!/usr/bin/env python3
"""Fake VLA for testing mpc_path_follower: publishes action chunks that follow a chicane path.

Assumes the robot starts on the reference path - no correction-from-a-large-offset logic.
That's a real, harder problem (see the MPC_testing history for why it was deliberately
descoped), not something silently handled here.

All 8 relative poses are future deltas - one waypoint_dt through 8*waypoint_dt ahead of the
current pose - never the current pose itself, matching aion_msgs/ActionChunk's wire format.
"""
import math

import numpy as np
import rclpy
from aion_msgs.msg import ActionChunk
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Header

from .chicane_path import ChicanePath

ODOM_TOPIC = '/odom'
ACTION_CHUNK_TOPIC = '/vla/action_chunk'

N_WAYPOINTS = 8            # fixed by aion_msgs/ActionChunk.msg (Pose2D[8] relative_poses)
WAYPOINT_DT = 1.0 / 3.0    # spacing between waypoints within a chunk [s]
CHUNK_RATE_HZ = 8.0        # action-chunk publish rate [Hz]
V_REF = 0.25               # nominal path-following speed [m/s]


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
        self._path = ChicanePath()
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

        # N_WAYPOINTS future path poses, one waypoint_dt through N_WAYPOINTS*waypoint_dt ahead
        # of the current pose - none of them the current pose itself.
        s0 = self._path.nearest_arclength(self._current_pose[:2])
        relative_poses = []
        for i in range(1, N_WAYPOINTS + 1):
            target = self._path.pose_at_arclength(s0 + V_REF * i * WAYPOINT_DT)
            rel_x, rel_y, rel_theta = pose_to_relative(target, self._current_pose)
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
