#!/usr/bin/env python3
"""Obstacle-blind stand-in for the VLA: publishes action chunks that point at a fixed goal.

The goal is latched from the first /odometry/filtered message: `goal_distance` metres straight ahead of the
robot's pose at that moment. From then on, every chunk is 8 waypoints spaced 0.1 m along the
straight line from the robot's current pose to the goal, each with the heading of that line,
expressed relative to the current pose (future deltas only, never the current pose itself). A
waypoint never goes past the goal, so near the goal the waypoints bunch up on it.

Subscribes: /odometry/filtered (nav_msgs/Odometry) - pose of base_link in the odom frame.
Publishes:  /vla/action_chunk (aion_msgs/ActionChunk), at 8 Hz.
Parameters:
  goal_distance   distance from the initial pose to the goal, along the initial heading [m]
"""
import math

import rclpy
from aion_msgs.msg import ActionChunk
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

ODOM_TOPIC = '/odometry/filtered'
ACTION_CHUNK_TOPIC = '/vla/action_chunk'
CHUNK_RATE_HZ = 8.0
N_WAYPOINTS = 8            # fixed by aion_msgs/ActionChunk.msg (Pose2D[8] relative_poses)
WAYPOINT_SPACING_M = 0.1   # distance between consecutive waypoints
DEFAULT_GOAL_DISTANCE_M = 6.0


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def chunk_toward_goal(pose, goal):
    """The 8 waypoints (x, y, theta), relative to `pose` = (x, y, theta), along the straight
    line from `pose` to `goal` = (x, y). A robot already at the goal keeps its own heading."""
    x, y, theta = pose
    to_goal_x, to_goal_y = goal[0] - x, goal[1] - y
    distance = math.hypot(to_goal_x, to_goal_y)
    if distance > 1e-9:
        dir_x, dir_y = to_goal_x / distance, to_goal_y / distance
        heading = math.atan2(dir_y, dir_x)
    else:
        dir_x, dir_y = math.cos(theta), math.sin(theta)
        heading = theta

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    relative_heading = wrap_to_pi(heading - theta)
    waypoints = []
    for i in range(1, N_WAYPOINTS + 1):
        along = min(i * WAYPOINT_SPACING_M, distance)
        dx, dy = along * dir_x, along * dir_y
        waypoints.append((dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t, relative_heading))
    return waypoints


class ChunkGeneratorNode(Node):
    def __init__(self):
        super().__init__('chunk_generator')
        self.declare_parameter('goal_distance', DEFAULT_GOAL_DISTANCE_M)
        self._goal_distance = float(self.get_parameter('goal_distance').value)

        self._pose = None     # (x, y, theta), latest odometry sample
        self._goal = None     # (x, y) in the odom frame, latched from the first odometry sample
        self._seq_num = 0

        self.create_subscription(Odometry, ODOM_TOPIC, self._odom_callback, 10)
        self._publisher = self.create_publisher(ActionChunk, ACTION_CHUNK_TOPIC, 10)
        self.create_timer(1.0 / CHUNK_RATE_HZ, self._publish_chunk)

    def _odom_callback(self, msg):
        p = msg.pose.pose.position
        self._pose = (p.x, p.y, yaw_from_quaternion(msg.pose.pose.orientation))
        if self._goal is None:
            x, y, theta = self._pose
            self._goal = (x + self._goal_distance * math.cos(theta),
                          y + self._goal_distance * math.sin(theta))
            self.get_logger().info(
                f'Goal latched at ({self._goal[0]:.3f}, {self._goal[1]:.3f}), '
                f'{self._goal_distance} m ahead of ({x:.3f}, {y:.3f}, {math.degrees(theta):.1f} deg)')

    def _publish_chunk(self):
        if self._pose is None:
            self.get_logger().warn('No odometry yet, not publishing', throttle_duration_sec=5.0)
            return

        msg = ActionChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.seq_num = self._seq_num
        msg.relative_poses = [Pose2D(x=x, y=y, theta=theta)
                              for x, y, theta in chunk_toward_goal(self._pose, self._goal)]
        self._publisher.publish(msg)
        self._seq_num += 1


def main(args=None):
    rclpy.init(args=args)
    node = ChunkGeneratorNode()
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
