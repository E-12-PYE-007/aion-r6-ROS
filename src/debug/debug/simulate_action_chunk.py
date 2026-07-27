#!/usr/bin/env python3
"""Publish deterministic fake VLA pose chunks for simulation testing.

This node stands in for the VLA/action-head output. It publishes
robot-relative pose chunks on the same topic used bythe VLA path,
so the downstream trajectory tracker, wheel reference logic, and
simulator motion path can be tested without changing those nodes.
"""

import math

import rclpy
from aion_msgs.msg import ActionChunk
from geometry_msgs.msg import Pose2D
from rclpy.node import Node

ACTION_TOPIC = '/vla/action_chunk'
PUBLISH_RATE_HZ = 3.0
STEP_DISTANCE = 0.12
TURN_PER_STEP = 0.08


class FakeActionChunkPublisherNode(Node):
    """Generate simple relative pose chunks that look like VLA outputs."""

    def __init__(self):
        super().__init__('action_chunk_simulator')

        self.declare_parameter('pattern', 'straight')
        self.pattern = str(self.get_parameter('pattern').value).lower()

        self.seq_num = 1
        self.publisher = self.create_publisher(ActionChunk, ACTION_TOPIC, 10)
        self.timer = self.create_timer(
            1.0 / PUBLISH_RATE_HZ,
            self.publish_action_chunk,
        )

        self.get_logger().info(
            f'Publishing fake ActionChunk pattern="{self.pattern}" on {ACTION_TOPIC}'
        )

    def build_pose(self, index):
        """Return the index-th robot-relative target pose for the test pattern."""
        distance = STEP_DISTANCE * index

        if self.pattern == 'stop':
            return 0.0, 0.0, 0.0

        if self.pattern == 'left_arc':
            theta = TURN_PER_STEP * index
            return distance * math.cos(0.5 * theta), distance * math.sin(0.5 * theta), theta

        if self.pattern == 'right_arc':
            theta = -TURN_PER_STEP * index
            return distance * math.cos(0.5 * theta), distance * math.sin(0.5 * theta), theta

        if self.pattern == 's_curve':
            theta = TURN_PER_STEP * math.sin(0.75 * index)
            y = 0.25 * math.sin(0.45 * index)
            return distance, y, theta

        # The default "straight" chunk moves forward in the robot x direction,
        # which matches the relative-pose convention used by the odom tracker.
        return distance, 0.0, 0.0

    def publish_action_chunk(self):
        """Publish one fake chunk in the same message shape as the real model."""
        msg = ActionChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.seq_num = self.seq_num

        for index in range(1, len(msg.relative_poses) + 1):
            x, y, theta = self.build_pose(index)
            pose = Pose2D()
            pose.x = float(x)
            pose.y = float(y)
            pose.theta = float(theta)
            msg.relative_poses[index - 1] = pose
            
        self.publisher.publish(msg)
        self.seq_num += 1


def main(args=None):
    rclpy.init(args=args)
    node = FakeActionChunkPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
