#!/usr/bin/env python3
"""Republishes the first DistanceMapSlice recorded in a rosbag on a timer, standing in for a
live nvblox. The map is static and keeps the frame_id it was recorded with.

Does not touch /tf: sim_robot is the live source of odom -> base_link.

Publishes: /nvblox_node/static_map_slice (nvblox_msgs/DistanceMapSlice), at 1 Hz.
Parameters:
  bag_path   sqlite3 rosbag (.db3) holding a /nvblox_node/static_map_slice topic
"""
import sqlite3

import rclpy
from nvblox_msgs.msg import DistanceMapSlice
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.serialization import deserialize_message

ESDF_TOPIC = '/nvblox_node/static_map_slice'
PUBLISH_RATE_HZ = 1.0
DEFAULT_BAG_PATH = (
    '/home/djstrahan/aion-r6-vla-master/aion-r6-ROS/safety_layer_bench_testing/'
    'rosbags/esdf_single_obs/esdf_single_obs_0.db3'
)


class StaticEsdfPublisherNode(Node):
    def __init__(self):
        super().__init__('static_esdf_publisher')
        self.declare_parameter('bag_path', DEFAULT_BAG_PATH)
        bag_path = self.get_parameter('bag_path').value

        self._msg = self._load_first_message(bag_path)
        self._publisher = self.create_publisher(DistanceMapSlice, ESDF_TOPIC, 10)
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish)
        self.get_logger().info(
            f'Publishing static ESDF from {bag_path} '
            f'({self._msg.width}x{self._msg.height} cells @ {self._msg.resolution:.3f}m, '
            f'frame {self._msg.header.frame_id})')

    @staticmethod
    def _load_first_message(bag_path):
        con = sqlite3.connect(bag_path)
        try:
            cur = con.cursor()
            topic_ids = [row[0] for row in cur.execute('SELECT id, name FROM topics')
                         if row[1] == ESDF_TOPIC]
            if not topic_ids:
                raise ValueError(f'{bag_path} has no {ESDF_TOPIC} topic')
            row = cur.execute(
                'SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1',
                (topic_ids[0],)).fetchone()
        finally:
            con.close()
        if row is None:
            raise ValueError(f'{bag_path} has no messages on {ESDF_TOPIC}')
        return deserialize_message(row[0], DistanceMapSlice)

    def _publish(self):
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(self._msg)


def main(args=None):
    rclpy.init(args=args)
    node = StaticEsdfPublisherNode()
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
