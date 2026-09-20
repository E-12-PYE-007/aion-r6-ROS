#!/usr/bin/env python3
"""Republishes one recorded DistanceMapSlice from a rosbag on a timer, standing in for a
live nvblox for testing mpc_path_follower against a real (but static) obstacle map.

Does not touch /tf: sim_robot is the live source of odom->base_link, and replaying the
bag's own recorded tf history alongside it would fight that. Delete once a live nvblox
(or a full simulator) exists.
"""
import sqlite3

import rclpy
from nvblox_msgs.msg import DistanceMapSlice
from rclpy.node import Node
from rclpy.serialization import deserialize_message

ESDF_TOPIC = '/nvblox_node/static_map_slice'
PUBLISH_RATE_HZ = 1.0


class StaticEsdfPublisherNode(Node):
    def __init__(self):
        super().__init__('static_esdf_publisher')
        self.declare_parameter('bag_path', '')
        bag_path = self.get_parameter('bag_path').value
        if not bag_path:
            raise ValueError('static_esdf_publisher requires the bag_path parameter')

        self._msg = self._load_first_message(bag_path)
        self._publisher = self.create_publisher(DistanceMapSlice, ESDF_TOPIC, 10)
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish)
        self.get_logger().info(f'Publishing static ESDF from {bag_path}')

    def _load_first_message(self, bag_path):
        con = sqlite3.connect(bag_path)
        cur = con.cursor()
        topic_id = [t[0] for t in cur.execute('SELECT id, name, type FROM topics').fetchall()
                    if t[1] == ESDF_TOPIC][0]
        row = cur.execute(
            'SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1',
            (topic_id,)).fetchone()
        return deserialize_message(row[0], DistanceMapSlice)

    def _publish(self):
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(self._msg)


def main(args=None):
    rclpy.init(args=args)
    node = StaticEsdfPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
