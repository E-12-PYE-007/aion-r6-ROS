#!/usr/bin/env python3
"""Display the rover body, heading and a bounded route from EKF odometry."""

import math
from collections import deque
from copy import deepcopy

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from visualization_msgs.msg import Marker, MarkerArray


class RoverVisualiser(Node):
    def __init__(self):
        super().__init__('rover_visualiser')
        odom_topic = self.declare_parameter('odom_topic', '/odometry/filtered').value
        self.length = float(self.declare_parameter('rover_length_m', 0.65).value)
        self.width = float(self.declare_parameter('rover_width_m', 0.50).value)
        self.height = float(self.declare_parameter('rover_height_m', 0.15).value)
        self.path_spacing = float(self.declare_parameter('path_spacing_m', 0.03).value)
        max_points = int(self.declare_parameter('max_path_points', 10000).value)
        if min(self.length, self.width, self.height) <= 0 or max_points < 1 or self.path_spacing <= 0:
            raise ValueError('Rover dimensions, path spacing and max_path_points must be positive')

        self.latest_odom = None
        self.path_poses = deque(maxlen=max_points)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.rover_pub = self.create_publisher(MarkerArray, 'rover_markers', qos)
        self.path_pub = self.create_publisher(Path, 'rover_path', qos)
        self.create_subscription(Odometry, odom_topic, self.odom_callback, qos_profile_sensor_data)
        self.create_timer(0.1, self.publish_visualisation)

    def odom_callback(self, msg):
        if not msg.header.frame_id:
            self.get_logger().warn('Odometry has no frame_id; cannot display it', throttle_duration_sec=5.0)
            return
        if self.latest_odom is not None and msg.header.frame_id != self.latest_odom.header.frame_id:
            self.path_poses.clear()
        self.latest_odom = msg
        position = msg.pose.pose.position
        if self.path_poses:
            previous = self.path_poses[-1].pose.position
            if math.hypot(position.x - previous.x, position.y - previous.y) < self.path_spacing:
                return
        pose = PoseStamped()
        pose.header = deepcopy(msg.header)
        pose.pose = deepcopy(msg.pose.pose)
        self.path_poses.append(pose)

    def make_rover_marker(self, msg, marker_id, marker_type):
        marker = Marker()
        marker.header = deepcopy(msg.header)
        marker.ns = 'rover'
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose = deepcopy(msg.pose.pose)
        marker.pose.position.z += self.height / 2.0
        marker.color.a = 1.0
        return marker

    def publish_visualisation(self):
        msg = self.latest_odom
        if msg is None:
            return

        body = self.make_rover_marker(msg, 0, Marker.CUBE)
        body.scale.x, body.scale.y, body.scale.z = self.length, self.width, self.height
        body.color.r, body.color.g, body.color.b = 0.85, 0.88, 0.92

        heading = self.make_rover_marker(msg, 1, Marker.ARROW)
        heading.pose.position.z += self.height / 2.0 + 0.04
        heading.scale.x, heading.scale.y, heading.scale.z = self.length * 0.8, 0.07, 0.10
        heading.color.r, heading.color.g, heading.color.b = 1.0, 0.30, 0.25

        markers = MarkerArray()
        markers.markers = [body, heading]
        self.rover_pub.publish(markers)

        path = Path()
        path.header = deepcopy(msg.header)
        path.poses = list(self.path_poses)
        self.path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = RoverVisualiser()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
