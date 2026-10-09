#!/usr/bin/env python3
"""Publish fake inputs for local weed_geotagger smoke tests."""

import json
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


class FakeWeedInputs(Node):
    def __init__(self):
        super().__init__('fake_weed_inputs')
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odometry/filtered', 10)
        self.detection_pub = self.create_publisher(String, '/weed_detections_json', 10)
        self.timer = self.create_timer(0.5, self.publish_inputs)

    def publish_inputs(self):
        stamp = self.get_clock().now().to_msg()

        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = 'camera_color_optical_frame'
        info.width = 640
        info.height = 480
        info.k = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
        self.info_pub.publish(info)

        depth = np.full((480, 640), 2.0, dtype=np.float32)
        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = 'camera_depth_optical_frame'
        image.height = 480
        image.width = 640
        image.encoding = '32FC1'
        image.is_bigendian = False
        image.step = 640 * 4
        image.data = depth.tobytes()
        self.depth_pub.publish(image)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = 1.0
        odom.pose.pose.position.y = 2.0
        yaw = math.radians(15.0)
        odom.pose.pose.orientation.z = math.sin(0.5 * yaw)
        odom.pose.pose.orientation.w = math.cos(0.5 * yaw)
        self.odom_pub.publish(odom)

        detections = String()
        detections.data = json.dumps({
            'detections': [{
                'bbox': [300, 220, 340, 260],
                'class': 'weed',
                'score': 0.95,
            }]
        })
        self.detection_pub.publish(detections)


def main(args=None):
    rclpy.init(args=args)
    node = FakeWeedInputs()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
