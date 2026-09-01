#!/usr/bin/env python3
"""Bridge node between the Orbbec Gemini 336 camera and Isaac ROS Visual SLAM.

Converts the Gemini's RGB color stream to mono8 (cuVSLAM expects grayscale
input on visual_slam/image_0) and relays the color camera_info plus the
depth image/camera_info onto the visual_slam/* topics untouched. All topic
names are configurable via parameters in case the stock Isaac ROS launch
file's remap args change between releases.
"""

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class CamSlamBridgeNode(Node):
    def __init__(self):
        super().__init__('cam_slam_bridge')

        self.declare_parameter('input_image_topic', '/camera/color/image_raw')
        self.declare_parameter('input_camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('input_depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('input_depth_camera_info_topic', '/camera/depth/camera_info')

        self.declare_parameter('output_image_topic', '/visual_slam/image_0')
        self.declare_parameter('output_camera_info_topic', '/visual_slam/camera_info_0')
        self.declare_parameter('output_depth_topic', '/visual_slam/depth_0')
        self.declare_parameter('output_depth_camera_info_topic', '/visual_slam/depth_camera_info_0')

        input_image_topic = self.get_parameter('input_image_topic').value
        input_camera_info_topic = self.get_parameter('input_camera_info_topic').value
        input_depth_topic = self.get_parameter('input_depth_topic').value
        input_depth_camera_info_topic = self.get_parameter('input_depth_camera_info_topic').value

        output_image_topic = self.get_parameter('output_image_topic').value
        output_camera_info_topic = self.get_parameter('output_camera_info_topic').value
        output_depth_topic = self.get_parameter('output_depth_topic').value
        output_depth_camera_info_topic = self.get_parameter('output_depth_camera_info_topic').value

        self.bridge = CvBridge()

        self.mono_image_pub = self.create_publisher(Image, output_image_topic, qos_profile_sensor_data)
        self.camera_info_pub = self.create_publisher(CameraInfo, output_camera_info_topic, qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(Image, output_depth_topic, qos_profile_sensor_data)
        self.depth_camera_info_pub = self.create_publisher(CameraInfo, output_depth_camera_info_topic, qos_profile_sensor_data)

        self.create_subscription(Image, input_image_topic, self.image_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, input_camera_info_topic, self.camera_info_pub.publish, qos_profile_sensor_data)
        self.create_subscription(Image, input_depth_topic, self.depth_pub.publish, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, input_depth_camera_info_topic, self.depth_camera_info_pub.publish, qos_profile_sensor_data)

        self.get_logger().info(
            f'Bridging {input_image_topic} (rgb8 -> mono8) to {output_image_topic}; '
            f'relaying {input_camera_info_topic} -> {output_camera_info_topic}, '
            f'{input_depth_topic} -> {output_depth_topic}, '
            f'{input_depth_camera_info_topic} -> {output_depth_camera_info_topic}'
        )

    def image_callback(self, msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except CvBridgeError as exc:
            self.get_logger().error(f'Failed to convert incoming image to rgb8: {exc}')
            return

        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        gray_msg = self.bridge.cv2_to_imgmsg(gray_image, encoding='mono8')
        gray_msg.header = msg.header
        self.mono_image_pub.publish(gray_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CamSlamBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
