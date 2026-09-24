#!/usr/bin/env python3
"""Best-effort local decoder for viewing the robot camera on the workstation.

Subscribes to the JPEG-compressed color stream over the network with BEST_EFFORT
QoS and a shallow queue, so the Jetson never has to buffer or retransmit frames
for this viewer. That avoids the reliable-QoS backlog (retransmits + publisher
back-pressure) that can drag down the camera pipeline's frame rate on a lossy
link. Frames are decoded and re-published as a plain sensor_msgs/Image locally,
also BEST_EFFORT, so RViz/rqt view an ordinary raw topic with no image_transport
involved.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge


class ImageDecoderNode(Node):
    def __init__(self):
        super().__init__('image_decoder')

        self.declare_parameter('compressed_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('output_topic', '/workstation/image_raw')
        in_topic = self.get_parameter('compressed_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        # Best-effort + depth 1: keep only the freshest frame and drop the rest
        # under load, instead of retransmitting. Retransmits are exactly what
        # create Jetson-side backlog, so we opt out on both ends.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, out_topic, qos)
        self.sub = self.create_subscription(CompressedImage, in_topic, self.callback, qos)
        self.get_logger().info(
            f"Decoding {in_topic} (best-effort) -> {out_topic} (best-effort)"
        )

    def callback(self, msg):
        try:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            out = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"Failed to decode frame: {e}")
            return
        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ImageDecoderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
