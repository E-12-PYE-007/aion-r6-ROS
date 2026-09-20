"""
Republish the Orbbec colour stream as ImageWithSeqNum on /cam.
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from custom_msgs.msg import ImageWithSeqNum

JPEG_QUALITY = 80


class CamSeqBridgeNode(Node):
    def __init__(self):
        super().__init__("cam_seq_bridge")

        self.jpeg_quality = JPEG_QUALITY

        self.bridge = CvBridge()
        self.seq_num = 0
        self.publisher = self.create_publisher(ImageWithSeqNum, "/cam", 10)

        self.create_subscription(Image, "/camera/color/image_raw", self.republish, qos_profile_sensor_data)

        self.get_logger().info("Stamping /camera/color/image_raw -> /cam with sequence numbers")

    def republish(self, msg: Image):
        """Wrap one raw frame as a numbered JPEG, numbering in arrival order."""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            self.get_logger().warn("JPEG encode failed")
            return

        compressed = CompressedImage()
        compressed.header = msg.header
        compressed.format = "jpeg"
        compressed.data = buf.tobytes()

        wrapped = ImageWithSeqNum()
        wrapped.header = compressed.header
        wrapped.img_seq_num = self.seq_num
        wrapped.img = compressed
        self.publisher.publish(wrapped)
        self.seq_num += 1


def main(args=None):
    rclpy.init(args=args)
    node = CamSeqBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
