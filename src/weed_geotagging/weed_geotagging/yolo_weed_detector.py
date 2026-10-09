#!/usr/bin/env python3
"""Run a YOLO weed detector on camera images and publish JSON detections."""

import json
from pathlib import Path

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class YoloWeedDetector(Node):
    def __init__(self):
        super().__init__('yolo_weed_detector')

        self.model_path = self.declare_parameter('model_path', '').value
        self.image_topic = self.declare_parameter('image_topic', '/camera/color/image_raw').value
        self.detections_topic = self.declare_parameter('detections_topic', '/weed_detections_json').value
        self.confidence = float(self.declare_parameter('confidence', 0.25).value)
        self.iou = float(self.declare_parameter('iou', 0.45).value)
        self.device = self.declare_parameter('device', '').value
        self.target_classes = self.parse_target_classes(
            self.declare_parameter('target_classes', '').value
        )
        self.publish_annotated = bool(self.declare_parameter('publish_annotated', True).value)
        self.annotated_topic = self.declare_parameter('annotated_topic', '/weed_detections/image').value

        if not self.model_path:
            raise RuntimeError('model_path parameter is required, e.g. model_path:=/path/to/best.pt')
        if not Path(self.model_path).is_file():
            raise RuntimeError(f'YOLO model file does not exist: {self.model_path}')

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                'ultralytics is required for yolo_weed_detector. '
                'Install it in the Jetson Python environment, e.g. pip install ultralytics'
            ) from exc

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)
        self.detections_pub = self.create_publisher(String, self.detections_topic, 10)
        self.annotated_pub = (
            self.create_publisher(Image, self.annotated_topic, 10)
            if self.publish_annotated else None
        )
        self.create_subscription(Image, self.image_topic, self.image_callback, 10)

        self.get_logger().info(
            f'Loaded YOLO model {self.model_path}; subscribing {self.image_topic}, '
            f'publishing {self.detections_topic}'
        )

    @staticmethod
    def parse_target_classes(value):
        if not value:
            return set()
        return {item.strip() for item in str(value).split(',') if item.strip()}

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert image: {exc}', throttle_duration_sec=5.0)
            return

        kwargs = {
            'conf': self.confidence,
            'iou': self.iou,
            'verbose': False,
        }
        if self.device:
            kwargs['device'] = self.device

        try:
            results = self.model.predict(image, **kwargs)
        except Exception as exc:
            self.get_logger().warn(f'YOLO inference failed: {exc}', throttle_duration_sec=5.0)
            return

        detections = []
        if results:
            result = results[0]
            names = result.names or {}
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = str(names.get(class_id, class_id))
                if self.target_classes and label not in self.target_classes:
                    continue

                xyxy = box.xyxy[0].tolist()
                score = float(box.conf[0].item())
                detections.append({
                    'bbox': [float(value) for value in xyxy],
                    'class': label,
                    'class_id': class_id,
                    'score': score,
                    'stamp': {
                        'sec': int(msg.header.stamp.sec),
                        'nanosec': int(msg.header.stamp.nanosec),
                    },
                })

            if self.annotated_pub is not None:
                annotated = result.plot()
                out = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                out.header = msg.header
                self.annotated_pub.publish(out)

        payload = String()
        payload.data = json.dumps({'detections': detections})
        self.detections_pub.publish(payload)


def main(args=None):
    rclpy.init(args=args)
    node = YoloWeedDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
