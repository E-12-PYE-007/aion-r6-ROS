#!/usr/bin/env python3
"""Project YOLO weed detections into the robot odometry/map frame.

Initial detection adapter:
    std_msgs/String JSON on detections_topic.

Accepted JSON shapes:
    {"detections": [{"bbox": [xmin, ymin, xmax, ymax], "class": "weed", "score": 0.9}]}
    [{"bbox": [xmin, ymin, xmax, ymax], "class": "weed", "score": 0.9}]

Once the YOLO node's real ROS message type is known, replace parse_detections()
or add another adapter without changing the projection/geotagging math.
"""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class Detection:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str = 'weed'
    score: float = 0.0

    @property
    def center(self):
        return 0.5 * (self.xmin + self.xmax), 0.5 * (self.ymin + self.ymax)


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def parse_detections(payload):
    data = json.loads(payload)
    if isinstance(data, dict):
        data = data.get('detections', [])

    detections = []
    for item in data:
        bbox = item.get('bbox') or item.get('xyxy')
        if bbox is None and all(k in item for k in ('xmin', 'ymin', 'xmax', 'ymax')):
            bbox = [item['xmin'], item['ymin'], item['xmax'], item['ymax']]
        if bbox is None or len(bbox) != 4:
            continue

        label = str(item.get('class', item.get('label', 'weed')))
        score = float(item.get('score', item.get('confidence', 0.0)))
        detections.append(Detection(
            xmin=float(bbox[0]),
            ymin=float(bbox[1]),
            xmax=float(bbox[2]),
            ymax=float(bbox[3]),
            label=label,
            score=score,
        ))
    return detections


def depth_image_to_array(msg):
    if msg.encoding == '16UC1':
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) * 0.001
    if msg.encoding == '32FC1':
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    raise ValueError(f"Unsupported depth encoding: {msg.encoding}")


class WeedGeotagger(Node):
    def __init__(self):
        super().__init__('weed_geotagger')

        self.detections_topic = self.declare_parameter('detections_topic', '/weed_detections_json').value
        self.depth_topic = self.declare_parameter('depth_topic', '/camera/depth/image_raw').value
        self.camera_info_topic = self.declare_parameter('camera_info_topic', '/camera/color/camera_info').value
        self.odom_topic = self.declare_parameter('odom_topic', '/odometry/filtered').value
        self.output_csv = self.declare_parameter('output_csv', '').value
        self.target_label = self.declare_parameter('target_label', 'weed').value
        self.min_score = float(self.declare_parameter('min_score', 0.0).value)
        self.depth_window_px = int(self.declare_parameter('depth_window_px', 5).value)
        self.max_depth_m = float(self.declare_parameter('max_depth_m', 8.0).value)

        self.camera_x = float(self.declare_parameter('camera_x', 0.2).value)
        self.camera_y = float(self.declare_parameter('camera_y', 0.0).value)
        self.camera_z = float(self.declare_parameter('camera_z', 0.235).value)
        self.camera_yaw = float(self.declare_parameter('camera_yaw', 0.0).value)

        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_odom = None
        self.next_marker_id = 0
        self.csv_file = None
        self.csv_writer = None

        if self.output_csv:
            output_path = Path(self.output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = output_path.open('a', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            if output_path.stat().st_size == 0:
                self.csv_writer.writerow([
                    'stamp_sec', 'label', 'score',
                    'odom_x', 'odom_y', 'odom_yaw',
                    'weed_x', 'weed_y', 'weed_z',
                    'depth_m', 'pixel_u', 'pixel_v',
                ])

        self.create_subscription(String, self.detections_topic, self.detections_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 20)
        self.marker_pub = self.create_publisher(MarkerArray, 'weed_markers', 10)

        self.get_logger().info(
            'Weed geotagger waiting for detections/depth/camera_info/odom: '
            f'{self.detections_topic}, {self.depth_topic}, '
            f'{self.camera_info_topic}, {self.odom_topic}'
        )

    def depth_callback(self, msg):
        try:
            self.latest_depth = (msg, depth_image_to_array(msg))
        except ValueError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=5.0)

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def odom_callback(self, msg):
        self.latest_odom = msg

    def detections_callback(self, msg):
        if self.latest_depth is None or self.latest_camera_info is None or self.latest_odom is None:
            self.get_logger().warn(
                'Missing depth, camera_info, or odom; cannot geotag yet',
                throttle_duration_sec=5.0,
            )
            return

        try:
            detections = parse_detections(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Bad detection JSON: {exc}', throttle_duration_sec=5.0)
            return

        markers = MarkerArray()
        for detection in detections:
            if self.target_label and detection.label != self.target_label:
                continue
            if detection.score < self.min_score:
                continue

            geotag = self.project_detection(detection)
            if geotag is None:
                continue
            weed_x, weed_y, weed_z, depth_m, u, v = geotag

            marker = self.make_marker(weed_x, weed_y, weed_z, detection)
            markers.markers.append(marker)
            self.write_csv(weed_x, weed_y, weed_z, depth_m, u, v, detection)

        if markers.markers:
            self.marker_pub.publish(markers)

    def project_detection(self, detection):
        depth_msg, depth = self.latest_depth
        camera_info = self.latest_camera_info
        odom = self.latest_odom

        u, v = detection.center
        depth_m = self.depth_at(depth, u, v)
        if depth_m is None:
            return None

        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]
        if fx == 0.0 or fy == 0.0:
            self.get_logger().warn('Camera intrinsics are invalid', throttle_duration_sec=5.0)
            return None

        # ROS optical frame convention: x right, y down, z forward.
        x_cam_right = (u - cx) * depth_m / fx
        y_cam_down = (v - cy) * depth_m / fy
        z_cam_forward = depth_m

        # Convert optical point into a planar robot-frame estimate.
        x_robot = self.camera_x + z_cam_forward * math.cos(self.camera_yaw) - x_cam_right * math.sin(self.camera_yaw)
        y_robot = self.camera_y + z_cam_forward * math.sin(self.camera_yaw) + x_cam_right * math.cos(self.camera_yaw)
        z_robot = self.camera_z - y_cam_down

        odom_pose = odom.pose.pose
        odom_x = odom_pose.position.x
        odom_y = odom_pose.position.y
        odom_yaw = yaw_from_quaternion(odom_pose.orientation)

        weed_x = odom_x + x_robot * math.cos(odom_yaw) - y_robot * math.sin(odom_yaw)
        weed_y = odom_y + x_robot * math.sin(odom_yaw) + y_robot * math.cos(odom_yaw)
        return weed_x, weed_y, z_robot, depth_m, u, v

    def depth_at(self, depth, u, v):
        center_u = int(round(u))
        center_v = int(round(v))
        half = max(0, self.depth_window_px // 2)
        u0 = max(0, center_u - half)
        u1 = min(depth.shape[1], center_u + half + 1)
        v0 = max(0, center_v - half)
        v1 = min(depth.shape[0], center_v + half + 1)
        if u0 >= u1 or v0 >= v1:
            return None

        window = depth[v0:v1, u0:u1]
        valid = window[np.isfinite(window)]
        valid = valid[(valid > 0.05) & (valid < self.max_depth_m)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def make_marker(self, x, y, z, detection):
        marker = Marker()
        marker.header.frame_id = 'odom'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'weeds'
        marker.id = self.next_marker_id
        self.next_marker_id += 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(x=float(x), y=float(y), z=float(z))
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 0.1
        marker.color.a = 0.9
        marker.lifetime.sec = 0
        return marker

    def write_csv(self, weed_x, weed_y, weed_z, depth_m, u, v, detection):
        if self.csv_writer is None:
            return
        odom = self.latest_odom
        pose = odom.pose.pose
        self.csv_writer.writerow([
            self.get_clock().now().nanoseconds / 1e9,
            detection.label,
            detection.score,
            pose.position.x,
            pose.position.y,
            yaw_from_quaternion(pose.orientation),
            weed_x,
            weed_y,
            weed_z,
            depth_m,
            u,
            v,
        ])
        self.csv_file.flush()

    def destroy_node(self):
        if self.csv_file is not None:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WeedGeotagger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
