#!/usr/bin/env python3
"""
    Data collection node for Aion R6. Collects a continuous stream of image
    and current pose at a rate of 3Hz
"""

from datetime import datetime
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
import px4_msgs.msg
from pathlib import Path
import cv2
from cv_bridge import CvBridge
import json
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs import msgs.Odometry
import math

# Define constants

DT = 1/3 # Sample rate


class ImageEncodeError(Exception):
    pass


class StreamDataCollectionNode(Node):
    def __init__(self):
        super().__init__('stream_data_collector')

        self.bridge = CvBridge()

        self.previous_img_time = 0
        self.current_pose = None
        self.current_vel = None

        self.declare_parameter('base_dir', Parameter.Type.STRING)
        base_dir = self.get_parameter('base_dir').get_parameter_value().string_value
        if not base_dir:
            raise RuntimeError(
                "base_dir parameter is required, e.g. --ros-args -p base_dir:=/path/to/trajectories"
            )

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.traj_dir = Path(base_dir) / f"{stamp}_{self.get_name()}"
        self.img_dir = self.traj_dir / "img"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.poses_path = self.traj_dir / "poses.jsonl"

        self.cam_subscriber = self.create_subscription(
            Image,
            "/vla/cam1", # placeholder topic name
            self.cam_callback,
            qos_profile_sensor_data,
        )

        self.odom_subscriber = self.create_subscription(
            nav_msgs.msg.Odometry,
            "/sim_odom",
            self.odom_callback,
            qos_profile_sensor_data
        )

    def cam_callback(self, msg):
        img_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 # image capture time in seconds

        if self.current_pose is None:
            self.get_logger().warn('No starting pose available')
            return # Cannot start logging chunks without a starting pose

        if img_time - self.previous_img_time > DT:
            try:
                self.log_img_pose_pair(msg)
                self.previous_img_time = img_time
            except ImageEncodeError:
                self.get_logger().warn('Failed to log image/pose pair, will retry next frame')

        return

    def odom_callback(self, msg):
        t = Time.from_msg(msg.header.stamp)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        self.current_pose = (
            Time.from_msg(msg.header.stamp),
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw_from_quat(msg.pose.pose.orientation),
        )

        tw = msg.twist.twist

        self.current_vel = (
            tw.linear.x,
            tw.angular.z
        )

    def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

    def encode_img(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        ok, encoded = cv2.imencode(
            '.jpg',
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )
        return ok, encoded

    def log_img_pose_pair(self,msg):

        img_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        ok, img = self.encode_img(msg)
        if not ok:
            raise ImageEncodeError('Failed to JPEG-encode camera frame')

        img_id = f"{int(img_time)}"
        image_path = self.img_dir / f"{img_id}.jpg"
        image_path.write_bytes(img.tobytes())

        record = {
            "image": image_path.name,
            "img_time": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            "pose": self.current_pose
            "vel": self.current_vel
        }
        with open(self.poses_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return

def main(args=None):
    rclpy.init(args=args)
    node = StreamDataCollectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
