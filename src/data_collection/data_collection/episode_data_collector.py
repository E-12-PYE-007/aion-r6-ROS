#!/usr/bin/env python3
"""
    Data collection node for Aion R6. Collects a continuous stream of image
    and current pose at a rate of 3Hz, gated into named episodes that are
    started/stopped via service calls rather than the node's own lifetime.
"""

import re
from datetime import datetime
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from pathlib import Path
import cv2
from cv_bridge import CvBridge
import json
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from aion_msgs.srv import StartEpisode
import math

# Define constants

DT = 1/3 # Sample rate
EPISODE_NAME_PATTERN = re.compile(r'^[A-Za-z0-9_-]+$')


class ImageEncodeError(Exception):
    pass


class EpisodeDataCollectionNode(Node):
    def __init__(self):
        super().__init__('episode_data_collector')

        self.bridge = CvBridge()

        self.previous_img_time = 0
        self.current_pose = None
        self.current_vel = None

        self.episode_dir = None
        self.img_dir = None
        self.poses_path = None
        self.episode_name = None
        self.episode_prompt = None
        self.frame_count = 0

        self.declare_parameter('base_dir', Parameter.Type.STRING)
        base_dir = self.get_parameter('base_dir').get_parameter_value().string_value
        if not base_dir:
            raise RuntimeError(
                "base_dir parameter is required, e.g. --ros-args -p base_dir:=/path/to/trajectories"
            )
        self.base_dir = Path(base_dir)

        cam_topic = self.declare_parameter('cam_topic', '/camera/color/image_raw').value
        odom_topic = self.declare_parameter('odom_topic', '/odometry/filtered').value

        self.cam_subscriber = self.create_subscription(
            Image,
            cam_topic,
            self.cam_callback,
            qos_profile_sensor_data,
        )

        self.odom_subscriber = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.start_episode_srv = self.create_service(
            StartEpisode, '~/start_episode', self.start_episode_cb
        )
        self.stop_episode_srv = self.create_service(
            Trigger, '~/stop_episode', self.stop_episode_cb
        )

    def start_episode_cb(self, request, response):
        if self.episode_dir is not None:
            response.success = False
            response.message = f"Already recording episode '{self.episode_name}'"
            response.episode_dir = str(self.episode_dir)
            return response

        name = request.name.strip() or datetime.now().strftime('%Y%m%d_%H%M%S')
        if not EPISODE_NAME_PATTERN.match(name):
            response.success = False
            response.message = f"Invalid episode name '{name}': only letters, digits, '_' and '-' allowed"
            response.episode_dir = ''
            return response

        episode_dir = self.base_dir / name
        if episode_dir.exists():
            response.success = False
            response.message = f"Episode directory already exists: {episode_dir}"
            response.episode_dir = ''
            return response

        img_dir = episode_dir / "img"
        img_dir.mkdir(parents=True, exist_ok=True)

        self.episode_dir = episode_dir
        self.img_dir = img_dir
        self.poses_path = episode_dir / "poses.jsonl"
        self.episode_name = name
        self.episode_prompt = request.prompt.strip()
        self.frame_count = 0
        self.previous_img_time = 0

        response.success = True
        response.message = f"Started episode '{name}'"
        response.episode_dir = str(episode_dir)
        return response

    def stop_episode_cb(self, request, response):
        if self.episode_dir is None:
            response.success = False
            response.message = "No episode in progress"
            return response

        response.success = True
        response.message = f"Stopped episode '{self.episode_name}' ({self.frame_count} frames)"
        if self.frame_count == 0:
            self.get_logger().error(
                f"Episode '{self.episode_name}' stopped with 0 frames logged -- "
                "check that both cam_topic and odom_topic are actually publishing"
            )

        self.episode_dir = None
        self.img_dir = None
        self.poses_path = None
        self.episode_name = None
        self.episode_prompt = None
        self.frame_count = 0
        return response

    def cam_callback(self, msg):
        if self.episode_dir is None:
            return # No episode in progress

        if self.current_pose is None:
            self.get_logger().warn('No starting pose available -- no odometry received yet', throttle_duration_sec=5.0)
            return # Cannot start logging without a starting pose

        img_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 # image capture time in seconds

        if img_time - self.previous_img_time > DT:
            try:
                self.log_img_pose_pair(msg)
                self.previous_img_time = img_time
            except ImageEncodeError:
                self.get_logger().warn('Failed to log image/pose pair, will retry next frame')

        return

    def odom_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.yaw_from_quat(msg.pose.pose.orientation)

        #self.get_logger().info(f'Received odom msg on {odom_topic}')

        self.current_pose = (
            t,
            x,
            y,
            theta
        )

        tw = msg.twist.twist

        self.current_vel = (
            tw.linear.x,
            tw.angular.z
        )

    @staticmethod
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

    def log_img_pose_pair(self, msg):

        img_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        ok, img = self.encode_img(msg)
        if not ok:
            raise ImageEncodeError('Failed to JPEG-encode camera frame')

        img_id = f"{int(img_time*1000)}"
        image_path = self.img_dir / f"{img_id}.jpg"
        image_path.write_bytes(img.tobytes())

        record = {
            "episode": self.episode_name,
            "prompt": self.episode_prompt,
            "image": image_path.name,
            "img_time": img_time,
            "pose": self.current_pose,
            "velocity": self.current_vel
        }
        with open(self.poses_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        self.frame_count += 1
        return

def main(args=None):
    rclpy.init(args=args)
    node = EpisodeDataCollectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
