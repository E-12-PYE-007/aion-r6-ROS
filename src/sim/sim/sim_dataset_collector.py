#!/usr/bin/env python3
"""
    Isaac Sim data collection node. Collects a continuous stream of image,
    odometry, velocity command, and action-chunk data using the same JSONL
    logging style as stream_data_collector.py.
"""

from datetime import datetime
from pathlib import Path
import json
import math

from aion_msgs.msg import ActionChunk
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
import yaml


DEFAULT_DT = 1 / 3


class ImageEncodeError(Exception):
    pass


def stamp_to_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def yaw_from_quaternion(quat):
    x = quat.x
    y = quat.y
    z = quat.z
    w = quat.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def transform_point(point, flip_isaac_y):
    x = float(point[0])
    y = float(point[1])
    if flip_isaac_y:
        y = -y
    return x, y


def yaw_value(raw_yaw, flip_isaac_y):
    yaw = float(raw_yaw or 0.0)
    if abs(yaw) > math.tau:
        yaw = math.radians(yaw)
    if flip_isaac_y:
        yaw = -yaw
    return wrap_to_pi(yaw)


def local_odom_to_world(local_pose, world_start_pose):
    _, local_x, local_y, local_yaw = local_pose
    start_x, start_y, start_yaw = world_start_pose
    cos_yaw = math.cos(start_yaw)
    sin_yaw = math.sin(start_yaw)
    world_x = start_x + cos_yaw * local_x - sin_yaw * local_y
    world_y = start_y + sin_yaw * local_x + cos_yaw * local_y
    return world_x, world_y, wrap_to_pi(start_yaw + local_yaw)


def load_world_start_pose(task_spec_path, task_id, flip_isaac_y):
    if not task_spec_path:
        return 0.0, 0.0, 0.0
    with Path(task_spec_path).open("r", encoding="utf-8") as f:
        task_spec = yaml.safe_load(f)
    scene_path = Path(task_spec["scene"]["source_yaml"])
    with scene_path.open("r", encoding="utf-8") as f:
        scene = yaml.safe_load(f)
    task = next((candidate for candidate in task_spec.get("tasks", []) if candidate.get("task_id") == task_id), {})
    starts = scene.get("rover_poses")
    start_name = task.get("start_pose")
    if isinstance(starts, dict) and isinstance(start_name, str) and start_name in starts:
        rover_pose = starts[start_name]
    else:
        rover_pose = scene.get("rover_pose", {})
    start_x, start_y = transform_point(rover_pose.get("position", [0.0, 0.0, 0.0]), flip_isaac_y)
    return start_x, start_y, yaw_value(rover_pose.get("yaw", 0.0), flip_isaac_y)


def parse_optional_json(value, field_name):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{field_name} must be valid JSON: {exc}") from exc


class SimDatasetCollectorNode(Node):
    def __init__(self):
        super().__init__('sim_dataset_collector')

        self.bridge = CvBridge()

        self.previous_img_time = 0
        self.current_pose = None
        self.current_local_pose = None
        self.current_velocity = None
        self.current_cmd_vel = None
        self.current_action_chunk = None

        self.declare_parameter('base_dir', Parameter.Type.STRING)
        self.declare_parameter('dataset_name', 'sim_fenceline')
        self.declare_parameter('trajectory_name', '')
        self.declare_parameter('task_spec', '')
        self.declare_parameter('task_id', '')
        self.declare_parameter('variant_id', 'nominal')
        self.declare_parameter('variant_type', 'nominal')
        self.declare_parameter('recovery_case', '')
        self.declare_parameter('language_instruction', 'Follow the fence.')
        self.declare_parameter('structured_task_json', '')
        self.declare_parameter('planner_settings_json', '')
        self.declare_parameter('speed_profile_json', '')
        self.declare_parameter('camera_topic', '/vla/cam')
        self.declare_parameter('odom_topic', '/sim_odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('action_chunk_topic', '/vla/action_chunk')
        self.declare_parameter('sample_frequency_hz', 3.0)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('flip_isaac_y', False)
        self.declare_parameter('flip_scene_y', False)
        self.declare_parameter('flip_runtime_odom_y', False)
        self.declare_parameter('flip_runtime_odom_yaw', True)

        base_dir = self.get_parameter('base_dir').get_parameter_value().string_value
        if not base_dir:
            raise RuntimeError(
                "base_dir parameter is required, e.g. --ros-args -p base_dir:=/path/to/trajectories"
            )

        self.dataset_name = self.get_parameter('dataset_name').get_parameter_value().string_value
        requested_name = self.get_parameter('trajectory_name').get_parameter_value().string_value
        self.task_spec_path = self.get_parameter('task_spec').get_parameter_value().string_value
        self.task_id = self.get_parameter('task_id').get_parameter_value().string_value
        self.variant_id = self.get_parameter('variant_id').get_parameter_value().string_value
        self.variant_type = self.get_parameter('variant_type').get_parameter_value().string_value
        self.recovery_case = self.get_parameter('recovery_case').get_parameter_value().string_value
        self.language_instruction = self.get_parameter('language_instruction').get_parameter_value().string_value
        self.structured_task_json = self.get_parameter('structured_task_json').get_parameter_value().string_value
        self.planner_settings_json = self.get_parameter('planner_settings_json').get_parameter_value().string_value
        self.speed_profile_json = self.get_parameter('speed_profile_json').get_parameter_value().string_value
        self.jpeg_quality = self.get_parameter('jpeg_quality').get_parameter_value().integer_value
        self.flip_isaac_y = self.get_parameter('flip_isaac_y').get_parameter_value().bool_value
        self.flip_scene_y = self.get_parameter('flip_scene_y').get_parameter_value().bool_value
        self.flip_runtime_odom_y = self.get_parameter('flip_runtime_odom_y').get_parameter_value().bool_value
        self.flip_runtime_odom_yaw = self.get_parameter('flip_runtime_odom_yaw').get_parameter_value().bool_value
        self.world_start_pose = load_world_start_pose(self.task_spec_path, self.task_id, self.flip_scene_y)
        self.planner_settings = parse_optional_json(self.planner_settings_json, "planner_settings_json")
        self.speed_profile = parse_optional_json(self.speed_profile_json, "speed_profile_json")

        sample_frequency_hz = self.get_parameter('sample_frequency_hz').get_parameter_value().double_value
        self.dt = 1.0 / sample_frequency_hz if sample_frequency_hz > 0 else DEFAULT_DT

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        traj_name = requested_name or f"{stamp}_{self.get_name()}"
        self.traj_dir = Path(base_dir) / traj_name
        self.img_dir = self.traj_dir / "img"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.poses_path = self.traj_dir / "poses.jsonl"
        self.metadata_path = self.traj_dir / "metadata.json"
        self.write_metadata(traj_name)

        self.cam_subscriber = self.create_subscription(
            Image,
            self.get_parameter('camera_topic').get_parameter_value().string_value,
            self.cam_callback,
            qos_profile_sensor_data,
        )

        self.odom_subscriber = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self.odom_callback,
            qos_profile_sensor_data,
        )

        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            self.get_parameter('cmd_vel_topic').get_parameter_value().string_value,
            self.cmd_vel_callback,
            10,
        )

        self.action_chunk_subscriber = self.create_subscription(
            ActionChunk,
            self.get_parameter('action_chunk_topic').get_parameter_value().string_value,
            self.action_chunk_callback,
            10,
        )

        self.get_logger().info(f"Collecting sim stream to {self.traj_dir}")

    def cam_callback(self, msg):
        img_time = stamp_to_seconds(msg.header.stamp)
        if img_time <= 0:
            img_time = self.get_clock().now().nanoseconds * 1e-9

        if self.current_pose is None:
            self.get_logger().warn('No starting pose available')
            return

        if img_time - self.previous_img_time > self.dt:
            try:
                self.log_img_pose_pair(msg, img_time)
                self.previous_img_time = img_time
            except ImageEncodeError:
                self.get_logger().warn('Failed to log sim sample, will retry next frame')

    def odom_callback(self, msg):
        msg_time = stamp_to_seconds(msg.header.stamp)
        if msg_time <= 0:
            msg_time = self.get_clock().now().nanoseconds * 1e-9

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        heading = yaw_from_quaternion(msg.pose.pose.orientation)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        yaw_rate = msg.twist.twist.angular.z

        if self.flip_runtime_odom_y:
            y = -y
            heading = -heading
            vy = -vy
            yaw_rate = -yaw_rate
        elif self.flip_runtime_odom_yaw:
            heading = -heading
            yaw_rate = -yaw_rate

        local_pose = (msg_time, x, y, heading)
        world_x, world_y, world_heading = local_odom_to_world(local_pose, self.world_start_pose)
        self.current_pose = (msg_time, world_x, world_y, world_heading)
        self.current_local_pose = local_pose
        self.current_velocity = (msg_time, vx, vy, yaw_rate)

    def cmd_vel_callback(self, msg):
        msg_time = self.get_clock().now().nanoseconds * 1e-9
        self.current_cmd_vel = {
            "time": msg_time,
            "linear_x": msg.linear.x,
            "linear_y": msg.linear.y,
            "angular_z": msg.angular.z,
        }

    def action_chunk_callback(self, msg):
        msg_time = stamp_to_seconds(msg.header.stamp)
        if msg_time <= 0:
            msg_time = self.get_clock().now().nanoseconds * 1e-9

        self.current_action_chunk = {
            "time": msg_time,
            "seq_num": msg.seq_num,
            "relative_poses": [
                [pose.x, pose.y, pose.theta]
                for pose in msg.relative_poses
            ],
        }

    def current_action_chunk_age_s(self, sample_time_s):
        if not isinstance(self.current_action_chunk, dict):
            return None
        action_time = self.current_action_chunk.get("time")
        try:
            return float(sample_time_s) - float(action_time)
        except (TypeError, ValueError):
            return None

    def encode_img(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        ok, encoded = cv2.imencode(
            '.jpg',
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        return ok, encoded

    def log_img_pose_pair(self, msg, img_time):
        ok, img = self.encode_img(msg)
        if not ok:
            raise ImageEncodeError('Failed to JPEG-encode camera frame')

        img_id = f"{int(img_time * 1000)}"
        image_path = self.img_dir / f"{img_id}.jpg"
        image_path.write_bytes(img.tobytes())

        record = {
            "image": image_path.name,
            "img_time": img_time,
            "pose": self.current_pose,
            "local_pose": self.current_local_pose,
            "velocity": self.current_velocity,
            "cmd_vel": self.current_cmd_vel,
            "action_chunk": self.current_action_chunk,
            "action_chunk_age_s": self.current_action_chunk_age_s(img_time),
            "language_instruction": self.language_instruction,
            "dataset_name": self.dataset_name,
            "task_id": self.task_id,
            "task_spec": self.task_spec_path,
            "variant_id": self.variant_id,
            "variant_type": self.variant_type,
            "recovery_case": self.recovery_case,
            "planner_settings": self.planner_settings,
            "speed_profile": self.speed_profile,
        }
        with open(self.poses_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def write_metadata(self, trajectory_name):
        structured_task = None
        if self.structured_task_json:
            structured_task = json.loads(self.structured_task_json)

        metadata = {
            "dataset_name": self.dataset_name,
            "trajectory_name": trajectory_name,
            "task_id": self.task_id,
            "variant_id": self.variant_id,
            "variant_type": self.variant_type,
            "recovery_case": self.recovery_case,
            "language_instruction": self.language_instruction,
            "structured_task": structured_task,
            "planner_settings": self.planner_settings,
            "speed_profile": self.speed_profile,
            "camera_topic": self.get_parameter('camera_topic').get_parameter_value().string_value,
            "odom_topic": self.get_parameter('odom_topic').get_parameter_value().string_value,
            "cmd_vel_topic": self.get_parameter('cmd_vel_topic').get_parameter_value().string_value,
            "action_chunk_topic": self.get_parameter('action_chunk_topic').get_parameter_value().string_value,
            "world_start_pose": self.world_start_pose,
            "sample_frequency_hz": self.get_parameter('sample_frequency_hz').get_parameter_value().double_value,
            "jpeg_quality": self.jpeg_quality,
            "flip_isaac_y": self.flip_isaac_y,
            "flip_scene_y": self.flip_scene_y,
            "flip_runtime_odom_y": self.flip_runtime_odom_y,
            "flip_runtime_odom_yaw": self.flip_runtime_odom_yaw,
            "format": "stream_jsonl",
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def main(args=None):
    rclpy.init(args=args)
    node = SimDatasetCollectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
