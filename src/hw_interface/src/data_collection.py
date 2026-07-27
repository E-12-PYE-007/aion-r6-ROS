#!/usr/bin/env python3
""" Data collection node for Aion R6. Collects position and heading data alongside
    imgs from camera, logging action chunks for use in VLA training."""

from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
import px4_msgs.msg
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import json
import math
import numpy as np
import rclpy
from rclpy.node import Node

# Define constants

DT = 1/3 # Spacing between waypoints to be collected
LOG_DIR = Path("data_collection_output")  # placeholder output directory

@dataclass
class PendingChunk:
    image: object
    anchor_time: float
    anchor_pose: tuple
    num_targets: int = 0
    targets: list = field(default_factory=list)


class DataCollectionNode(Node):
    def __init__(self):
        super().__init__('aion_data_collector')

        self.current_chunk = None
        self.next_chunk = None
        self.current_pose = None

        self.log_dir = LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.log_dir / "chunks.jsonl"

        self.cam_subscriber = self.create_subscription(
            Image,
            "/vla/cam1", # placeholder topic name
            self.cam_callback,
            qos_profile_sensor_data,
        )

        self.odom_subscriber = self.create_subscription(
            px4_msgs.msg.VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.odom_callback,
            qos_profile_sensor_data
        )

    def encode_img(self, msg):
        bgr = self.ros_image_to_bgr(msg)
        ok, encoded = cv2.imencode(
            '.jpg',
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )
        return ok, encoded
    @staticmethod
    def ros_image_to_bgr(msg):
        """Convert common ROS image encodings into OpenCV BGR for JPEG encoding."""
        channels = 4 if msg.encoding in ('rgba8', 'bgra8') else 3
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, channels))
        if msg.encoding == 'rgb8':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if msg.encoding == 'rgba8':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if msg.encoding == 'bgra8':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image[..., :3]

    def cam_callback(self, msg):
        if self.current_pose is None:
            return # Cannot start logging chunks without a starting pose

        # TODO: Check synchronicity between ros2 clock and px4 clock - maybe better to use current_pose time as anchor time or an internal timer
        anchor_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 # image capture time in seconds

        if self.current_chunk is None: # No current_chunk being worked on - save img and start building. Takes branch on startup only.
            # Encode image as jpg
            ok, encoded = self.encode_img(msg)
            if not ok:
                self.get_logger().warn('Failed to JPEG-encode camera frame')
                return

            self.current_chunk = PendingChunk(encoded, anchor_time, self.current_pose)

        elif self.next_chunk is None and (anchor_time-self.current_chunk.anchor_time) > DT*4:
            ok, encoded = self.encode_img(msg)
            if not ok:
                self.get_logger().warn('Failed to JPEG-encode camera frame')
                return
            self.next_chunk = PendingChunk(encoded, anchor_time, self.current_pose)

        return

    def odom_callback(self, msg):

        self.current_pose = (msg.timestamp, msg.x, msg.y, msg.heading)
        if self.next_chunk is not None:
            self.maybe_add_odom(msg, self.next_chunk)

        if self.current_chunk is not None:
            self.maybe_add_odom(msg, self.current_chunk)

            if self.current_chunk.num_targets == 8:
                self.log_chunk(self.current_chunk)
                self.current_chunk=self.next_chunk
                self.next_chunk=None

    def maybe_add_odom(self, msg, chunk):
        dt = (msg.timestamp*1e-6)-chunk.anchor_time
        if dt > DT*(chunk.num_targets+1):
            chunk.num_targets+=1
            chunk.targets.append(self.get_relative_pose(chunk))
        

    def log_chunk(self, chunk):
        if not self.chunk_is_valid(chunk):
            return

        chunk_id = f"{int(chunk.anchor_time * 1000)}"
        image_path = self.log_dir / f"{chunk_id}.jpg"
        image_path.write_bytes(chunk.image.tobytes())

        record = {
            "image": image_path.name,
            "anchor_time": chunk.anchor_time,
            "anchor_pose": chunk.anchor_pose,
            "targets": chunk.targets,
        }
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def chunk_is_valid(self, chunk):
        # TODO: reject the chunk if any target's delta_t drifted too far from its expected i*DT
        return True

    def get_relative_pose(self, chunk):
        dt, x, y, heading = self.current_pose
        a_dt, a_x, a_y, a_heading = chunk.anchor_pose
        d_heading = math.atan2(math.sin(heading - a_heading), math.cos(heading - a_heading))
        return (dt - a_dt, x - a_x, y - a_y, d_heading)

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
