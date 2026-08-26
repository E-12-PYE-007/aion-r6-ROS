#!/usr/bin/env python3
""" Data collection node for Aion R6. Collects position and heading data alongside
    imgs from camera, logging action chunks for use in VLA training."""

from datetime import datetime
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
import px4_msgs.msg
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import cv2
from cv_bridge import CvBridge
import json
import math
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

# Define constants

DT = 1/3 # Spacing between waypoints to be collected


class ImageEncodeError(Exception):
    pass

@dataclass
class PendingChunk:
    image: object
    anchor_time: float
    anchor_pose: tuple
    num_targets: int = 0
    targets: list = field(default_factory=list)


class ChunkDataCollectionNode(Node):
    def __init__(self):
        super().__init__('chunk_data_collector')

        self.bridge = CvBridge()

        self.current_chunk = None
        self.next_chunk = None
        self.current_pose = None

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
            px4_msgs.msg.VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.odom_callback,
            qos_profile_sensor_data
        )

    def encode_img(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        ok, encoded = cv2.imencode(
            '.jpg',
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )
        return ok, encoded

    def cam_callback(self, msg):
        if self.current_pose is None:
            return # Cannot start logging chunks without a starting pose

        try:
            # TODO: Check synchronicity between ros2 clock and px4 clock - maybe better to use current_pose time as anchor time or an internal timer
            anchor_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 # image capture time in seconds

            if self.current_chunk is None: # No current_chunk being worked on - save img and start building. Takes branch on startup only.
                # Encode image as jpg
                ok, encoded = self.encode_img(msg)
                if not ok:
                    raise ImageEncodeError('Failed to JPEG-encode camera frame')

                self.current_chunk = PendingChunk(encoded, anchor_time, self.current_pose)

            elif self.next_chunk is None and (anchor_time-self.current_chunk.anchor_time) > DT*4:
                ok, encoded = self.encode_img(msg)
                if not ok:
                    raise ImageEncodeError('Failed to JPEG-encode camera frame')
                self.next_chunk = PendingChunk(encoded, anchor_time, self.current_pose)
        except ImageEncodeError:
            self.get_logger().warn('Failed to log image/pose pair, will retry next frame')

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
        image_path = self.img_dir / f"{chunk_id}.jpg"
        image_path.write_bytes(chunk.image.tobytes())

        record = {
            "image": image_path.name,
            "anchor_time": chunk.anchor_time,
            "anchor_pose": chunk.anchor_pose,
            "targets": chunk.targets,
        }
        with open(self.poses_path, "a") as f:
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
    node = ChunkDataCollectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
