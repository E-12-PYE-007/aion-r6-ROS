#!/usr/bin/env python3

import os
from collections import deque
from functools import lru_cache

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image as PILImage

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from custom_msgs.msg import ActionChunk, HiddenState, ImageWithSeqNum

from ag_vla.edge_adapter import Edge_adapter


RESUME_STEP = 750000
OBS_ENCODING_SIZE = 1024
MHA_NUM_ATTENTION_HEADS = 4
MHA_NUM_ATTENTION_LAYERS = 4
MHA_FF_DIM_FACTOR = 4
IMAGE_INTERMEDIATE_SIZE = (224, 224)  # PIL resize before tensor resize (matches reference)
IMAGE_SIZE = (96, 96)
METRIC_WAYPOINT_SPACING = 0.1 
IMAGE_BUFFER_SIZE = 80  # 8 secs worth of data
SYS1_RATE_HZ = 8.0
NUM_ACTIONS_CHUNK = 8

SHEAD_PATH = "/home/vla-cap/AsyncVLA_release"


_normalise = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class Sys1(Node):
    def __init__(self):
        super().__init__("sys1")
        self.get_logger().info("[AgVLA Sys1] initialising...")

        # Load model
        shead, self.device = _load_model(SHEAD_PATH, RESUME_STEP)
        self.inference = Inference(shead, self.device)
        self.get_logger().info("[AgVLA Sys1] Model loaded")

        # Initialise states
        self.bridge = CvBridge()
        self.curr_img = None
        self.curr_img_seq_num = None
        self.img_buffer = {}
        self.img_buffer_keys = deque(maxlen=IMAGE_BUFFER_SIZE)
        self.latest_hidden_state = None
        self.latest_hidden_seq_num = None

        # Publishers
        self.action_chunk_pub = self.create_publisher(ActionChunk, "/agvla/action_chunk", 1)

        # Subscribers
        self.create_subscription(HiddenState, "/agvla/hidden_state", self.hidden_state_callback, 1)
        self.create_subscription(ImageWithSeqNum, "/cam", self.img_callback, 1)

        self.get_logger().info("[AgVLA Sys1] Subscribed to /cam (ImageWithSeqNum)")

        self.create_timer(1.0 / SYS1_RATE_HZ, self.timer_callback)
        self.get_logger().info("[AgVLA Sys1] Triggering main control loop...")

    def img_callback(self, msg: ImageWithSeqNum):
        img = PILImage.fromarray(self.bridge.compressed_imgmsg_to_cv2(msg.img, desired_encoding="rgb8"))
        self._store_frame(img, msg.img_seq_num)

    def _store_frame(self, img: PILImage.Image, seq_num: int):
        processed = _process_image(img, self.device)

        # Evict oldest entry before it gets dropped from the deque
        if len(self.img_buffer_keys) == IMAGE_BUFFER_SIZE:
            self.img_buffer.pop(self.img_buffer_keys[0], None)
        self.img_buffer_keys.append(seq_num)
        self.img_buffer[seq_num] = processed
        self.curr_img = processed
        self.curr_img_seq_num = seq_num

    def hidden_state_callback(self, msg: HiddenState):
        self.latest_hidden_state = torch.tensor(msg.hidden_states.data, dtype=torch.float32).reshape(1, 8, 1024).to(torch.bfloat16).to(self.device)
        self.latest_hidden_seq_num = msg.img_seq_num

    def timer_callback(self):
        projected_actions = self.latest_hidden_state
        old_img_seq_num = self.latest_hidden_seq_num
        past_img = self.img_buffer.get(old_img_seq_num)
        curr_img = self.curr_img
        curr_img_seq_num = self.curr_img_seq_num

        if projected_actions is None or curr_img is None or past_img is None:
            self.get_logger().info("[AgVLA Sys1] Missing imgs or actions")
            return

        poses = self.inference.run(curr_img, past_img, projected_actions)
        self.publish_action_chunk(poses, old_img_seq_num, curr_img_seq_num)

    def publish_action_chunk(
        self,
        poses: np.ndarray,
        old_img_seq_num: int,
        curr_img_seq_num: int,
    ):
        chunk = ActionChunk()
        chunk.header.stamp = self.get_clock().now().to_msg()
        chunk.header.frame_id = "base_link"
        chunk.old_img_seq_num = old_img_seq_num
        chunk.curr_img_seq_num = curr_img_seq_num

        for t in range(NUM_ACTIONS_CHUNK):
            pose = Pose2D()
            pose.x = float(poses[0, t, 0]) * METRIC_WAYPOINT_SPACING
            pose.y = float(poses[0, t, 1]) * METRIC_WAYPOINT_SPACING
            pose.theta = float(np.arctan2(poses[0, t, 3], poses[0, t, 2]))
            chunk.relative_poses[t] = pose

        self.action_chunk_pub.publish(chunk)
        self.get_logger().info(f"[AgVLA Sys1] Published action chunk old_img_seq={old_img_seq_num} "
            f"curr_img_seq={curr_img_seq_num}")


class Inference:
    def __init__(self, shead, device):
        self.shead = shead
        self.device = device

    def run(self, curr_img: torch.Tensor, past_img: torch.Tensor, projected_actions: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            predicted_delta = self.shead(curr_img, past_img, projected_actions)
            predicted_poses = _delta_to_pose(predicted_delta)
        return predicted_poses.cpu().float().numpy()


@lru_cache(maxsize=1)
def _load_model(shead_path: str, resume_step: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    shead = Edge_adapter(
        obs_encoding_size=OBS_ENCODING_SIZE,
        mha_num_attention_heads=MHA_NUM_ATTENTION_HEADS,
        mha_num_attention_layers=MHA_NUM_ATTENTION_LAYERS,
        mha_ff_dim_factor=MHA_FF_DIM_FACTOR,
    )

    checkpoint_path = os.path.join(shead_path, f"shead--{resume_step}_checkpoint.pt")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    shead.load_state_dict(state_dict, strict=False)
    shead.to(torch.bfloat16).to(device)
    shead.eval()

    return shead, device


def _process_image(img: PILImage.Image, device: torch.device) -> torch.Tensor:
    img = img.resize(IMAGE_INTERMEDIATE_SIZE, PILImage.BILINEAR)
    tensor = TF.to_tensor(img)
    tensor = TF.resize(tensor, list(IMAGE_SIZE)).unsqueeze(0)
    return _normalise(tensor).to(device).to(torch.bfloat16)


def _delta_to_pose(delta: torch.Tensor) -> torch.Tensor:
    """Convert predicted delta actions [N, T, 4] to absolute poses [N, T, 4]."""
    dx = delta[..., 0]
    dy = delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])

    N, T = dx.shape
    poses = []
    x, y, theta = dx[:, 0], dy[:, 0], dtheta[:, 0]
    poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))

    for t in range(1, T):
        ct, st = torch.cos(theta), torch.sin(theta)
        x = x + ct * dx[:, t] - st * dy[:, t]
        y = y + st * dx[:, t] + ct * dy[:, t]
        theta = theta + dtheta[:, t]
        poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))

    return torch.stack(poses, dim=1)


def main(args=None):
    rclpy.init(args=args)

    sys1_node = Sys1()
    rclpy.spin(sys1_node)

    sys1_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
