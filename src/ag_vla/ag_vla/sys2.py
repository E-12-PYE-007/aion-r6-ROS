#!/usr/bin/env python3

import os
from functools import lru_cache
from threading import Lock

import numpy as np
import torch
from PIL import Image as PILImage
from torch.nn.utils.rnn import pad_sequence

import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from custom_msgs.msg import HiddenState, ImageWithSeqNum
from peft import PeftModel

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.small_head import Proj_Actiontokens
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast


RESUME_STEP = 750000
DEVICE_TYPE = "cuda"

# FINETUNE_ADAPTER_DIR = "/home/vla-cap/AsyncVLA/agvla_weights/out/h100/r32_a16_dora1_lr0.0005_bs16/20260904_132010/step-0015000/lora_adapter"
# FINETUNE_ADAPTER_DIR = "/home/vla-cap/AsyncVLA/agvla_weights/out/a100/r32_a16_dora0_lr0.0005_bs16/20260904_143315/step-0010000/lora_adapter"
# FINETUNE_DIR = "/home/vla-cap/AsyncVLA/agvla_weights/out/h100/r32_a16_dora0_lr0.0005_bs16/20260916_102204/step-0005000"
# FINETUNE_DIR = "/home/vla-cap/AsyncVLA/agvla_weights/out/h100/20260918/r32_a16_dora0_lr0.0001_bs16_B_fp32_lr1e-4/step-0005000"
# FINETUNE_DIR = "/home/vla-cap/AsyncVLA/agvla_weights/body_r32_a16_dora0_lr0.0001_bs16_B_fp32_lr1e-4/step-0005000"
# FINETUNE_DIR = "/home/vla-cap/AsyncVLA/agvla_weights/body_r32_a16_dora0_lr0.0001_bs16_freeze_actproj/step-0003500"
FINETUNE_DIR = "/home/vla-cap/AsyncVLA/agvla_weights/real_r32_a16_dora0_lr0.0001_bs16_from2k/step-0003000"
# FINETUNE_DIR = ""

METRIC_WAYPOINT_SPACING = 0.1  # metres per waypoint unit (matches sys1)
SYS2_RATE_HZ = 5.0


def _seq_num_from_stamp(stamp) -> int:
    """Per-frame id derived from a Header stamp, in milliseconds.

    Isaac's raw sensor_msgs/Image carries no seq num like ImageWithSeqNum does, so
    sys1 and sys2 each derive one from the same message stamp to stay aligned. Kept in
    milliseconds so it fits the int32/uint32 seq_num fields for any realistic sim run.
    Must match sys1's copy of this helper.
    """
    return stamp.sec * 1000 + stamp.nanosec // 1_000_000


class Sys2(Node):
    def __init__(self):
        super().__init__("sys2")
        self.get_logger().info("[AgVLA Sys2] initialising...")

        self.declare_parameter("vla_path", "")
        vla_path = self.get_parameter("vla_path").get_parameter_value().string_value

        # Load model
        vla, action_proj, action_head, device, num_patches, action_tokenizer, processor = _load_model(self, vla_path, RESUME_STEP, FINETUNE_DIR)
        self.inference = Inference(vla, action_proj, action_head, device, num_patches, action_tokenizer, processor)
        self.get_logger().info("[AgVLA Sys2] Model loaded")

        self.declare_parameter("goal", "")
        self.goal_text = self.get_parameter("goal").get_parameter_value().string_value
        self.get_logger().info(f"[AgVLA Sys2] Goal set as: '{self.goal_text}'")

        self.declare_parameter("sim", "")
        self.sim = self.get_parameter("sim").get_parameter_value().string_value
        self.get_logger().info(f"[AgVLA Sys2] Simulation environment: '{self.sim or 'none'}'")

        # Latest observation; locked because img_callback and timer_callback run on
        # different executor threads.
        self._img_lock = Lock()
        self.latest_img = None
        self.latest_img_seq_num = None

        # Publishers
        self.hidden_state_pub = self.create_publisher(HiddenState, "/agvla/hidden_state", 1)

        # Subscribers. The camera topic gets its own callback group so frames keep
        # arriving during the forward pass; see main().
        self.bridge = CvBridge()
        if self.sim == "isaac":
            # Isaac publishes a plain sensor_msgs/Image on /vla/cam.
            self.create_subscription(
                Image, "/vla/cam", self.isaac_img_callback, 1,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )
            self.get_logger().info("[AgVLA Sys2] Subscribed to /vla/cam (Image, isaac)")
        else:
            # Real robot / default: ImageWithSeqNum on /cam.
            self.create_subscription(
                ImageWithSeqNum, "/cam", self.img_callback, 1,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )
            self.get_logger().info("[AgVLA Sys2] Subscribed to /cam (ImageWithSeqNum)")

        self.create_timer(
            1.0 / SYS2_RATE_HZ, self.timer_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().info(f"[AgVLA Sys2] Triggering main loop at {SYS2_RATE_HZ} Hz...")

    def img_callback(self, msg: ImageWithSeqNum):
        """Stash latest frame; inference runs on the timer, not here."""
        img = PILImage.fromarray(self.bridge.compressed_imgmsg_to_cv2(msg.img, desired_encoding="rgb8"))
        with self._img_lock:
            self.latest_img = img
            self.latest_img_seq_num = msg.img_seq_num

    def isaac_img_callback(self, msg: Image):
        """Isaac's /vla/cam is a raw Image with no seq num, so derive one from the
        header stamp. sys1 sees the same message and derives the same id, which the
        async pipeline relies on to pair this frame with sys2's hidden state."""
        img = PILImage.fromarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8"))
        with self._img_lock:
            self.latest_img = img
            self.latest_img_seq_num = _seq_num_from_stamp(msg.header.stamp)

    def timer_callback(self):
        with self._img_lock:
            img = self.latest_img
            seq_num = self.latest_img_seq_num
        if img is None:
            return

        # Node clock, so it follows use_sim_time if that is ever enabled.
        start = self.get_clock().now()
        projected, _ = self.inference.run(img, self.goal_text)
        inference_ms = (self.get_clock().now() - start).nanoseconds / 1e6

        # Newest frame to land during the forward pass.
        with self._img_lock:
            end_seq_num = self.latest_img_seq_num

        hidden_state_msg = HiddenState()
        hidden_state_msg.header.stamp = self.get_clock().now().to_msg()
        hidden_state_msg.img_seq_num = seq_num
        hidden_state_msg.end_img_seq_num = end_seq_num
        hidden_state_msg.inference_ms = inference_ms
        hidden_state_msg.hidden_states.data = projected.reshape(-1).astype(np.float32).tolist()

        self.hidden_state_pub.publish(hidden_state_msg)
        self.get_logger().info(
            f"[AgVLA Sys2] Published hidden state for img_seq={seq_num} "
            f"(newest at finish: {end_seq_num}, {inference_ms:.0f}ms)"
        )


class Inference:
    def __init__(self, vla, action_proj, action_head, device, num_patches, action_tokenizer, processor):
        self.vla = vla
        self.action_proj = action_proj
        self.action_head = action_head
        self.device = device
        self.num_patches = num_patches
        self.action_tokenizer = action_tokenizer
        self.processor = processor
        # TODO: uncomment and pass pose_projector here when enabling proprio conditioning
        # self.pose_projector = pose_projector

    def run(self, img: PILImage.Image, goal_text: str):
        """Return (projected_hidden_state, omni_actions) as (float32 numpy, float32 numpy)."""
        batch = self._prepare_batch(img, goal_text)
        return self._forward(batch)

    def _prepare_batch(self, img: PILImage.Image, goal_text: str) -> dict:
        IGNORE_INDEX = -100
        actions = np.random.rand(8, 4)  # dummy actions for token layout only

        action_chunk_string = self.action_tokenizer(actions[0]) + "".join(self.action_tokenizer(actions[1:]))

        prompt_builder = PurePromptBuilder("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {goal_text}?")
        prompt_builder.add_turn("gpt", action_chunk_string)

        tokenizer = self.processor.tokenizer
        input_ids = torch.tensor(tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[: -(len(action_chunk_string) + 1)] = IGNORE_INDEX

        pixel_values = self.processor.image_processor.apply_transform(img)

        input_ids_b = pad_sequence([input_ids], batch_first=True, padding_value=tokenizer.pad_token_id)
        labels_b = pad_sequence([labels], batch_first=True, padding_value=IGNORE_INDEX)
        input_ids_b = input_ids_b[:, : tokenizer.model_max_length]
        labels_b = labels_b[:, : tokenizer.model_max_length]

        stacked = torch.stack([pixel_values])
        pixel_values_b = torch.cat([stacked, stacked], dim=1)  # duplicate as goal image

        return {
            "pixel_values": pixel_values_b,
            "input_ids": input_ids_b,
            "attention_mask": input_ids_b.ne(tokenizer.pad_token_id),
            "labels": labels_b,
        }

    def _forward(self, batch: dict) -> np.ndarray:
        modality_id = torch.as_tensor([7], dtype=torch.float32, device=self.device)

        with torch.no_grad(), torch.autocast(DEVICE_TYPE, dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = self.vla(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(self.device),
                modality_id=modality_id.to(torch.bfloat16),
                labels=batch["labels"].to(self.device),
                output_hidden_states=True,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=False,
                # TODO: uncomment to match original run_asyncvla.py behaviour (always conditions on a
                # goal pose token even in language-only mode). If uncommented, also uncomment the
                # pose_projector lines in _load_model and add 1 to num_patches there.
                # proprio=torch.zeros(1, 4, dtype=torch.bfloat16, device=self.device),
                # proprio_projector=self.pose_projector,
            )

        gt_token_ids = batch["labels"][:, 1:].to(self.device)
        action_mask = get_current_action_mask(gt_token_ids) | get_next_actions_mask(gt_token_ids)

        text_hidden = output.hidden_states[-1][:, self.num_patches : -1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden = text_hidden[action_mask].reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)

        with torch.no_grad():
            mid_bf16 = modality_id.to(torch.bfloat16)
            projected = self.action_proj.predict_action(actions_hidden.detach(), mid_bf16)
            omni_actions = self.action_head.predict_action(actions_hidden.detach(), mid_bf16)

        return (
            projected.detach().to(torch.float32).cpu().numpy(),
            omni_actions.detach().to(torch.float32).cpu().numpy(),
        )


def _remove_ddp_prefix(state_dict: dict) -> dict:
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _load_checkpoint(module_name: str, path: str, step: int) -> dict:
    import os

    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    if not os.path.exists(checkpoint_path) and module_name == "pose_projector":
        checkpoint_path = os.path.join(path, f"proprio_projector--{step}_checkpoint.pt")
    return _remove_ddp_prefix(torch.load(checkpoint_path, map_location="cpu"))


@lru_cache(maxsize=1)
def _load_model(self, vla_path: str, resume_step: int, finetune_dir: str = ""):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    if finetune_dir:
        adapter_dir = os.path.join(finetune_dir, "lora_adapter")
        rclpy.logging.get_logger("sys2").info(f"[AgVLA Sys2] Applying fine-tuned adapter: {adapter_dir}")
        vla = PeftModel.from_pretrained(vla, adapter_dir).merge_and_unload().to(device)

    vla.vision_backbone.set_num_images_in_input(2)
    vla.to(dtype=torch.bfloat16, device=device)

    # Skip vision encoding on the duplicated goal image.
    # We also append a zero pose token so the total patch prefix is 513 tokens.
    def _patched_process_vision(pixel_values, language_embeddings=None, use_film=False):
        obs = pixel_values[:, :6]
        img, img_fused = torch.split(obs, [3, 3], dim=1)
        patches = vla.vision_backbone.featurizer(img)
        patches_fused = vla.vision_backbone.fused_featurizer(img_fused)
        single = torch.cat([patches, patches_fused], dim=2)
        projected = vla.projector(single)          # (B, 256, llm_dim)
        tiled = projected.repeat(1, 2, 1)          # (B, 512, llm_dim)
        zero_pose = torch.zeros(
            tiled.shape[0], 1, tiled.shape[2], dtype=tiled.dtype, device=tiled.device
        )
        return torch.cat([tiled, zero_pose], dim=1)  # (B, 513, llm_dim)
    vla._process_vision_features = _patched_process_vision

    action_proj = Proj_Actiontokens(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=1024)
    if finetune_dir:
        action_proj_checkpoint = os.path.join(finetune_dir, "action_proj.pt")
        rclpy.logging.get_logger("sys2").info(
            f"[AgVLA Sys2] Loading fine-tuned action_proj weights: {action_proj_checkpoint}"
        )
        action_proj.load_state_dict(
            _remove_ddp_prefix(torch.load(action_proj_checkpoint, map_location="cpu"))
        )
    else:
        action_proj.load_state_dict(_load_checkpoint("action_proj", vla_path, resume_step))
    action_proj = action_proj.to(torch.bfloat16).to(device)

    # Upstream run_vla.py runs both of these in eval mode. Leaving them in train mode
    # keeps dropout active during inference and makes the output non-deterministic.
    vla.eval()
    action_proj.eval()

    action_head = L1RegressionActionHead_idcat(
        input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM
    )
    action_head.load_state_dict(_load_checkpoint("action_head", vla_path, resume_step))
    action_head = action_head.to(torch.bfloat16).to(device).eval()

    # TODO: uncomment to load pose_projector and enable proprio conditioning (see _forward TODO).
    # from prismatic.models.projectors import ProprioProjector
    # from prismatic.vla.constants import POSE_DIM
    # pose_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=POSE_DIM)
    # pose_projector.load_state_dict(_load_checkpoint('pose_projector', vla_path, resume_step))
    # pose_projector = pose_projector.to(torch.bfloat16).to(device)

    num_patches = (
        vla.vision_backbone.get_num_patches()
        * vla.vision_backbone.get_num_images_in_input()
        + 1  # pose slot appended in _patched_process_vision; matches training layout
    )
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    return vla, action_proj, action_head, device, num_patches, action_tokenizer, processor


def main(args=None):
    rclpy.init(args=args)

    sys2_node = Sys2()
    # MultiThreaded so /cam is still serviced while timer_callback blocks in the forward
    # pass. Single-threaded, end_img_seq_num would always just echo img_seq_num.
    executor = MultiThreadedExecutor()
    executor.add_node(sys2_node)
    try:
        executor.spin()
    finally:
        sys2_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
