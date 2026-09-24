#!/usr/bin/env python3
"""
Keyboard console for teleop-driven episode data collection.

Runs in one attached terminal on the Jetson:
  arrow keys drive the rover by publishing /cmd_vel
  x starts an episode
  s stops the episode and asks whether to save it
  q quits

The arrow-key teleop behavior mirrors key_teleop's simple mobile-base model:
recent arrow presses map to fixed linear/angular rates, and commands return
to zero when keys stop repeating.
"""

import select
import shutil
import sys
import termios
import tty
from pathlib import Path

import rclpy
from aion_msgs.srv import StartEpisode
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_srvs.srv import Trigger

COLLECTOR_NODE = 'episode_data_collector'
START_KEY = 'x'
STOP_KEY = 's'
QUIT_KEYS = {'q', '\x03'}  # 'q' or Ctrl+C
NAME_HINT = 'e.g. fls, fro, fltr, frtl'
FENCE_PROMPTS = {
    'fl': 'follow the fence on your left',
    'fr': 'follow the fence on your right',
}

KEY_UP = '\x1b[A'
KEY_DOWN = '\x1b[B'
KEY_RIGHT = '\x1b[C'
KEY_LEFT = '\x1b[D'
KEY_UP_ALT = '\x1bOA'
KEY_DOWN_ALT = '\x1bOB'
KEY_RIGHT_ALT = '\x1bOC'
KEY_LEFT_ALT = '\x1bOD'

KEY_ALIASES = {
    KEY_UP_ALT: KEY_UP,
    KEY_DOWN_ALT: KEY_DOWN,
    KEY_RIGHT_ALT: KEY_RIGHT,
    KEY_LEFT_ALT: KEY_LEFT,
}


def prompt_from_episode_name(name):
    descriptor = name.lower()
    for prefix, prompt in FENCE_PROMPTS.items():
        if descriptor.startswith(prefix):
            return prompt
    return None


def read_key(timeout_sec):
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout_sec)
        if not ready:
            return None

        key = sys.stdin.read(1)
        if key != '\x1b':
            return key

        # Arrow keys arrive as three-byte escape sequences.
        sequence = [key]
        for _ in range(2):
            ready, _, _ = select.select([sys.stdin], [], [], 0.01)
            if not ready:
                break
            sequence.append(sys.stdin.read(1))
        return KEY_ALIASES.get(''.join(sequence), ''.join(sequence))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class CollectionInterfaceClient(Node):
    def __init__(self):
        super().__init__('collection_interface')

        cmd_vel_topic = self.declare_parameter('cmd_vel_topic', '/cmd_vel').value
        self.forward_rate = float(self.declare_parameter('forward_rate', 0.4).value)
        self.backward_rate = float(self.declare_parameter('backward_rate', 0.4).value)
        self.rotation_rate = float(self.declare_parameter('rotation_rate', 0.4).value)
        self.hz = float(self.declare_parameter('hz', 10.0).value)
        self.key_timeout = float(self.declare_parameter('key_timeout', 0.5).value)

        self.start_cli = self.create_client(StartEpisode, f'/{COLLECTOR_NODE}/start_episode')
        self.stop_cli = self.create_client(Trigger, f'/{COLLECTOR_NODE}/stop_episode')
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.recording = False
        self.episode_dir = None
        self.last_pressed = {}
        self.last_command = None
        self.last_printed_command = None

    def wait_for_services(self, timeout_sec=5.0):
        for cli, name in ((self.start_cli, 'start_episode'), (self.stop_cli, 'stop_episode')):
            if not cli.wait_for_service(timeout_sec=timeout_sec):
                raise RuntimeError(
                    f"Service '{name}' not available - is {COLLECTOR_NODE} running?"
                )

    def start_episode(self, base_name, prompt):
        name = base_name
        suffix = 1
        while True:
            req = StartEpisode.Request()
            req.name = name
            req.prompt = prompt
            future = self.start_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()

            if resp.success:
                self.recording = True
                self.episode_dir = Path(resp.episode_dir)
                print(f"[recording] {resp.message} -> {resp.episode_dir}")
                return

            if 'already exists' in resp.message:
                suffix += 1
                name = f"{base_name}_{suffix:02d}"
                continue

            print(f"[error] {resp.message}")
            return

    def stop_episode(self):
        future = self.stop_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp.success:
            self.recording = False
        print(f"[{'stopped' if resp.success else 'error'}] {resp.message}")
        return resp.success

    def confirm_save(self):
        if self.episode_dir is None:
            return
        answer = input('Save episode? [y/n]: ').strip().lower()
        if answer.startswith('n'):
            shutil.rmtree(self.episode_dir, ignore_errors=True)
            print(f'[discarded] {self.episode_dir}')
        else:
            print(f'[saved] {self.episode_dir}')
        self.episode_dir = None

    def handle_drive_key(self, key):
        if key in {KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT}:
            self.last_pressed[key] = self.get_clock().now()
            return True
        return False

    def publish_drive_command(self, force=False):
        now = self.get_clock().now()
        active_keys = [
            key for key, stamp in self.last_pressed.items()
            if (now - stamp).nanoseconds / 1e9 < self.key_timeout
        ]
        self.last_pressed = {key: self.last_pressed[key] for key in active_keys}

        linear = 0.0
        angular = 0.0
        if KEY_UP in active_keys:
            linear += self.forward_rate
        if KEY_DOWN in active_keys:
            linear -= self.backward_rate
        if KEY_LEFT in active_keys:
            angular += self.rotation_rate
        if KEY_RIGHT in active_keys:
            angular -= self.rotation_rate

        command = (linear, angular)
        if not force and command == self.last_command:
            return

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_vel_pub.publish(cmd)
        self.last_command = command
        self.print_drive_command(command)

    def stop_drive(self):
        self.last_pressed.clear()
        if self.last_command != (0.0, 0.0):
            self.cmd_vel_pub.publish(Twist())
            self.last_command = (0.0, 0.0)
            self.print_drive_command((0.0, 0.0))

    def print_drive_command(self, command):
        if command == self.last_printed_command:
            return
        linear, angular = command
        print(f"[drive] linear.x={linear:.2f} angular.z={angular:.2f}")
        self.last_printed_command = command


def prompt_for_episode(node):
    node.stop_drive()
    name = input(f"Episode name ({NAME_HINT}): ").strip()
    if not name:
        print('[warn] empty name, cancelled')
        return

    prompt = prompt_from_episode_name(name)
    if prompt is None:
        print('[warn] name must start with fl or fr, cancelled')
        return

    print(f'[prompt] {prompt}')
    node.start_episode(name, prompt)


def main(args=None):
    rclpy.init(args=args)
    node = CollectionInterfaceClient()

    try:
        node.wait_for_services()
    except RuntimeError as e:
        print(e)
        node.destroy_node()
        rclpy.shutdown()
        return

    print(
        f"Ready. arrow keys drive  [{START_KEY}] start episode  "
        f"[{STOP_KEY}] stop episode  [q] quit"
    )
    print(
        f"Drive rates: forward={node.forward_rate:.2f} m/s  "
        f"backward={node.backward_rate:.2f} m/s  turn={node.rotation_rate:.2f} rad/s"
    )

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            key = read_key(1.0 / node.hz)
            drive_key = False

            if key in QUIT_KEYS:
                node.stop_drive()
                if node.recording and node.stop_episode():
                    node.confirm_save()
                break

            if key == START_KEY:
                if node.recording:
                    print('[warn] already recording, stop it first')
                else:
                    prompt_for_episode(node)
            elif key == STOP_KEY:
                node.stop_drive()
                if not node.recording:
                    print('[warn] not currently recording')
                elif node.stop_episode():
                    node.confirm_save()
            elif key is not None:
                drive_key = node.handle_drive_key(key)

            node.publish_drive_command(force=drive_key)
    finally:
        node.stop_drive()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
