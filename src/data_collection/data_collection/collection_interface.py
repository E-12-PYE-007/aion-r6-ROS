#!/usr/bin/env python3
"""
    Keyboard-driven client for episode_data_collector's start/stop services.
    Mirrors stream_data_collector, but using services and keyboard input.
    Run alongside teleop; prompts for an episode name and manages naming
    collisions by auto-incrementing a numeric suffix.
"""

import shutil
import sys
import termios
import tty
from pathlib import Path
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from aion_msgs.srv import StartEpisode

COLLECTOR_NODE = 'episode_data_collector'
START_KEY = 'x'
STOP_KEY = 's'
QUIT_KEYS = {'q', '\x03'} # 'q' or Ctrl+C
NAME_HINT = 'e.g. <target>_<follow_side>_turn_<turns>'


def read_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class CollectionInterfaceClient(Node):
    def __init__(self):
        super().__init__('collection_interface')
        self.start_cli = self.create_client(StartEpisode, f'/{COLLECTOR_NODE}/start_episode')
        self.stop_cli = self.create_client(Trigger, f'/{COLLECTOR_NODE}/stop_episode')
        self.recording = False
        self.episode_dir = None

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

    print(f"Ready. [{START_KEY}] start episode  [{STOP_KEY}] stop episode  [q] quit")

    try:
        while rclpy.ok():
            key = read_key()

            if key in QUIT_KEYS:
                if node.recording and node.stop_episode():
                    node.confirm_save()
                break

            elif key == START_KEY:
                if node.recording:
                    print('[warn] already recording, stop it first')
                    continue
                name = input(f"Episode name ({NAME_HINT}): ").strip()
                if not name:
                    print('[warn] empty name, cancelled')
                    continue
                prompt = input('Prompt describing this episode: ').strip()
                if not prompt:
                    print('[warn] empty prompt, cancelled')
                    continue
                node.start_episode(name, prompt)

            elif key == STOP_KEY:
                if not node.recording:
                    print('[warn] not currently recording')
                    continue
                if node.stop_episode():
                    node.confirm_save()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
