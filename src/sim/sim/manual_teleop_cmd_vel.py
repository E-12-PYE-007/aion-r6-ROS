#!/usr/bin/env python3
"""Keyboard teleop for Isaac sim dataset collection with controller-like limits."""

from __future__ import annotations

import argparse
import math
import select
import sys
import termios
import time
import tty

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node


HELP = """
Manual Isaac teleop

  a/d   : increase/decrease target yaw rate
  x     : zero yaw rate
  w     : resume auto-speed, or increase manual target speed
  s     : decrease manual target speed
  m     : toggle auto-speed/manual-speed
  space : stop
  q     : quit
"""


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def ramp(current: float, target: float, limit_per_s: float, dt: float) -> float:
    delta = clamp(target - current, -limit_per_s * dt, limit_per_s * dt)
    return current + delta


class ManualTeleop(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("manual_teleop_cmd_vel")
        self.publisher = self.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.max_speed = float(args.max_speed_mps)
        self.max_reverse_speed = float(args.max_reverse_speed_mps)
        self.max_yaw_rate = float(args.max_yaw_rate_radps)
        self.max_accel = float(args.max_accel_mps2)
        self.max_decel = float(args.max_decel_mps2)
        self.max_angular_accel = float(args.max_angular_accel_radps2)
        self.auto_speed = bool(args.auto_speed)
        self.auto_drive_enabled = bool(args.auto_speed)
        self.min_auto_speed = float(args.min_auto_speed_mps)
        self.turn_slowdown = float(args.turn_slowdown)
        self.speed_step = float(args.speed_step_mps)
        self.yaw_step = float(args.yaw_step_radps)
        self.publish_hz = float(args.publish_hz)
        self.target_speed = self.max_speed if self.auto_speed else 0.0
        self.target_yaw_rate = 0.0
        self.current_speed = 0.0
        self.current_yaw_rate = 0.0
        self.last_time = time.monotonic()
        self.timer = self.create_timer(1.0 / max(self.publish_hz, 1.0), self.tick)

    def handle_key(self, key: str) -> bool:
        if key == "q":
            return False
        if key == "w":
            if self.auto_speed:
                self.auto_drive_enabled = True
            else:
                self.target_speed = clamp(self.target_speed + self.speed_step, -self.max_reverse_speed, self.max_speed)
        elif key == "s":
            self.target_speed = clamp(self.target_speed - self.speed_step, -self.max_reverse_speed, self.max_speed)
        elif key == "a":
            self.target_yaw_rate = clamp(self.target_yaw_rate + self.yaw_step, -self.max_yaw_rate, self.max_yaw_rate)
        elif key == "d":
            self.target_yaw_rate = clamp(self.target_yaw_rate - self.yaw_step, -self.max_yaw_rate, self.max_yaw_rate)
        elif key == "x":
            self.target_yaw_rate = 0.0
        elif key == "m":
            self.auto_speed = not self.auto_speed
            self.auto_drive_enabled = self.auto_speed
            if self.auto_speed:
                self.target_speed = self.max_speed
        elif key == " ":
            self.auto_drive_enabled = False
            self.target_speed = 0.0
            self.target_yaw_rate = 0.0
        self.get_logger().info(
            f"mode={'auto' if self.auto_speed else 'manual'} target speed={self.target_speed:.2f} m/s yaw_rate={self.target_yaw_rate:.2f} rad/s",
            throttle_duration_sec=0.25,
        )
        return True

    def desired_speed(self) -> float:
        if not self.auto_speed:
            return self.target_speed
        if not self.auto_drive_enabled:
            return 0.0
        yaw_fraction = min(abs(self.target_yaw_rate) / max(self.max_yaw_rate, 1e-6), 1.0)
        speed = self.max_speed * (1.0 - self.turn_slowdown * yaw_fraction)
        return clamp(speed, self.min_auto_speed, self.max_speed)

    def tick(self) -> None:
        now = time.monotonic()
        dt = max(now - self.last_time, 1e-3)
        self.last_time = now
        desired_speed = self.desired_speed()
        accel_limit = self.max_accel if abs(desired_speed) > abs(self.current_speed) else self.max_decel
        self.current_speed = ramp(self.current_speed, desired_speed, accel_limit, dt)
        self.current_yaw_rate = ramp(self.current_yaw_rate, self.target_yaw_rate, self.max_angular_accel, dt)
        msg = Twist()
        msg.linear.x = float(self.current_speed)
        msg.angular.z = float(self.current_yaw_rate)
        self.publisher.publish(msg)

    def stop(self) -> None:
        self.target_speed = 0.0
        self.target_yaw_rate = 0.0
        self.auto_drive_enabled = False
        self.current_speed = 0.0
        self.current_yaw_rate = 0.0
        self.publisher.publish(Twist())


def read_key(timeout_s: float) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return None
    return sys.stdin.read(1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--max-speed-mps", type=float, default=0.30)
    parser.add_argument("--max-reverse-speed-mps", type=float, default=0.0)
    parser.add_argument("--max-yaw-rate-radps", type=float, default=0.45)
    parser.add_argument("--max-accel-mps2", type=float, default=0.25)
    parser.add_argument("--max-decel-mps2", type=float, default=0.35)
    parser.add_argument("--max-angular-accel-radps2", type=float, default=0.60)
    parser.add_argument("--auto-speed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-auto-speed-mps", type=float, default=0.10)
    parser.add_argument("--turn-slowdown", type=float, default=0.70)
    parser.add_argument("--speed-step-mps", type=float, default=0.05)
    parser.add_argument("--yaw-step-radps", type=float, default=0.08)
    parser.add_argument("--publish-hz", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rclpy.init()
    node = ManualTeleop(args)
    old_settings = termios.tcgetattr(sys.stdin)
    print(HELP, flush=True)
    try:
        tty.setcbreak(sys.stdin.fileno())
        running = True
        while rclpy.ok() and running:
            rclpy.spin_once(node, timeout_sec=0.02)
            key = read_key(0.0)
            if key:
                running = node.handle_key(key)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
