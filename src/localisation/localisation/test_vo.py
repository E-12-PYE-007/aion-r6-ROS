#!/usr/bin/env python3
"""Fixed open-loop test maneuvers for visual-odometry evaluation.

Drives either a straight line of a set distance or a turn on the spot through a
set angle, at a fixed speed, then stops. Both are open-loop timed rather than
closed on odometry, so the rover's actual displacement should be measured
externally -- the point is a repeatable motion to compare VO against, not a
precisely metered one.

    ros2 launch localisation test_vo.launch.py mode:=straight
    ros2 launch localisation test_vo.launch.py mode:=rotate
"""

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

PUBLISH_RATE_HZ = 20.0
STOP_REPEATS = 5


class TestVoNode(Node):
    def __init__(self):
        super().__init__('test_vo')

        self.declare_parameter('mode', 'straight')
        self.declare_parameter('speed', 0.3)
        self.declare_parameter('distance', 1.0)
        self.declare_parameter('angle_deg', 360.0)

        mode = self.get_parameter('mode').value
        speed = self.get_parameter('speed').value

        if speed <= 0.0:
            raise ValueError(f'speed must be positive, got {speed}')

        self.command = Twist()
        if mode == 'straight':
            target = self.get_parameter('distance').value
            self.command.linear.x = speed
            units = 'm'
        elif mode == 'rotate':
            target = math.radians(self.get_parameter('angle_deg').value)
            self.command.angular.z = speed
            units = 'rad'
        else:
            raise ValueError(f"mode must be 'straight' or 'rotate', got '{mode}'")

        self.duration = target / speed
        self.finished = False

        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self.tick)

        self.get_logger().info(
            f'{mode}: {target:.3f}{units} at {speed:.2f}{units}/s '
            f'-> driving for {self.duration:.2f}s'
        )

    def tick(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed >= self.duration:
            self.timer.cancel()
            self.stop()
            self.finished = True
            self.get_logger().info(f'maneuver complete after {elapsed:.2f}s')
            return
        self.publisher.publish(self.command)

    def stop(self):
        for _ in range(STOP_REPEATS):
            self.publisher.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    try:
        node = TestVoNode()
    except ValueError as exc:
        print(f'test_vo: {exc}')
        rclpy.shutdown()
        return

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
