#!/usr/bin/env python3
"""A stand-in for the rover stack. aion_system.sh runs it in the end-to-end test; it publishes what the display reads.

target_classes:= and weed:= in AION_LAUNCH_ARGS decide which weeds it reports, as the real launch would.
"""
import math
import os
import time

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray

ALL_CLASSES = ['plant1', 'plant2', 'plant3', 'plant4']
RADIUS_M, OMEGA = 0.4, 0.5


class FakeRobot(Node):
    def __init__(self):
        super().__init__('fake_robot')
        args = os.environ.get('AION_LAUNCH_ARGS', '').split()
        classes = ALL_CLASSES
        for arg in args:
            if arg.startswith('target_classes:='):
                classes = [c for c in arg.split(':=', 1)[1].split(',') if c]
        self.weed_enabled = 'weed:=false' not in args
        self.tracks = []
        for i, label in enumerate(ALL_CLASSES):
            if label in classes:
                for k in range(2):
                    angle = (i * 2 + k) * 0.7
                    self.tracks.append((len(self.tracks), label, 0.25 * math.cos(angle) + RADIUS_M * 0.5, 0.25 * math.sin(angle)))
        self.t0 = time.monotonic()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.odom = self.create_publisher(Odometry, '/odometry/local', 10)
        self.diag = self.create_publisher(DiagnosticArray, '/system_diagnostics', 10)
        self.filtering = self.create_publisher(Bool, '/safety_layer/filtering_active', 10)
        self.reversing = self.create_publisher(Bool, '/safety_layer/reversing', 10)
        self.markers = self.create_publisher(MarkerArray, '/weed_markers', latched)
        self.image = self.create_publisher(CompressedImage, '/weed_detections/image/compressed', 10)
        self.create_timer(0.05, self.publish_odometry)
        self.create_timer(1.0, self.publish_diagnostics)
        self.create_timer(1 / 15, self.publish_flags)
        if self.weed_enabled:
            self.create_timer(1.0, self.publish_markers)
            self.create_timer(0.1, self.publish_image)
            self.publish_markers()

    def t(self):
        return time.monotonic() - self.t0

    def publish_odometry(self):
        msg = Odometry()
        msg.header.frame_id = 'odom'
        a = OMEGA * self.t()
        msg.pose.pose.position.x = RADIUS_M * math.cos(a)
        msg.pose.pose.position.y = RADIUS_M * math.sin(a)
        yaw = a + math.pi / 2
        msg.pose.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.pose.orientation.w = math.cos(yaw / 2)
        self.odom.publish(msg)

    def publish_diagnostics(self):
        self.diag.publish(DiagnosticArray())

    def publish_flags(self):
        phase = self.t() % 8.0
        self.filtering.publish(Bool(data=3.0 <= phase < 6.5))
        self.reversing.publish(Bool(data=5.0 <= phase < 6.5))

    def publish_markers(self):
        out = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        out.markers.append(clear)
        for track_id, label, x, y in self.tracks:
            sphere = Marker()
            sphere.ns, sphere.id, sphere.type, sphere.action = 'weeds', track_id, Marker.SPHERE, Marker.ADD
            sphere.header.frame_id = 'odom'
            sphere.pose.position.x, sphere.pose.position.y = x, y
            text = Marker()
            text.ns, text.id, text.type, text.action = 'weed_labels', track_id, Marker.TEXT_VIEW_FACING, Marker.ADD
            text.header.frame_id = 'odom'
            text.pose.position.x, text.pose.position.y = x, y
            text.text = f'{label} | ID {track_id}'
            out.markers += [sphere, text]
        self.markers.publish(out)

    def publish_image(self):
        frame = np.full((360, 640, 3), (236, 227, 207), dtype=np.uint8)
        x = int(60 + 440 * (0.5 + 0.5 * math.sin(self.t())))
        cv2.rectangle(frame, (x, 120), (x + 120, 260), (110, 140, 255), 3)
        ok, jpeg = cv2.imencode('.jpg', frame)
        msg = CompressedImage()
        msg.format = 'jpeg'
        msg.data = jpeg.tobytes()
        self.image.publish(msg)


def main():
    rclpy.init()
    node = FakeRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
