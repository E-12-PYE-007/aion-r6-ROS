#!/usr/bin/env python3
"""Project YOLO weed detections into persistent local-map weed tracks."""

import csv
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class Detection:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str = 'weed'
    score: float = 0.0
    stamp_sec: int = 0
    stamp_nanosec: int = 0

    @property
    def width(self):
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self):
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self):
        return self.width * self.height


@dataclass
class DepthStats:
    pixel_u: float
    pixel_v: float
    depth_m: float
    depth_mean_m: float
    depth_std_m: float
    depth_min_m: float
    depth_max_m: float
    valid_samples: int
    total_samples: int
    inlier_samples: int
    valid_ratio: float
    depth_percentile: float


@dataclass
class WeedMeasurement:
    x: float
    y: float
    z: float
    x_robot: float
    y_robot: float
    z_robot: float
    depth: DepthStats
    odom_x: float
    odom_y: float
    odom_yaw: float
    linear_x: float
    angular_z: float
    depth_stamp_sec: float
    depth_image_width: int
    depth_image_height: int
    odom_stamp_sec: float
    odom_age_sec: float


@dataclass
class WeedTrack:
    track_id: int
    label: str
    x: float
    y: float
    z: float
    score: float
    observations: int
    first_seen_sec: float
    last_seen_sec: float
    last_update_weight: float
    last_match_distance: float
    required_observations: int


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def parse_detections(payload):
    data = json.loads(payload)
    if isinstance(data, dict):
        data = data.get('detections', [])

    detections = []
    for item in data:
        bbox = item.get('bbox') or item.get('xyxy')
        if bbox is None and all(k in item for k in ('xmin', 'ymin', 'xmax', 'ymax')):
            bbox = [item['xmin'], item['ymin'], item['xmax'], item['ymax']]
        if bbox is None or len(bbox) != 4:
            continue

        stamp = item.get('stamp', {}) if isinstance(item.get('stamp', {}), dict) else {}
        label = str(item.get('class', item.get('label', 'weed')))
        score = float(item.get('score', item.get('confidence', 0.0)))
        detections.append(Detection(
            xmin=float(bbox[0]),
            ymin=float(bbox[1]),
            xmax=float(bbox[2]),
            ymax=float(bbox[3]),
            label=label,
            score=score,
            stamp_sec=int(stamp.get('sec', 0)),
            stamp_nanosec=int(stamp.get('nanosec', 0)),
        ))
    return detections


def depth_image_to_array(msg):
    if msg.encoding == '16UC1':
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) * 0.001
    if msg.encoding == '32FC1':
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    raise ValueError(f"Unsupported depth encoding: {msg.encoding}")


class WeedGeotagger(Node):
    def __init__(self):
        super().__init__('weed_geotagger')

        self.detections_topic = self.declare_parameter('detections_topic', '/weed_detections_json').value
        self.depth_topic = self.declare_parameter('depth_topic', '/camera/depth/image_raw').value
        self.camera_info_topic = self.declare_parameter('camera_info_topic', '/camera/color/camera_info').value
        self.odom_topic = self.declare_parameter('odom_topic', '/odometry/filtered').value
        self.output_csv = self.declare_parameter('output_csv', '').value
        self.target_label = self.declare_parameter('target_label', '').value
        self.min_score = float(self.declare_parameter('min_score', 0.0).value)
        self.max_depth_m = float(self.declare_parameter('max_depth_m', 8.0).value)

        self.bbox_shrink_ratio = float(self.declare_parameter('bbox_shrink_ratio', 0.2).value)
        self.max_depth_samples = int(self.declare_parameter('max_depth_samples', 800).value)
        self.min_depth_samples = int(self.declare_parameter('min_depth_samples', 20).value)
        self.depth_mad_scale = float(self.declare_parameter('depth_mad_scale', 2.5).value)
        self.foreground_depth_percentile = float(self.declare_parameter('foreground_depth_percentile', 20.0).value)
        self.foreground_depth_tolerance_m = float(self.declare_parameter('foreground_depth_tolerance_m', 0.6).value)
        self.merge_radius_m = float(self.declare_parameter('merge_radius_m', 0.35).value)
        self.turn_tracking_angular_z = float(self.declare_parameter('turn_tracking_angular_z', 0.15).value)
        self.turning_merge_radius_m = float(self.declare_parameter('turning_merge_radius_m', 0.75).value)
        self.new_track_max_angular_z = float(self.declare_parameter('new_track_max_angular_z', 999.0).value)
        self.new_track_turn_cooldown_sec = float(self.declare_parameter('new_track_turn_cooldown_sec', 0.0).value)
        self.min_confirmed_observations = int(self.declare_parameter('min_confirmed_observations', 3).value)
        self.turning_min_confirmed_observations = int(
            self.declare_parameter('turning_min_confirmed_observations', 6).value
        )
        self.track_merge_radius_m = float(self.declare_parameter('track_merge_radius_m', 0.25).value)
        self.odom_buffer_sec = float(self.declare_parameter('odom_buffer_sec', 10.0).value)
        self.depth_buffer_sec = float(self.declare_parameter('depth_buffer_sec', 10.0).value)
        self.max_sync_age_sec = float(self.declare_parameter('max_sync_age_sec', 0.1).value)
        self.track_min_update_weight = float(self.declare_parameter('track_min_update_weight', 0.10).value)
        self.track_max_update_weight = float(self.declare_parameter('track_max_update_weight', 0.45).value)
        self.angular_slowdown_start = float(self.declare_parameter('angular_slowdown_start', 0.2).value)
        self.angular_slowdown_stop = float(self.declare_parameter('angular_slowdown_stop', 0.8).value)

        self.camera_x = float(self.declare_parameter('camera_x', 0.2).value)
        self.camera_y = float(self.declare_parameter('camera_y', 0.0).value)
        self.camera_z = float(self.declare_parameter('camera_z', 0.235).value)
        self.camera_yaw = float(self.declare_parameter('camera_yaw', 0.0).value)

        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_odom = None
        self.depth_buffer = deque()
        self.odom_buffer = deque()
        self.last_turn_time_sec = None
        self.next_track_id = 1
        self.tracks = []
        self.csv_file = None
        self.csv_writer = None

        if self.output_csv:
            output_path = Path(self.output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = output_path.open('a', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            if output_path.stat().st_size == 0:
                self.csv_writer.writerow([
                    'stamp_sec', 'detection_stamp_sec', 'event', 'track_id',
                    'label', 'score', 'update_weight', 'match_distance_m',
                    'observations', 'required_observations', 'confirmed',
                    'raw_weed_x', 'raw_weed_y', 'raw_weed_z',
                    'track_weed_x', 'track_weed_y', 'track_weed_z',
                    'odom_x', 'odom_y', 'odom_yaw', 'linear_x', 'angular_z',
                    'x_robot', 'y_robot', 'z_robot',
                    'depth_m', 'depth_mean_m', 'depth_std_m', 'depth_min_m', 'depth_max_m',
                    'depth_valid_samples', 'depth_total_samples', 'depth_inlier_samples', 'depth_valid_ratio',
                    'depth_percentile',
                    'pixel_u', 'pixel_v',
                    'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax', 'bbox_width', 'bbox_height', 'bbox_area',
                    'depth_stamp_sec', 'odom_stamp_sec', 'odom_age_sec',
                    'camera_info_width', 'camera_info_height',
                    'depth_image_width', 'depth_image_height',
                ])

        self.create_subscription(String, self.detections_topic, self.detections_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 20)
        self.marker_pub = self.create_publisher(MarkerArray, 'weed_markers', 10)

        self.get_logger().info(
            'Weed geotagger waiting for detections/depth/camera_info/odom: '
            f'{self.detections_topic}, {self.depth_topic}, '
            f'{self.camera_info_topic}, {self.odom_topic}'
        )

    def depth_callback(self, msg):
        try:
            depth_array = depth_image_to_array(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=5.0)
            return

        depth_stamp_sec = stamp_to_sec(msg.header.stamp)
        self.latest_depth = (msg, depth_array)
        self.depth_buffer.append((depth_stamp_sec, msg, depth_array))
        while self.depth_buffer and (depth_stamp_sec - self.depth_buffer[0][0]) > self.depth_buffer_sec:
            self.depth_buffer.popleft()

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def odom_callback(self, msg):
        self.latest_odom = msg
        odom_stamp_sec = stamp_to_sec(msg.header.stamp)
        self.odom_buffer.append((odom_stamp_sec, msg))
        while self.odom_buffer and (odom_stamp_sec - self.odom_buffer[0][0]) > self.odom_buffer_sec:
            self.odom_buffer.popleft()
        if abs(msg.twist.twist.angular.z) > self.turn_tracking_angular_z:
            self.last_turn_time_sec = self.get_clock().now().nanoseconds / 1e9

    def detections_callback(self, msg):
        if self.latest_depth is None or self.latest_camera_info is None or self.latest_odom is None:
            self.get_logger().warn(
                'Missing depth, camera_info, or odom; cannot geotag yet',
                throttle_duration_sec=5.0,
            )
            return

        try:
            detections = parse_detections(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Bad detection JSON: {exc}', throttle_duration_sec=5.0)
            return

        updated_any = False
        for detection in detections:
            if self.target_label and detection.label != self.target_label:
                continue
            if detection.score < self.min_score:
                continue

            odom = self.odom_for_detection(detection)
            depth_sample = self.depth_for_detection(detection)
            if odom is None or depth_sample is None:
                continue

            measurement = self.project_detection(detection, depth_sample, odom)
            if measurement is None:
                continue

            track, event, match_distance = self.update_track(detection, measurement)
            if track is None:
                continue
            self.write_csv(track, event, match_distance, measurement, detection)
            self.merge_nearby_tracks()
            updated_any = True

        if updated_any:
            self.publish_track_markers()

    def odom_for_detection(self, detection):
        if self.latest_odom is None:
            return None
        detection_stamp_sec = detection.stamp_sec + detection.stamp_nanosec * 1e-9
        if detection_stamp_sec <= 0.0 or not self.odom_buffer:
            return self.latest_odom
        odom_stamp_sec, odom = min(
            self.odom_buffer,
            key=lambda item: abs(item[0] - detection_stamp_sec),
        )
        if abs(odom_stamp_sec - detection_stamp_sec) > self.max_sync_age_sec:
            self.get_logger().warn(
                'Skipping detection because closest odom is '
                f'{odom_stamp_sec - detection_stamp_sec:.3f}s from image stamp',
                throttle_duration_sec=2.0,
            )
            return None
        return odom

    def depth_for_detection(self, detection):
        if self.latest_depth is None:
            return None
        detection_stamp_sec = detection.stamp_sec + detection.stamp_nanosec * 1e-9
        if detection_stamp_sec <= 0.0 or not self.depth_buffer:
            return self.latest_depth
        depth_stamp_sec, depth_msg, depth = min(
            self.depth_buffer,
            key=lambda item: abs(item[0] - detection_stamp_sec),
        )
        if abs(depth_stamp_sec - detection_stamp_sec) > self.max_sync_age_sec:
            self.get_logger().warn(
                'Skipping detection because closest depth is '
                f'{depth_stamp_sec - detection_stamp_sec:.3f}s from image stamp',
                throttle_duration_sec=2.0,
            )
            return None
        return depth_msg, depth

    def project_detection(self, detection, depth_sample, odom):
        depth_msg, depth = depth_sample
        camera_info = self.latest_camera_info

        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]
        if fx == 0.0 or fy == 0.0:
            self.get_logger().warn('Camera intrinsics are invalid', throttle_duration_sec=5.0)
            return None

        depth_stats = self.depth_stats_for_bbox(depth, detection, camera_info)
        if depth_stats is None:
            return None

        x_cam_right = (depth_stats.pixel_u - cx) * depth_stats.depth_m / fx
        y_cam_down = (depth_stats.pixel_v - cy) * depth_stats.depth_m / fy
        z_cam_forward = depth_stats.depth_m

        # Optical x is right, while base_link y is left, so the lateral sign flips.
        x_robot = self.camera_x + z_cam_forward * math.cos(self.camera_yaw) + x_cam_right * math.sin(self.camera_yaw)
        y_robot = self.camera_y + z_cam_forward * math.sin(self.camera_yaw) - x_cam_right * math.cos(self.camera_yaw)
        z_robot = self.camera_z - y_cam_down

        odom_pose = odom.pose.pose
        odom_x = odom_pose.position.x
        odom_y = odom_pose.position.y
        odom_yaw = yaw_from_quaternion(odom_pose.orientation)

        weed_x = odom_x + x_robot * math.cos(odom_yaw) - y_robot * math.sin(odom_yaw)
        weed_y = odom_y + x_robot * math.sin(odom_yaw) + y_robot * math.cos(odom_yaw)

        return WeedMeasurement(
            x=weed_x,
            y=weed_y,
            z=z_robot,
            x_robot=x_robot,
            y_robot=y_robot,
            z_robot=z_robot,
            depth=depth_stats,
            odom_x=odom_x,
            odom_y=odom_y,
            odom_yaw=odom_yaw,
            linear_x=odom.twist.twist.linear.x,
            angular_z=odom.twist.twist.angular.z,
            depth_stamp_sec=stamp_to_sec(depth_msg.header.stamp),
            depth_image_width=depth.shape[1],
            depth_image_height=depth.shape[0],
            odom_stamp_sec=stamp_to_sec(odom.header.stamp),
            odom_age_sec=(detection.stamp_sec + detection.stamp_nanosec * 1e-9) - stamp_to_sec(odom.header.stamp),
        )

    def depth_stats_for_bbox(self, depth, detection, camera_info):
        if detection.width <= 1.0 or detection.height <= 1.0:
            return None

        shrink = min(max(self.bbox_shrink_ratio, 0.0), 0.45)
        xmin = detection.xmin + detection.width * shrink
        xmax = detection.xmax - detection.width * shrink
        ymin = detection.ymin + detection.height * shrink
        ymax = detection.ymax - detection.height * shrink

        color_width = float(camera_info.width or depth.shape[1])
        color_height = float(camera_info.height or depth.shape[0])
        scale_x = depth.shape[1] / color_width
        scale_y = depth.shape[0] / color_height

        u0 = max(0, int(math.floor(xmin * scale_x)))
        u1 = min(depth.shape[1], int(math.ceil(xmax * scale_x)))
        v0 = max(0, int(math.floor(ymin * scale_y)))
        v1 = min(depth.shape[0], int(math.ceil(ymax * scale_y)))
        if u0 >= u1 or v0 >= v1:
            return None

        total_pixels = (u1 - u0) * (v1 - v0)
        stride = max(1, int(math.ceil(math.sqrt(total_pixels / max(1, self.max_depth_samples)))))
        ys, xs = np.mgrid[v0:v1:stride, u0:u1:stride]
        samples = depth[ys, xs]
        total_samples = int(samples.size)

        valid_mask = np.isfinite(samples) & (samples > 0.05) & (samples < self.max_depth_m)
        if int(np.count_nonzero(valid_mask)) < self.min_depth_samples:
            return None

        valid_depths = samples[valid_mask]
        valid_xs = xs[valid_mask].astype(np.float32)
        valid_ys = ys[valid_mask].astype(np.float32)

        percentile = min(50.0, max(1.0, self.foreground_depth_percentile))
        foreground_depth = float(np.percentile(valid_depths, percentile))
        foreground_limit = min(
            self.max_depth_m,
            foreground_depth + max(0.0, self.foreground_depth_tolerance_m),
        )
        foreground_mask = valid_depths <= foreground_limit
        valid_depths = valid_depths[foreground_mask]
        valid_xs = valid_xs[foreground_mask]
        valid_ys = valid_ys[foreground_mask]

        if valid_depths.size < self.min_depth_samples:
            return None

        median_depth = float(np.median(valid_depths))
        mad = float(np.median(np.abs(valid_depths - median_depth)))
        if mad > 1e-6:
            inlier_mask = np.abs(valid_depths - median_depth) <= self.depth_mad_scale * 1.4826 * mad
            valid_depths = valid_depths[inlier_mask]
            valid_xs = valid_xs[inlier_mask]
            valid_ys = valid_ys[inlier_mask]

        if valid_depths.size < self.min_depth_samples:
            return None

        color_us = valid_xs / scale_x
        color_vs = valid_ys / scale_y
        return DepthStats(
            pixel_u=float(np.mean(color_us)),
            pixel_v=float(np.mean(color_vs)),
            depth_m=foreground_depth,
            depth_mean_m=float(np.mean(valid_depths)),
            depth_std_m=float(np.std(valid_depths)),
            depth_min_m=float(np.min(valid_depths)),
            depth_max_m=float(np.max(valid_depths)),
            valid_samples=int(np.count_nonzero(valid_mask)),
            total_samples=total_samples,
            inlier_samples=int(valid_depths.size),
            valid_ratio=float(valid_depths.size) / float(max(1, total_samples)),
            depth_percentile=percentile,
        )

    def update_track(self, detection, measurement):
        best_track = None
        best_distance = None
        for track in self.tracks:
            if track.label != detection.label:
                continue
            distance = math.hypot(track.x - measurement.x, track.y - measurement.y)
            if best_distance is None or distance < best_distance:
                best_track = track
                best_distance = distance

        now_sec = self.get_clock().now().nanoseconds / 1e9
        turning = self.is_turning(measurement)
        match_radius = self.turning_merge_radius_m if turning else self.merge_radius_m
        if best_track is None or best_distance is None or best_distance > match_radius:
            if self.should_suppress_new_track(measurement, now_sec):
                return None, 'ignored_turn_new', math.nan

            required_observations = (
                self.turning_min_confirmed_observations if turning
                else self.min_confirmed_observations
            )
            track = WeedTrack(
                track_id=self.next_track_id,
                label=detection.label,
                x=measurement.x,
                y=measurement.y,
                z=measurement.z,
                score=detection.score,
                observations=1,
                first_seen_sec=now_sec,
                last_seen_sec=now_sec,
                last_update_weight=1.0,
                last_match_distance=math.nan,
                required_observations=required_observations,
            )
            self.next_track_id += 1
            self.tracks.append(track)
            event = 'confirmed_new' if self.is_confirmed(track) else 'candidate_new'
            return track, event, math.nan

        weight = self.measurement_update_weight(detection, measurement)
        was_confirmed = self.is_confirmed(best_track)
        if turning and not was_confirmed:
            best_track.required_observations = max(
                best_track.required_observations,
                self.turning_min_confirmed_observations,
            )
        best_track.x = (1.0 - weight) * best_track.x + weight * measurement.x
        best_track.y = (1.0 - weight) * best_track.y + weight * measurement.y
        best_track.z = (1.0 - weight) * best_track.z + weight * measurement.z
        best_track.score = max(best_track.score, detection.score)
        best_track.observations += 1
        best_track.last_seen_sec = now_sec
        best_track.last_update_weight = weight
        best_track.last_match_distance = best_distance
        if not was_confirmed and self.is_confirmed(best_track):
            event = 'confirmed'
        elif self.is_confirmed(best_track):
            event = 'update'
        else:
            event = 'candidate_update'
        return best_track, event, best_distance

    def is_confirmed(self, track):
        return track.observations >= track.required_observations

    def is_turning(self, measurement):
        return abs(measurement.angular_z) > self.turn_tracking_angular_z

    def should_suppress_new_track(self, measurement, now_sec):
        if self.new_track_max_angular_z < 999.0 and abs(measurement.angular_z) > self.new_track_max_angular_z:
            return True
        if self.last_turn_time_sec is None:
            return False
        return (now_sec - self.last_turn_time_sec) < self.new_track_turn_cooldown_sec

    def merge_nearby_tracks(self):
        merged = True
        while merged:
            merged = False
            for i, first in enumerate(list(self.tracks)):
                for second in list(self.tracks)[i + 1:]:
                    if first.label != second.label:
                        continue
                    distance = math.hypot(first.x - second.x, first.y - second.y)
                    if distance > self.track_merge_radius_m:
                        continue

                    keep, drop = (
                        (first, second)
                        if (first.observations, -first.track_id) >= (second.observations, -second.track_id)
                        else (second, first)
                    )
                    total = keep.observations + drop.observations
                    keep.x = (keep.x * keep.observations + drop.x * drop.observations) / total
                    keep.y = (keep.y * keep.observations + drop.y * drop.observations) / total
                    keep.z = (keep.z * keep.observations + drop.z * drop.observations) / total
                    keep.score = max(keep.score, drop.score)
                    keep.observations = total
                    keep.first_seen_sec = min(keep.first_seen_sec, drop.first_seen_sec)
                    keep.last_seen_sec = max(keep.last_seen_sec, drop.last_seen_sec)
                    keep.required_observations = min(
                        keep.required_observations,
                        drop.required_observations,
                    )
                    self.tracks.remove(drop)
                    merged = True
                    break
                if merged:
                    break

    def measurement_update_weight(self, detection, measurement):
        score_quality = min(1.0, max(0.05, detection.score))
        depth_quality = min(1.0, max(0.05, measurement.depth.valid_ratio * 2.0))
        distance_quality = min(1.0, max(0.2, 1.0 - measurement.depth.depth_m / max(0.1, self.max_depth_m)))
        turn_quality = self.turn_quality(abs(measurement.angular_z))
        quality = score_quality * depth_quality * distance_quality * turn_quality
        return self.track_min_update_weight + quality * (self.track_max_update_weight - self.track_min_update_weight)

    def turn_quality(self, angular_speed):
        if angular_speed <= self.angular_slowdown_start:
            return 1.0
        if angular_speed >= self.angular_slowdown_stop:
            return 0.5
        span = max(1e-6, self.angular_slowdown_stop - self.angular_slowdown_start)
        ratio = (angular_speed - self.angular_slowdown_start) / span
        return 1.0 - 0.5 * ratio

    def publish_track_markers(self):
        markers = MarkerArray()
        for track in self.tracks:
            if not self.is_confirmed(track):
                continue
            markers.markers.append(self.make_marker(track))
        if markers.markers:
            self.marker_pub.publish(markers)

    def make_marker(self, track):
        marker = Marker()
        marker.header.frame_id = 'odom'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'weeds'
        marker.id = track.track_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(x=float(track.x), y=float(track.y), z=float(track.z))
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 0.1
        marker.color.a = 0.9
        marker.lifetime.sec = 0
        return marker

    def write_csv(self, track, event, match_distance, measurement, detection):
        if self.csv_writer is None:
            return
        camera_info = self.latest_camera_info
        self.csv_writer.writerow([
            self.get_clock().now().nanoseconds / 1e9,
            detection.stamp_sec + detection.stamp_nanosec * 1e-9,
            event,
            track.track_id,
            detection.label,
            detection.score,
            track.last_update_weight,
            match_distance,
            track.observations,
            track.required_observations,
            self.is_confirmed(track),
            measurement.x,
            measurement.y,
            measurement.z,
            track.x,
            track.y,
            track.z,
            measurement.odom_x,
            measurement.odom_y,
            measurement.odom_yaw,
            measurement.linear_x,
            measurement.angular_z,
            measurement.x_robot,
            measurement.y_robot,
            measurement.z_robot,
            measurement.depth.depth_m,
            measurement.depth.depth_mean_m,
            measurement.depth.depth_std_m,
            measurement.depth.depth_min_m,
            measurement.depth.depth_max_m,
            measurement.depth.valid_samples,
            measurement.depth.total_samples,
            measurement.depth.inlier_samples,
            measurement.depth.valid_ratio,
            measurement.depth.depth_percentile,
            measurement.depth.pixel_u,
            measurement.depth.pixel_v,
            detection.xmin,
            detection.ymin,
            detection.xmax,
            detection.ymax,
            detection.width,
            detection.height,
            detection.area,
            measurement.depth_stamp_sec,
            measurement.odom_stamp_sec,
            measurement.odom_age_sec,
            camera_info.width,
            camera_info.height,
            measurement.depth_image_width,
            measurement.depth_image_height,
        ])
        self.csv_file.flush()

    def destroy_node(self):
        if self.csv_file is not None:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WeedGeotagger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
