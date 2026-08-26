#!/usr/bin/env python3
"""Pure-pursuit controller node for the Aion R6.

Subscribes to /odom (nav_msgs/Odometry) for the current pose and to
/vla/action_chunk (aion_msgs/ActionChunk) for the target motion, and
publishes body velocity commands on cmd_vel (geometry_msgs/Twist).
"""

import math

import rclpy
from aion_msgs.msg import ActionChunk
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

MAX_LINEAR_VELOCITY = 0.3   # [m/s] Shared rover speed limit
MAX_ANGULAR_VELOCITY = 0.3  # [rad/s] Shared rover yaw rate limit

LOOKAHEAD_DISTANCE = 0.2  # [m]
CONTROL_PERIOD_SEC = 1.0 / 30.0  # Control rate of 30Hz
HERMITE_SAMPLES_PER_SEGMENT = 10


def sign(value):
    return -1.0 if value < 0.0 else 1.0


def saturate(linear_vel, angular_vel):
    """Combined linear/angular saturation: scales both together (preserving
    turning radius) rather than clamping each independently, so a command
    that needs to turn sharply doesn't get its curvature distorted by
    clamping yaw rate alone.
    """
    if abs(linear_vel) <= MAX_LINEAR_VELOCITY:
        if abs(angular_vel) <= MAX_ANGULAR_VELOCITY:
            return linear_vel, angular_vel
        rd = linear_vel / angular_vel
        return (
            MAX_ANGULAR_VELOCITY * sign(linear_vel) * abs(rd),
            MAX_ANGULAR_VELOCITY * sign(angular_vel),
        )

    if abs(angular_vel) <= 0.001:
        return MAX_LINEAR_VELOCITY * sign(linear_vel), 0.0

    rd = linear_vel / angular_vel
    if abs(rd) >= MAX_LINEAR_VELOCITY / MAX_ANGULAR_VELOCITY:
        return (
            MAX_LINEAR_VELOCITY * sign(linear_vel),
            MAX_LINEAR_VELOCITY * sign(angular_vel) / abs(rd),
        )
    return (
        MAX_ANGULAR_VELOCITY * sign(linear_vel) * abs(rd),
        MAX_ANGULAR_VELOCITY * sign(angular_vel),
    )


def generate_waypoints(relative_poses):
    """Interpolate a Hermite curve between consecutive action-chunk waypoints
    so heading (theta) shapes the path, not just position.
    """
    waypoints = []
    for i in range(len(relative_poses) - 1):
        x0, y0, theta0 = relative_poses[i].x, relative_poses[i].y, relative_poses[i].theta
        x1, y1, theta1 = relative_poses[i + 1].x, relative_poses[i + 1].y, relative_poses[i + 1].theta

        # Scale factor based on distance between points to define curve shape
        scale = math.hypot(x1 - x0, y1 - y0)

        # Scaled tangent vectors at start and end, derived from heading
        m0x, m0y = scale * math.cos(theta0), scale * math.sin(theta0)
        m1x, m1y = scale * math.cos(theta1), scale * math.sin(theta1)

        # Sample segment 10 times and store points
        for j in range(HERMITE_SAMPLES_PER_SEGMENT):
            t = j / HERMITE_SAMPLES_PER_SEGMENT
            t2, t3 = t * t, t * t * t

            # Hermite basis polynomials
            h00 = 2 * t3 - 3 * t2 + 1  # 1 at t=0, 0 at t=1
            h10 = t3 - 2 * t2 + t      # tangent influence at t=0
            h01 = -2 * t3 + 3 * t2     # 0 at t=0, 1 at t=1
            h11 = t3 - t2              # tangent influence at t=1

            waypoints.append((
                h00 * x0 + h10 * m0x + h01 * x1 + h11 * m1x,
                h00 * y0 + h10 * m0y + h01 * y1 + h11 * m1y,
            ))
    return waypoints


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PurePursuitControllerNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')

        self._current_pose = None          # (x, y, theta), latest /odom sample
        self._current_action_chunk = None  # latest ActionChunk message

        self._anchor_pose = None      # current_pose captured when _last_chunk_seq was set
        self._waypoints = []          # Hermite-interpolated path for the current chunk, in x,y
        self._waypoint_idx = 0        # Index of current lookahead point
        self._last_chunk_seq = None   # None until the first chunk is received

        self._odom_subscription = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self._action_chunk_subscription = self.create_subscription(
            ActionChunk, '/vla/action_chunk', self.action_chunk_callback, 10)
        self._cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self._control_timer = self.create_timer(CONTROL_PERIOD_SEC, self.control_loop)

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        self._current_pose = (position.x, position.y, yaw_from_quaternion(msg.pose.pose.orientation))

    def action_chunk_callback(self, msg):
        self._current_action_chunk = msg

    def control_loop(self):
        if self._current_pose is None or self._current_action_chunk is None:
            self.get_logger().warn('No pose/action chunk received yet, skipping tick')
            return

        linear_vel, angular_vel = self.compute_command(self._current_action_chunk, self._current_pose)

        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self._cmd_vel_publisher.publish(cmd)

    def compute_command(self, action_chunk, current_pose):
        if self._last_chunk_seq is None or action_chunk.seq_num != self._last_chunk_seq:
            # Anchor to current_pose here since the message doesn't carry the pose it was
            # actually conditioned on - approximates away VLA inference/transport latency.
            self._anchor_pose = current_pose
            self._waypoints = generate_waypoints(action_chunk.relative_poses)
            self._waypoint_idx = 0
            self._last_chunk_seq = action_chunk.seq_num

        anchor_x, anchor_y, anchor_theta = self._anchor_pose
        current_x, current_y, current_theta = current_pose

        # Move origin of current pose to be relative to anchor pose, then rotate
        # the displacement vector into the anchor frame.
        dx = current_x - anchor_x
        dy = current_y - anchor_y
        relative_x = dx * math.cos(anchor_theta) + dy * math.sin(anchor_theta)
        relative_y = -dx * math.sin(anchor_theta) + dy * math.cos(anchor_theta)

        lookahead_x = 0.0
        lookahead_y = 0.0
        euclid_dist = 0.0
        found_lookahead = False

        for i in range(self._waypoint_idx, len(self._waypoints)):
            point_x, point_y = self._waypoints[i]
            lookahead_x = point_x - relative_x
            lookahead_y = point_y - relative_y
            euclid_dist = math.hypot(lookahead_x, lookahead_y)
            if euclid_dist > LOOKAHEAD_DISTANCE:
                self._waypoint_idx = i
                found_lookahead = True
                break

        if not found_lookahead:
            # Ran off the end of the chunk without finding a point past the lookahead
            # distance - stop rather than keep chasing the final waypoint.
            return 0.0, 0.0

        # Rotate the lookahead vector from the anchor frame into the robot's live body frame.
        delta_theta = current_theta - anchor_theta
        body_x = lookahead_x * math.cos(delta_theta) + lookahead_y * math.sin(delta_theta)
        body_y = -lookahead_x * math.sin(delta_theta) + lookahead_y * math.cos(delta_theta)

        curvature = 2.0 * body_y / (euclid_dist * euclid_dist)  # Curvature to approach path on

        return saturate(MAX_LINEAR_VELOCITY, curvature * MAX_LINEAR_VELOCITY)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
