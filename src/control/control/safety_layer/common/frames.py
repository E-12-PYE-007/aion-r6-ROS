"""Planar pose and angle helpers."""
import math

import numpy as np
from geometry_msgs.msg import Pose
from tf2_geometry_msgs import do_transform_pose


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def transform_pose_2d(x, y, theta, tf):
    """Apply a TransformStamped to a planar (x, y, theta) pose."""
    pose_in = Pose()
    pose_in.position.x = x
    pose_in.position.y = y
    pose_in.orientation.z = math.sin(theta / 2.0)
    pose_in.orientation.w = math.cos(theta / 2.0)

    pose_out = do_transform_pose(pose_in, tf)

    return pose_out.position.x, pose_out.position.y, yaw_from_quaternion(pose_out.orientation)
