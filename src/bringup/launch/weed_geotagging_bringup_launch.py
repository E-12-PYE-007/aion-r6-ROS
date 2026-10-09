"""Bring up robot localisation plus depth camera streams for weed geotagging.

This launch prepares the inputs a geotagging node will need:
- depth/color camera streams and camera TF
- wheel/IMU localisation, with optional visual-odom fusion if Isaac ROS VSLAM
  is publishing `/visual_slam/tracking/vo_pose_covariance`

It does not launch the YOLO detector or weed geotagging node yet.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def include_bringup_launch(filename):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('bringup'),
                'launch',
                filename,
            )
        )
    )


def generate_launch_description():
    return LaunchDescription([
        include_bringup_launch('weed_geotagging_camera_launch.py'),
        include_bringup_launch('local_localisation_launch_vodom.py'),
    ])
