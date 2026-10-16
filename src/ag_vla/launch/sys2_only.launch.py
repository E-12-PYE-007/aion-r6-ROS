#!/usr/bin/env python3
"""AgVLA sys2 (backbone VLA) — workstation side of a split deployment.

Publishes /agvla/hidden_state for sys1 on the robot to consume.
Both machines must be on the same network with a matching ROS_DOMAIN_ID.

    ros2 launch ag_vla sys2_only.launch.py
    ros2 launch ag_vla sys2_only.launch.py goal:="Find the red door"

Pair with (on the robot):
    ros2 launch ag_vla ag_vla.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

VLA_PATH = "/home/vla-cap/AsyncVLA/AsyncVLA_release"


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "goal",
            default_value="Go to the black chair",
            description="Language goal for the VLA.",
        ),
        SetEnvironmentVariable("TF_CPP_MIN_LOG_LEVEL", "3"),
        SetEnvironmentVariable("TF_ENABLE_ONEDNN_OPTS", "0"),
        Node(
            package="ag_vla",
            executable="sys2_dsk",
            name="sys2",
            output="screen",
            parameters=[
                {"vla_path": VLA_PATH},
                {"goal": LaunchConfiguration("goal")},
            ],
        ),
    ])
