#!/usr/bin/env python3
"""AgVLA sys1 (edge adapter) — rover side of a split deployment.

Starts cam_seq_bridge, which numbers the Orbbec frames onto /cam, and sys1, which pairs
those frames with /agvla/hidden_state from the workstation running sys2 and publishes
/agvla/action_chunk for pure_pursuit_controller.

Both machines must be on the same network with a matching ROS_DOMAIN_ID.

    ros2 launch ag_vla ag_vla.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="ag_vla",
            executable="cam_seq_bridge",
            name="cam_seq_bridge",
            output="screen",
        ),
        Node(
            package="ag_vla",
            executable="sys1_hw",
            name="sys1",
            output="screen",
        ),
    ])
