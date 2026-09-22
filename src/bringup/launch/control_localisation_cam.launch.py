"""Bring up the rover for real action-chunk-following (no simulated input):

camera (Gemini 336) ---------------------------------------+
                                                             v
    [/vla/action_chunk] -> pure_pursuit_controller -> cmd_vel_to_roboclaw -> roboclaw_for_motors
                                    ^                                              |
                                    +---------------- encoder_localisation <-------+

Unlike pp-roboclaw-test.py, this does not launch simulate_action_chunk --
/vla/action_chunk must be published by something else (e.g. ag_vla's sys1).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('localisation'),
                'launch',
                'camera.launch.py',
            )
        )
    )

    return LaunchDescription([
        camera_launch,
        Node(
            package='control',
            executable='roboclaw_for_motors',
            name='roboclaw_for_motors',
            output='screen',
        ),
        Node(
            package='localisation',
            executable='encoder_localisation',
            name='encoder_localisation',
            output='screen',
        ),
        Node(
            package='control',
            executable='cmd_vel_to_roboclaw',
            name='cmd_vel_to_roboclaw',
            output='screen',
        ),
        Node(
            package='control',
            executable='pure_pursuit_controller',
            name='pure_pursuit_controller',
            output='screen',
        ),
    ])
