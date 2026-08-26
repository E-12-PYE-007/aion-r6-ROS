"""Bring up the full action-chunk-following chain:

simulate_action_chunk -> pure_pursuit_controller -> cmd_vel_to_roboclaw -> roboclaw_for_motors
                                    ^                                              |
                                    +---------------- encoder_localisation <-------+

`simulate_action_chunk` stands in for the VLA model until that's wired up.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pattern_arg = DeclareLaunchArgument(
        'pattern',
        default_value='straight',
        description='Fake action-chunk motion pattern: straight, left_arc, right_arc, s_curve, stop',
    )

    return LaunchDescription([
        pattern_arg,
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
        Node(
            package='debug',
            executable='simulate_action_chunk',
            name='simulate_action_chunk',
            output='screen',
            parameters=[{'pattern': LaunchConfiguration('pattern')}],
        ),
    ])
