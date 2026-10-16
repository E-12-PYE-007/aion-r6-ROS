"""Pure pursuit controller.

  /odometry/local + /agvla/action_chunk -> pure_pursuit_controller -> cmd_vel
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='control',
            executable='pure_pursuit_controller',
            name='pure_pursuit_controller',
            output='screen',
        ),
    ])
