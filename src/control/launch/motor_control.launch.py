"""Motor control: cmd_vel in, wheel velocity commands out.

  cmd_vel -> cmd_vel_to_roboclaw -> set_motor_velocity -> roboclaw_for_motors -> Roboclaw

roboclaw_for_motors is the only node that opens the Roboclaw's serial port, so nothing else
may start it.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='control',
            executable='cmd_vel_to_roboclaw',
            name='cmd_vel_to_roboclaw',
            output='screen',
        ),
        Node(
            package='control',
            executable='roboclaw_for_motors',
            name='roboclaw_for_motors',
            output='screen',
        ),
    ])
