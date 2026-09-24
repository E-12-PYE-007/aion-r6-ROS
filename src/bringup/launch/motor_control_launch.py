"""Motor control: cmd_vel in, wheel velocity commands out.

  cmd_vel -> cmd_vel_to_roboclaw -> set_motor_velocity -> roboclaw_for_motors -> Roboclaw

cmd_vel_to_roboclaw always starts. roboclaw_for_motors, the only node that opens the Roboclaw's
serial port, starts only with start_driver:=true. local_localisation_launch.py and
global_localisation_launch.py already start it, and a second driver on the same port would
send conflicting commands and read the encoders twice.

Arguments: start_driver (default false).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'start_driver', default_value='false',
            description='Also start roboclaw_for_motors (leave false if a localisation launch is running)'),
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
            condition=IfCondition(LaunchConfiguration('start_driver')),
        ),
    ])
