"""CBF safety layer.

  /odometry/local + /agvla/action_chunk + nvblox ESDF -> cbf_safety_layer -> cmd_vel

Arguments: patch_size, patch_resolution, print_solve_time.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from control.safety_layer.constants import PATCH_SIZE, PATCH_RESOLUTION


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('patch_size', default_value=str(PATCH_SIZE),
                              description='ESDF patch size [cells per side, even]'),
        DeclareLaunchArgument('patch_resolution', default_value=str(PATCH_RESOLUTION),
                              description='ESDF patch cell size [m]'),
        DeclareLaunchArgument('print_solve_time', default_value='false',
                              description='Log the CBF-QP solve time every tick'),
        TimerAction(period=2.0, actions=[
            Node(
                package='control',
                executable='cbf_safety_layer',
                name='cbf_safety_layer',
                output='screen',
                parameters=[{
                    'patch_size': ParameterValue(LaunchConfiguration('patch_size'), value_type=int),
                    'patch_resolution': ParameterValue(LaunchConfiguration('patch_resolution'), value_type=float),
                    'print_solve_time': ParameterValue(LaunchConfiguration('print_solve_time'), value_type=bool),
                }],
            ),
        ]),
    ])
