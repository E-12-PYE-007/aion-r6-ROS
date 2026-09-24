"""Launches the MPC safety layer: VLA action chunks in, cmd_vel out.

Arguments: patch_size, patch_resolution (ESDF patch geometry).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from control.safety_layer.common.constants import PATCH_SIZE, PATCH_RESOLUTION


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('patch_size', default_value=str(PATCH_SIZE),
                              description='ESDF patch size [cells per side, even]'),
        DeclareLaunchArgument('patch_resolution', default_value=str(PATCH_RESOLUTION),
                              description='ESDF patch cell size [m]'),
        Node(
            package='control',
            executable='mpc_safety_layer',
            name='mpc_safety_layer',
            output='screen',
            parameters=[{
                'patch_size': ParameterValue(LaunchConfiguration('patch_size'), value_type=int),
                'patch_resolution': ParameterValue(LaunchConfiguration('patch_resolution'), value_type=float),
            }],
        ),
    ])
