"""Open the desktop weed map without starting robot hardware or inference."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(get_package_share_directory('weed_geotagging'), 'rviz', 'weed_map.rviz')
    return LaunchDescription([
        DeclareLaunchArgument('odom_topic', default_value='/odometry/local'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=config),
        DeclareLaunchArgument('rover_length_m', default_value='0.65'),
        DeclareLaunchArgument('rover_width_m', default_value='0.50'),
        DeclareLaunchArgument('rover_height_m', default_value='0.15'),
        DeclareLaunchArgument('path_spacing_m', default_value='0.03'),
        DeclareLaunchArgument('max_path_points', default_value='10000'),
        Node(
            package='weed_geotagging', executable='rover_visualiser',
            name='rover_visualiser', output='screen',
            parameters=[{
                'odom_topic': LaunchConfiguration('odom_topic'),
                'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
                **{name: ParameterValue(LaunchConfiguration(name), value_type=float) for name in (
                    'rover_length_m', 'rover_width_m', 'rover_height_m', 'path_spacing_m',
                )},
                'max_path_points': ParameterValue(LaunchConfiguration('max_path_points'), value_type=int),
            }],
        ),
        Node(
            package='rviz2', executable='rviz2', name='weed_map', output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{
                'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
            }],
            condition=IfCondition(LaunchConfiguration('start_rviz')),
        ),
    ])
