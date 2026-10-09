"""Depth-enabled Gemini 330-series camera bringup for weed geotagging.

This keeps the existing data-collection RGB-only launch untouched. Weed
geotagging needs depth and a stable camera-to-robot transform so detections can
be projected from image pixels into the robot/map frame.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    base_to_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_camera_link',
        arguments=[
            '--x', '0.2', '--y', '0', '--z', '0.235',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_link',
        ],
    )

    gemini_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('orbbec_camera'),
                'launch',
                'gemini_330_series.launch.py',
            ])
        ),
        launch_arguments={
            'camera_name': 'camera',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_point_cloud': 'true',
            'publish_tf': 'true',
            'enable_laser': 'true',
            'depth_fps': '30',
        }.items(),
    )

    return LaunchDescription([
        base_to_camera_tf,
        gemini_launch,
    ])
