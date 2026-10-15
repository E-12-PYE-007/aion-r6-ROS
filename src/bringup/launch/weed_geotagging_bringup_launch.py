"""Bring up the rover stack needed for weed geotagging.

This launch starts the RGB-registered depth Gemini camera, the existing local EKF
localisation stack, and the weed_geotagging package's combined YOLO + geotagger
launch. Joystick teleop is still started from the workstation with
teleop_launch.py.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def include_bringup_launch(filename):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('bringup'),
                'launch',
                filename,
            )
        )
    )


def generate_launch_description():
    camera_x = LaunchConfiguration('camera_x')
    camera_y = LaunchConfiguration('camera_y')
    camera_z = LaunchConfiguration('camera_z')
    camera_yaw = LaunchConfiguration('camera_yaw')

    base_to_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_camera_link',
        arguments=[
            '--x', camera_x,
            '--y', camera_y,
            '--z', camera_z,
            '--roll', '0',
            '--pitch', '0',
            '--yaw', camera_yaw,
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_link',
        ],
    )

    depth_camera = IncludeLaunchDescription(
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
            # RGB bounding boxes must sample depth in the colour camera viewpoint.
            'depth_registration': 'true',
            'align_mode': 'SW',
            'align_target_stream': 'COLOR',
            'enable_point_cloud': 'true',
            'publish_tf': 'true',
            'enable_laser': 'true',
            'depth_fps': '30',
        }.items(),
    )

    cmd_vel_converter = Node(
        package='control',
        executable='cmd_vel_to_roboclaw',
        name='cmd_vel_to_roboclaw',
        output='screen',
    )

    weed_detection = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('weed_geotagging'),
                'launch',
                'weed_detection_geotagging_launch.py',
            )
        ),
        launch_arguments={
            'model_path': LaunchConfiguration('model_path'),
            'output_csv': LaunchConfiguration('output_csv'),
            'output_odom_csv': LaunchConfiguration('output_odom_csv'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'publish_annotated': LaunchConfiguration('publish_annotated'),
            'max_depth_m': LaunchConfiguration('max_depth_m'),
            'foreground_depth_percentile': LaunchConfiguration('foreground_depth_percentile'),
            'foreground_depth_tolerance_m': LaunchConfiguration('foreground_depth_tolerance_m'),
            'merge_radius_m': LaunchConfiguration('merge_radius_m'),
            'turning_merge_radius_m': LaunchConfiguration('turning_merge_radius_m'),
            'track_merge_radius_m': LaunchConfiguration('track_merge_radius_m'),
            'max_sync_age_sec': LaunchConfiguration('max_sync_age_sec'),
            'camera_x': camera_x,
            'camera_y': camera_y,
            'camera_z': camera_z,
            'camera_yaw': camera_yaw,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('output_csv', default_value=''),
        DeclareLaunchArgument('output_odom_csv', default_value=''),
        DeclareLaunchArgument('odom_topic', default_value='/odometry/filtered'),
        DeclareLaunchArgument('publish_annotated', default_value='false'),
        DeclareLaunchArgument('max_depth_m', default_value='3.0'),
        DeclareLaunchArgument('foreground_depth_percentile', default_value='40.0'),
        DeclareLaunchArgument('foreground_depth_tolerance_m', default_value='0.4'),
        DeclareLaunchArgument('merge_radius_m', default_value='0.35'),
        DeclareLaunchArgument('turning_merge_radius_m', default_value='0.9'),
        DeclareLaunchArgument('track_merge_radius_m', default_value='0.5'),
        DeclareLaunchArgument('max_sync_age_sec', default_value='0.1'),
        DeclareLaunchArgument('camera_x', default_value='0.2'),
        DeclareLaunchArgument('camera_y', default_value='0.0'),
        DeclareLaunchArgument('camera_z', default_value='0.235'),
        DeclareLaunchArgument('camera_yaw', default_value='0.0'),
        depth_camera,
        base_to_camera_tf,
        include_bringup_launch('local_localisation_launch.py'),
        cmd_vel_converter,
        weed_detection,
    ])
