from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    """
    Bring up the front Gemini 336 with colour and depth, depth registered to colour.

    /camera/depth/image_raw is the depth image aligned to the colour viewpoint (software
    alignment), in camera_color_optical_frame; the original is on /camera/depth/image_unaligned.

    Also publishes the base_link -> camera_link mounting transform: 0.2m forward, 0m left,
    0.235m up, mounted level and forward-facing (identity rotation).
    """
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
            'depth_registration': 'true',
            'align_mode': 'SW',
            'align_target_stream': 'COLOR',
            'publish_tf': 'true',
            'enable_laser': 'true',
            'depth_fps': '30',
        }.items(),
    )

    return LaunchDescription([base_to_camera_tf, gemini_launch])
