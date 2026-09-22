"""Robot-side bringup for teleop-driven episode data collection. Run on the
Jetson, then use `ros2 run data_collection collection_interface` in a separate
attached SSH terminal to drive with arrow keys and start/stop episodes.

cmd_vel_to_roboclaw converts /cmd_vel into roboclaw commands.
local_localisation_launch.py brings up roboclaw_for_motors (the other half
of motor control), mavros, wheel-encoder localisation, and the EKF -- it
owns roboclaw_for_motors itself, so it's included here rather than launching
a second, conflicting instance of that node.
camera_launch.py brings up the RGB-only camera stream.
episode_data_collector logs camera + odometry into named episodes.

Does not launch collection_interface (the start/stop/prompt keyboard client)
-- it needs a real attached terminal, which ros2 launch doesn't reliably
provide. Run it by hand after this launch is up.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    base_dir_arg = DeclareLaunchArgument(
        'base_dir',
        description='Directory episodes are written under, e.g. /path/to/trajectories',
    )
    cam_topic_arg = DeclareLaunchArgument(
        'cam_topic',
        default_value='/camera/color/image_raw',
        description='Camera Image topic to log',
    )
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/odometry/filtered',
        description='Odometry topic to log (default assumes local_localisation_launch.py, no GPS)',
    )

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('bringup'),
                'launch',
                'camera_launch.py',
            )
        )
    )

    localisation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('bringup'),
                'launch',
                'local_localisation_launch.py',
            )
        )
    )

    episode_data_collector_node = Node(
        package='data_collection',
        executable='episode_data_collector',
        name='episode_data_collector',
        output='screen',
        parameters=[{
            'base_dir': LaunchConfiguration('base_dir'),
            'cam_topic': LaunchConfiguration('cam_topic'),
            'odom_topic': LaunchConfiguration('odom_topic'),
        }],
    )

    cmd_vel_to_roboclaw_node = Node(
        package='control',
        executable='cmd_vel_to_roboclaw',
        name='cmd_vel_to_roboclaw',
        output='screen',
    )

    return LaunchDescription([
        base_dir_arg,
        cam_topic_arg,
        odom_topic_arg,
        cmd_vel_to_roboclaw_node,
        localisation_launch,
        camera_launch,
        episode_data_collector_node,
    ])
