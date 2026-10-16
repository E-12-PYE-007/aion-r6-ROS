"""Global localisation: /odometry/local + GPS -> /odometry/global.

Runs on top of local_localisation.launch.py and mavros.

Arguments: use_sim_time.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CONFIG_DIR = os.path.join(get_package_share_directory('localisation'), 'config')
GLOBAL_EKF_CONFIG_FILE = os.path.join(CONFIG_DIR, 'global_ekf.yaml')
NAVSAT_CONFIG_FILE = os.path.join(CONFIG_DIR, 'navsat_transform_node.yaml')


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    global_ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_global',
        output='screen',
        parameters=[
            GLOBAL_EKF_CONFIG_FILE,
            {
                'use_sim_time': use_sim_time,
            },
        ],
        remappings=[
            ('odometry/filtered', '/odometry/global'),
        ],
    )

    navsat_transform_node = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform_node',
        output='screen',
        parameters=[
            NAVSAT_CONFIG_FILE,
            {'use_sim_time': use_sim_time},
        ],
        remappings=[
            ('imu', '/mavros_fcu/mavros_fcu/data'),
            ('gps/fix', '/mavros_fcu/mavros_fcu/raw/fix'),
            ('odometry/filtered', '/odometry/local'),
            ('odometry/gps', '/odometry/gps'),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock',
        ),
        global_ekf_node,
        navsat_transform_node,
    ])
