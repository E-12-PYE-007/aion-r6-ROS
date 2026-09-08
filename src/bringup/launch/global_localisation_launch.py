import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

LOCAL_EKF_CONFIG_FILE = os.path.join(
        get_package_share_directory('localisation'),
        'config',
        'local_ekf.yaml',
    )

GLOBAL_EKF_CONFIG_FILE = os.path.join(
        get_package_share_directory('localisation'),
        'config',
        'global_ekf.yaml',
    )

NAVSAT_CONFIG_FILE = os.path.join(
        get_package_share_directory('bringup'),
        'config',
        'navsat_transform_node.yaml',
    )

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    
    encoder_node = Node(
        package='localisation',
        executable='encoder_localisation',
        name='encoder_localisation',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
    )

    local_ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_local',
        output='screen',
        parameters=[
            LOCAL_EKF_CONFIG_FILE,
            {
                'use_sim_time': use_sim_time,
            },
        ],
        remappings=[
            ('odometry/filtered', '/odometry/local'),
        ],
    )

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

    navsat_node = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform',
        output='screen',
        parameters=[
            NAVSAT_CONFIG_FILE,
            {'use_sim_time': use_sim_time},
        ],
        remappings=[
            ('imu', '/mavros_fcu/mavros_fcu/data'),
            ('gps/fix',
            '/mavros_fcu/mavros_fcu/global_position/raw/fix'),
            ('odometry/filtered', '/odometry/global'),
            ('odometry/gps', '/odometry/gps'),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
        ),
        encoder_node,
        local_ekf_node,
        global_ekf_node,
        navsat_node,
    ])