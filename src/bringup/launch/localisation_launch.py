import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    ekf_config_file = os.path.join(
        get_package_share_directory('localisation'),
        'config',
        'local_ekf.yaml',
    )

    encoder_node = Node(
        package='localisation',
        executable='encoder_localisation',
        name='encoder_localisation',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            ekf_config_file,
            {
                'use_sim_time': use_sim_time,
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock',
        ),
        encoder_node,
        ekf_node,
    ])