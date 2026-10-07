"""Bring up the rover for real action-chunk-following (no simulated input):

camera (Gemini 336) ---------------------------------------+
                                                             v
    [/vla/action_chunk] -> pure_pursuit_controller -> cmd_vel_to_roboclaw -> roboclaw_for_motors
                                    ^                                              |
                                    +------------------------ ekf_filter_node_local <--- encoder_localisation
                                                                       ^
                                                                mavros (IMU)

Unlike pp-roboclaw-test.py, this does not launch simulate_action_chunk --
/vla/action_chunk must be published by something else (e.g. ag_vla's sys1).

mavros is included because the EKF config (local_ekf_wheel_imu.yaml) fuses
IMU data from /mavros_fcu/mavros_fcu/data_raw alongside wheel odometry.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

EKF_CONFIG_FILE = os.path.join(
        get_package_share_directory('localisation'),
        'config',
        'local_ekf_wheel_imu.yaml',
    )


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('localisation'),
                'launch',
                'camera.launch.py',
            )
        )
    )

    mavros_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('bringup'),
                'launch',
                'mavros-test.py',
            )
        )
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_local',
        output='screen',
        parameters=[
            EKF_CONFIG_FILE,
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
        camera_launch,
        mavros_launch,
        Node(
            package='control',
            executable='roboclaw_for_motors',
            name='roboclaw_for_motors',
            output='screen',
        ),
        Node(
            package='localisation',
            executable='encoder_localisation',
            name='encoder_localisation',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
            }],
        ),
        ekf_node,
        Node(
            package='control',
            executable='cmd_vel_to_roboclaw',
            name='cmd_vel_to_roboclaw',
            output='screen',
        ),
        Node(
            package='control',
            executable='pure_pursuit_controller',
            name='pure_pursuit_controller',
            output='screen',
        ),
    ])
