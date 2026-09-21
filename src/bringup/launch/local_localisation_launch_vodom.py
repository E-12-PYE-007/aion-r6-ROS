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
        'local_ekf_wheel_imu_vodom.yaml',
    )


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    roboclaw_node = Node(
        package='control',
        executable='roboclaw_for_motors',
        name='roboclaw_for_motors',
        output='screen',
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

    encoder_node = Node(
        package='localisation',
        executable='encoder_localisation',
        name='encoder_localisation',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
    )

    vo_pose_relay_node = Node(
        package='localisation',
        executable='vo_pose_relay',
        name='vo_pose_relay',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'vo_topic': '/visual_slam/tracking/vo_pose_covariance',
            'reference_topic': '/odometry/wheel',
            'output_topic': '/odometry/vo_pose_odom',
            'output_frame': 'odom',
        }],
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
        roboclaw_node,
        mavros_launch,
        encoder_node,
        vo_pose_relay_node,
        ekf_node,
    ])