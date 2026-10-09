"""Drive on VLA action chunks with pure pursuit only (no safety layer).

  local localisation (roboclaw driver, encoders, local EKF) -> /odometry/filtered
  /agvla/action_chunk -> pure_pursuit_controller -> cmd_vel -> motor control

The VLA runs separately.

Arguments: use_sim_time.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

BRINGUP_LAUNCH_DIR = os.path.join(get_package_share_directory('bringup'), 'launch')


def generate_launch_description():
    local_localisation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(BRINGUP_LAUNCH_DIR, 'local_localisation_launch.py')),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
    )

    motor_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(BRINGUP_LAUNCH_DIR, 'motor_control_launch.py')),
        launch_arguments={'start_driver': 'false'}.items(),  # local localisation starts the driver
    )

    pure_pursuit_node = Node(
        package='control',
        executable='pure_pursuit_controller',
        name='pure_pursuit_controller',
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use simulation clock'),
        local_localisation_launch,
        motor_control_launch,
        pure_pursuit_node,
    ])
