"""Drive on VLA action chunks through the CBF safety layer.

  local localisation (roboclaw driver, encoders, local EKF) -> /odometry/filtered
  /vla/action_chunk + nvblox ESDF -> cbf_safety_layer -> cmd_vel -> motor control

nvblox and the VLA run separately.

Arguments: use_sim_time, patch_size, patch_resolution, print_solve_time.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from control.safety_layer.constants import PATCH_SIZE, PATCH_RESOLUTION

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

    cbf_safety_layer_node = Node(
        package='control',
        executable='cbf_safety_layer',
        name='cbf_safety_layer',
        output='screen',
        parameters=[{
            'patch_size': ParameterValue(LaunchConfiguration('patch_size'), value_type=int),
            'patch_resolution': ParameterValue(LaunchConfiguration('patch_resolution'), value_type=float),
            'print_solve_time': ParameterValue(LaunchConfiguration('print_solve_time'), value_type=bool),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use simulation clock'),
        DeclareLaunchArgument('patch_size', default_value=str(PATCH_SIZE),
                              description='ESDF patch size [cells per side, even]'),
        DeclareLaunchArgument('patch_resolution', default_value=str(PATCH_RESOLUTION),
                              description='ESDF patch cell size [m]'),
        DeclareLaunchArgument('print_solve_time', default_value='false',
                              description='Log the CBF-QP solve time every tick'),
        local_localisation_launch,
        motor_control_launch,
        cbf_safety_layer_node,
    ])
