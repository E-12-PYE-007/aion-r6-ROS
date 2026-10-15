"""Rover base: everything needed to drive and have local odometry.

  mavros (IMU) + motor control (Roboclaw driver, cmd_vel converter) + local localisation -> /odometry/local

Arguments: use_sim_time.
"""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def include(package, launch_file, **launch_arguments):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            get_package_share_directory(package) + '/launch/' + launch_file),
        launch_arguments=launch_arguments.items(),
    )


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use simulation clock'),
        include('localisation', 'mavros.launch.py'),
        include('control', 'motor_control.launch.py'),
        include('localisation', 'local_localisation.launch.py',
                use_sim_time=LaunchConfiguration('use_sim_time')),
    ])
