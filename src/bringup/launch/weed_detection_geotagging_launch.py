"""Launch the YOLO weed detector and local-EKF geotagger.

Start camera/localisation first, or run this alongside
weed_geotagging_bringup_launch.py while developing.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('output_csv', default_value=''),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('weed_geotagging'),
                    'launch',
                    'weed_detection_geotagging_launch.py',
                )
            ),
            launch_arguments={
                'model_path': LaunchConfiguration('model_path'),
                'output_csv': LaunchConfiguration('output_csv'),
            }.items(),
        )
    ])
