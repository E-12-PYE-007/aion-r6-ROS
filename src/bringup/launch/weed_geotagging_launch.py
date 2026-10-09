"""Launch the weed geotagging node with configurable topic names.

Use this after starting camera/localisation, or alongside
weed_geotagging_bringup_launch.py in another terminal while developing.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('weed_geotagging'),
                    'launch',
                    'weed_geotagging_launch.py',
                )
            )
        )
    ])
