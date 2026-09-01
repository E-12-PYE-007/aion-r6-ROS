import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('bringup')
    plugin_yaml = os.path.join(pkg_share, 'config', 'mavros_pluginlists.yaml')
    config_yaml = os.path.join(pkg_share, 'config', 'mavros_config.yaml')

    return LaunchDescription([
        Node(
            package='mavros',
            executable='mavros_node',
            name='mavros_fcu',
            output='screen',
            parameters=[
                {'fcu_url': '/dev/pixhawk:57600'},
                {'gcs_url': ''},               # no GCS bridge on this link
                {'target_system_id': 1},
                {'target_component_id': 1},
                plugin_yaml,
                config_yaml,
            ],
        ),
    ])