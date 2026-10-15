"""The event display bridge. Run it on the workstation, then open http://localhost:<port>.

Arguments: jetson_host, jetson_fallback_host (e.g. the Jetson's hotspot address), port.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('jetson_host', default_value='10.43.254.23',
                              description='Address of the Jetson'),
        DeclareLaunchArgument('jetson_fallback_host', default_value='',
                              description="Address to try if the first does not answer, e.g. the Jetson's hotspot"),
        DeclareLaunchArgument('port', default_value='8080', description='Port the page is served on'),
        Node(
            package='endeavour_display',
            executable='bridge',
            name='display_bridge',
            output='screen',
            parameters=[{
                'jetson_host': LaunchConfiguration('jetson_host'),
                'jetson_fallback_host': LaunchConfiguration('jetson_fallback_host'),
                'port': ParameterValue(LaunchConfiguration('port'), value_type=int),
            }],
        ),
    ])
