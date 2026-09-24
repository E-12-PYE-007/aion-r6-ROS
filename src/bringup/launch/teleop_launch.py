"""Standalone arrow-key teleop using key_teleop.

For real data-collection runs, prefer:

    ros2 run data_collection collection_interface

That combined interface drives with the same arrow-key velocity model while
also starting/stopping episodes in one terminal. This launch file is kept as
a simple drive-only test wrapper around the standard key_teleop package.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    forward_rate_arg = DeclareLaunchArgument(
        'forward_rate',
        default_value='0.4',
        description='Forward speed in m/s for the up arrow',
    )
    backward_rate_arg = DeclareLaunchArgument(
        'backward_rate',
        default_value='0.4',
        description='Reverse speed in m/s for the down arrow',
    )
    rotation_rate_arg = DeclareLaunchArgument(
        'rotation_rate',
        default_value='0.4',
        description='Yaw rate in rad/s for left/right arrows',
    )
    hz_arg = DeclareLaunchArgument(
        'hz',
        default_value='10.0',
        description='Command publish rate in Hz',
    )

    key_teleop_node = Node(
        package='key_teleop',
        executable='key_teleop',
        name='key_teleop',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'twist_stamped_enabled': False,
            'forward_rate': ParameterValue(LaunchConfiguration('forward_rate'), value_type=float),
            'backward_rate': ParameterValue(LaunchConfiguration('backward_rate'), value_type=float),
            'rotation_rate': ParameterValue(LaunchConfiguration('rotation_rate'), value_type=float),
            'hz': ParameterValue(LaunchConfiguration('hz'), value_type=float),
        }],
        remappings=[
            ('key_vel', '/cmd_vel'),
        ],
    )

    return LaunchDescription([
        forward_rate_arg,
        backward_rate_arg,
        rotation_rate_arg,
        hz_arg,
        key_teleop_node,
    ])
