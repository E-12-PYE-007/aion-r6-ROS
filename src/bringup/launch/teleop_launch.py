"""Controller teleop machinery, meant for the workstation (joystick attached
here), not the Jetson.

joy_node -> teleop_twist_joy, publishing /cmd_vel for the robot-side bringup
(teleop_data_collection_launch.py) to consume over the network.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    max_linear_speed_arg = DeclareLaunchArgument(
        'max_linear_speed',
        default_value='0.4',
        description='Max linear speed in m/s (teleop_twist_joy scale_linear.x) -- conservative default, tune before real runs',
    )
    max_yaw_rate_arg = DeclareLaunchArgument(
        'max_yaw_rate',
        default_value='0.4',
        description='Max yaw rate in rad/s (teleop_twist_joy scale_angular.yaw)',
    )

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
    )

    teleop_twist_joy_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy',
        output='screen',
        parameters=[{
            # Xbox One controller layout (teleop_twist_joy's xbox.config.yaml preset).
            # Verify with `ros2 topic echo /joy` on first connect -- wired vs Bluetooth
            # enumeration can differ.
            'axis_linear.x': 1,        # left stick vertical
            'axis_angular.yaw': 0,     # left stick horizontal
            'enable_button': 5,        # RB (deadman) -- left trigger didn't register as a button on this controller
            'enable_turbo_button': -1, # disabled -- no turbo tier
            'require_enable_button': False,
            'scale_linear.x': ParameterValue(LaunchConfiguration('max_linear_speed'), value_type=float),
            'scale_angular.yaw': ParameterValue(LaunchConfiguration('max_yaw_rate'), value_type=float),
        }],
    )

    return LaunchDescription([
        max_linear_speed_arg,
        max_yaw_rate_arg,
        joy_node,
        teleop_twist_joy_node,
    ])
