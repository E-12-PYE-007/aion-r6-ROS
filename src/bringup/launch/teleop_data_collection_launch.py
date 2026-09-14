"""Bring up teleop-driven episode data collection:

joy_node -> teleop_twist_joy -> cmd_vel_to_roboclaw -> roboclaw_for_motors
episode_data_collector logs camera + odometry into named episodes.

Does not launch episode_recorder (the start/stop keyboard client) -- it needs
a real attached terminal, which ros2 launch doesn't reliably provide. Run it
by hand in its own terminal: `ros2 run data_collection episode_recorder`.

Also does not bring up the camera driver or localisation/EKF chain -- run
those separately for now.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    base_dir_arg = DeclareLaunchArgument(
        'base_dir',
        description='Directory episodes are written under, e.g. /path/to/trajectories',
    )
    cam_topic_arg = DeclareLaunchArgument(
        'cam_topic',
        default_value='/camera/color/image_raw',
        description='Camera Image topic to log',
    )
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/odometry/filtered',
        description='Odometry topic to log (default assumes local_localisation_launch.py, no GPS)',
    )
    max_linear_speed_arg = DeclareLaunchArgument(
        'max_linear_speed',
        default_value='0.3',
        description='Max linear speed in m/s (teleop_twist_joy scale_linear.x) -- conservative default, tune before real runs',
    )
    max_yaw_rate_arg = DeclareLaunchArgument(
        'max_yaw_rate',
        default_value='0.3',
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
            'enable_button': 2,        # left trigger (deadman)
            'enable_turbo_button': -1, # disabled -- no turbo tier
            'require_enable_button': True,
            'scale_linear.x': ParameterValue(LaunchConfiguration('max_linear_speed'), value_type=float),
            'scale_angular.yaw': ParameterValue(LaunchConfiguration('max_yaw_rate'), value_type=float),
        }],
    )

    episode_data_collector_node = Node(
        package='data_collection',
        executable='episode_data_collector',
        name='episode_data_collector',
        output='screen',
        parameters=[{
            'base_dir': LaunchConfiguration('base_dir'),
            'cam_topic': LaunchConfiguration('cam_topic'),
            'odom_topic': LaunchConfiguration('odom_topic'),
        }],
    )

    cmd_vel_to_roboclaw_node = Node(
        package='control',
        executable='cmd_vel_to_roboclaw',
        name='cmd_vel_to_roboclaw',
        output='screen',
    )

    roboclaw_for_motors_node = Node(
        package='control',
        executable='roboclaw_for_motors',
        name='roboclaw_for_motors',
        output='screen',
    )

    return LaunchDescription([
        base_dir_arg,
        cam_topic_arg,
        odom_topic_arg,
        max_linear_speed_arg,
        max_yaw_rate_arg,
        joy_node,
        teleop_twist_joy_node,
        cmd_vel_to_roboclaw_node,
        roboclaw_for_motors_node,
        episode_data_collector_node,
    ])
