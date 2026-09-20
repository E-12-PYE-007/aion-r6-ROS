from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """
    Run a fixed test maneuver for VO evaluation.

    mode:=straight  drives `distance` metres forward
    mode:=rotate    turns `angle_deg` degrees on the spot

    Assumes cmd_vel_to_roboclaw and roboclaw_for_motors are already running.
    """
    return LaunchDescription([
        DeclareLaunchArgument(
            'mode',
            default_value='straight',
            description="'straight' or 'rotate'",
        ),
        DeclareLaunchArgument('speed', default_value='0.3'),
        DeclareLaunchArgument('distance', default_value='1.0'),
        DeclareLaunchArgument('angle_deg', default_value='360.0'),
        Node(
            package='localisation',
            executable='test_vo',
            name='test_vo',
            output='screen',
            parameters=[{
                'mode': ParameterValue(LaunchConfiguration('mode'), value_type=str),
                'speed': ParameterValue(LaunchConfiguration('speed'), value_type=float),
                'distance': ParameterValue(LaunchConfiguration('distance'), value_type=float),
                'angle_deg': ParameterValue(LaunchConfiguration('angle_deg'), value_type=float),
            }],
        ),
    ])
