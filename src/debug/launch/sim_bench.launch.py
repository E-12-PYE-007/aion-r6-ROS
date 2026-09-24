"""Bench-test sim: sim_robot (unicycle + CSV log) and static_esdf_publisher (bag ESDF).

Starts no controller and no chunk generator - publish cmd_vel yourself, or launch a safety
layer alongside. Shuts everything down when sim_robot finishes (max_duration reached, 30 s by default).

Arguments: dx, dy, dtheta (start-pose delta from the default start), goal_distance,
controller (log label), output_path, max_duration, bag_path.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

DEFAULT_BAG_PATH = (
    '/home/djstrahan/aion-r6-vla-master/aion-r6-ROS/safety_layer_bench_testing/'
    'rosbags/esdf_single_obs/esdf_single_obs_0.db3'
)


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument('dx', default_value='0.0', description='Start x delta from the default start [m]'),
        DeclareLaunchArgument('dy', default_value='0.0', description='Start y delta from the default start [m]'),
        DeclareLaunchArgument('dtheta', default_value='0.0', description='Start heading delta from the default start [rad]'),
        DeclareLaunchArgument('goal_distance', default_value='6.0', description='Written to the log header only [m]'),
        DeclareLaunchArgument('controller', default_value='unknown', description="Log label, e.g. 'cbf' or 'mpc'"),
        DeclareLaunchArgument('output_path', default_value='', description='CSV path; default sim_logs/run_<timestamp>.csv'),
        DeclareLaunchArgument('max_duration', default_value='30.0', description='Stop after this many seconds; 0 runs until stopped'),
        DeclareLaunchArgument('bag_path', default_value=DEFAULT_BAG_PATH, description='Rosbag holding the ESDF slice to publish'),
    ]

    sim_robot = Node(
        package='debug',
        executable='sim_robot',
        name='sim_robot',
        output='screen',
        parameters=[{
            'dx': ParameterValue(LaunchConfiguration('dx'), value_type=float),
            'dy': ParameterValue(LaunchConfiguration('dy'), value_type=float),
            'dtheta': ParameterValue(LaunchConfiguration('dtheta'), value_type=float),
            'goal_distance': ParameterValue(LaunchConfiguration('goal_distance'), value_type=float),
            'max_duration': ParameterValue(LaunchConfiguration('max_duration'), value_type=float),
            'controller': ParameterValue(LaunchConfiguration('controller'), value_type=str),
            'output_path': ParameterValue(LaunchConfiguration('output_path'), value_type=str),
            'esdf_bag': ParameterValue(LaunchConfiguration('bag_path'), value_type=str),
        }],
    )

    static_esdf_publisher = Node(
        package='debug',
        executable='static_esdf_publisher',
        name='static_esdf_publisher',
        output='screen',
        parameters=[{'bag_path': ParameterValue(LaunchConfiguration('bag_path'), value_type=str)}],
    )

    shutdown_when_sim_finishes = RegisterEventHandler(
        OnProcessExit(target_action=sim_robot, on_exit=[EmitEvent(event=Shutdown())]))

    return LaunchDescription(arguments + [sim_robot, static_esdf_publisher, shutdown_when_sim_finishes])
