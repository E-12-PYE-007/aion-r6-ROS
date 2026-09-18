"""Bring up the mpc_path_follower test chain:

sim_robot -> /odom, /tf (odom->base_link) -> mpc_path_follower, chunk_generator
chunk_generator -> /vla/action_chunk -> mpc_path_follower
mpc_path_follower -> cmd_vel -> sim_robot
mpc_plotter listens to /odom, /vla/action_chunk, and cmd_vel; renders an mp4 on shutdown.

`sim_robot` and `chunk_generator` stand in for real hardware and the real VLA model until
those exist - swap either out independently, nothing else in the graph depends on how
they're implemented, only on the topics/frames they produce.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    output_path_arg = DeclareLaunchArgument(
        'output_path',
        default_value='mpc_plotter.mp4',
        description='Where mpc_plotter saves the animation on shutdown',
    )

    return LaunchDescription([
        output_path_arg,
        Node(
            package='debug',
            executable='sim_robot',
            name='sim_robot',
            output='screen',
        ),
        Node(
            package='debug',
            executable='chunk_generator',
            name='chunk_generator',
            output='screen',
        ),
        Node(
            package='control',
            executable='mpc_path_follower',
            name='mpc_path_follower',
            output='screen',
        ),
        Node(
            package='debug',
            executable='mpc_plotter',
            name='mpc_plotter',
            output='screen',
            parameters=[{'output_path': LaunchConfiguration('output_path')}],
            # save_animation() renders + encodes the whole run on shutdown, which for a
            # multi-minute run can easily exceed launch's default 5s/10s SIGINT/SIGTERM grace -
            # without this, a long run gets its animation SIGKILLed mid-write and left corrupt.
            sigterm_timeout='60',
            sigkill_timeout='15',
        ),
    ])
