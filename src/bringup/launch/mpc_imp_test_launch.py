"""Bring up the mpc_path_follower test chain:

sim_robot -> /odom, /tf (odom->base_link) -> mpc_path_follower, chunk_generator
chunk_generator -> /vla/action_chunk -> mpc_path_follower
static_esdf_publisher -> /nvblox_node/static_map_slice -> mpc_path_follower
mpc_path_follower -> cmd_vel -> sim_robot
mpc_plotter listens to /odom, /vla/action_chunk, cmd_vel, and the ESDF; renders an mp4 on shutdown.

`sim_robot`, `chunk_generator`, and `static_esdf_publisher` stand in for real hardware, the
real VLA model, and a live nvblox, respectively, until those exist - swap any of them out
independently, nothing else in the graph depends on how they're implemented, only on the
topics/frames they produce.

Robot starts at (2.0, -3.5) heading +x - chosen (checked directly against the bag) so the
straight reference line crosses a real obstacle around x=4.1-4.4, not just open space.
`static_esdf_publisher`'s bag is recorded at 0.05m resolution, not the 0.1m production
default in constants.py - mpc_path_follower's patch_resolution parameter is overridden here
to match it for this test only.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DEFAULT_BAG_PATH = (
    '/home/djstrahan/aion-r6-vla-master/aion-r6-ROS/'
    'mpc_dev_artefacts/rosbags/esdf_single_obs/esdf_single_obs_0.db3'
)


def generate_launch_description():
    output_path_arg = DeclareLaunchArgument(
        'output_path',
        default_value='mpc_plotter.mp4',
        description='Where mpc_plotter saves the animation on shutdown',
    )
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value=DEFAULT_BAG_PATH,
        description='Rosbag to pull one static DistanceMapSlice message from',
    )

    return LaunchDescription([
        output_path_arg,
        bag_path_arg,
        Node(
            package='debug',
            executable='sim_robot',
            name='sim_robot',
            output='screen',
            parameters=[{'initial_x': 2.0, 'initial_y': -3.5, 'initial_theta': 0.0}],
        ),
        Node(
            package='debug',
            executable='chunk_generator',
            name='chunk_generator',
            output='screen',
        ),
        Node(
            package='debug',
            executable='static_esdf_publisher',
            name='static_esdf_publisher',
            output='screen',
            parameters=[{'bag_path': LaunchConfiguration('bag_path')}],
        ),
        Node(
            package='control',
            executable='mpc_path_follower',
            name='mpc_path_follower',
            output='screen',
            parameters=[{'patch_resolution': 0.05}],
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
