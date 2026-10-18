"""The event display bridge. Run it on the workstation, then open http://localhost:<port>.

It first works out which network it is on. If the Jetson answers over ssh at jetson_host it uses the Fast DDS
discovery server. Otherwise, if it answers at jetson_fallback_host (its hotspot), it unsets ROS_DISCOVERY_SERVER
and ROS_SUPER_CLIENT, as you do by hand on every machine, and tells the Jetson script to do the same.

Arguments: jetson_host, jetson_fallback_host, discovery_server, discovery_mode (auto, server or multicast), port.
"""
import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, LogInfo, OpaqueFunction, SetEnvironmentVariable,
                            UnsetEnvironmentVariable)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from endeavour_display.jetson import ssh_reachable
from endeavour_display.profiles import DEFAULT_SERVER, choose_profile


def start_bridge(context):
    value = lambda name: LaunchConfiguration(name).perform(context)
    profile = choose_profile(value('discovery_mode'), value('jetson_host'), value('jetson_fallback_host'),
                             value('discovery_server'), ssh_reachable, os.environ)
    actions = [LogInfo(msg=f'Display network: {profile.name} (discovery {profile.discovery}), Jetson at {profile.hosts[0]}')]
    for name, setting in profile.environment.items():
        actions.append(UnsetEnvironmentVariable(name) if setting is None else SetEnvironmentVariable(name, setting))
    hosts = [h for h in profile.hosts if h]
    plants_file = os.path.join(get_package_share_directory('endeavour_display'), 'config', 'plants.yaml')
    actions.append(Node(
        package='endeavour_display',
        executable='bridge',
        name='display_bridge',
        output='screen',
        parameters=[{
            'jetson_host': hosts[0] if hosts else '',
            'jetson_fallback_host': hosts[1] if len(hosts) > 1 else '',
            'discovery': profile.discovery,
            'port': int(value('port')),
            'plants_file': plants_file,
        }],
    ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('jetson_host', default_value='10.43.254.23', description='Address of the Jetson on the LAN'),
        DeclareLaunchArgument('jetson_fallback_host', default_value='10.42.0.1',
                              description="The Jetson's address on its hotspot, tried if the first does not answer"),
        DeclareLaunchArgument('discovery_server', default_value=DEFAULT_SERVER,
                              description='Fast DDS discovery server, used on the LAN'),
        DeclareLaunchArgument('discovery_mode', default_value='auto',
                              description='auto: pick by which address answers; server; multicast'),
        DeclareLaunchArgument('port', default_value='8080', description='Port the page is served on'),
        OpaqueFunction(function=start_bridge),
    ])
