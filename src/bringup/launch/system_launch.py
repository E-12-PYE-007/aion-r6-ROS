"""Unified launch harness for the Aion R6: native rover stack + AgVLA, in one shot.

Brings up the full rover-side system:
  - localisation + motor control + a VLA-chunk controller
  - the AgVLA edge stack (camera, cam_seq_bridge, sys1)

The controller is chosen by the `safety_layer` argument; each option brings its own
localisation + motor control, so exactly one is launched:
  safety_layer:=true  -> safety_layer_launch.py  (pure pursuit nominal + CBF-QP filter)
  safety_layer:=false -> pure_pursuit_launch.py  (pure pursuit only, no filter)

sys2 (the VLA backend) and nvblox run separately; see ag_vla.launch.py and the safety layer.

    ros2 launch bringup system_launch.py                      # safety layer on (default)
    ros2 launch bringup system_launch.py safety_layer:=false  # pure pursuit only

Arguments: safety_layer.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

BRINGUP_LAUNCH_DIR = os.path.join(get_package_share_directory('bringup'), 'launch')
AG_VLA_LAUNCH_DIR = os.path.join(get_package_share_directory('ag_vla'), 'launch')


def generate_launch_description():
    safety_layer = LaunchConfiguration('safety_layer')

    safety_layer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(BRINGUP_LAUNCH_DIR, 'safety_layer_launch.py')),
        condition=IfCondition(safety_layer),
    )

    pure_pursuit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(BRINGUP_LAUNCH_DIR, 'pure_pursuit_launch.py')),
        condition=UnlessCondition(safety_layer),
    )

    ag_vla_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(AG_VLA_LAUNCH_DIR, 'ag_vla.launch.py')),
    )

    return LaunchDescription([
        DeclareLaunchArgument('safety_layer', default_value='true',
                              description='true: run the CBF safety layer; false: pure pursuit only'),
        safety_layer_launch,
        pure_pursuit_launch,
        ag_vla_launch,
    ])
