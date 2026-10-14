"""Unified launch harness for the Aion R6: native rover stack + AgVLA + logging, in one shot.

Composes the individual launches/nodes directly (it does not go through
safety_layer_launch / pure_pursuit_launch, which bundle their own localisation):

  - localisation      local  -> local_localisation_launch.py        (wheel+IMU EKF)
                      vslam  -> global_vslam_localisation_launch.py (adds a VSLAM global EKF)
  - motor control     motor_control_launch.py (localisation starts the roboclaw driver)
  - controller        safety_layer:=true  -> cbf_safety_layer       (pure pursuit + CBF-QP filter)
                      safety_layer:=false -> pure_pursuit_controller (no filter)
  - AgVLA edge stack  ag_vla.launch.py (camera, cam_seq_bridge, sys1; owns the one camera)
  - diagnostics       system_monitor -> /system_diagnostics (CPU/GPU/memory/process)
  - recording         ros2 bag of the whole run

The controller drives on the local wheel+IMU EKF: /odometry/filtered in local mode,
/odometry/local in vslam mode (remapped here). Under vslam the VSLAM-fused /odometry/global
is published for logging but is not in the control loop.

The bag records every topic (-a) except the motor-driver topics, so it also captures the
camera, VSLAM and nvblox topics from the separately-run containers. Written to
~/testing_rosbags/<bag_name>, one subdir per run.

Camera images are bagged compressed only: colour as JPEG (.../compressed), depth as lossless
PNG (.../compressedDepth) and IR as lossless PNG (.../compressed, set in ag_vla camera.launch.py).
The depth point cloud (/camera/depth/points, ~5.5 MB/msg) is not bagged; rebuild it from depth +
camera_info. Measured ~10 MB/s total; raw_ir:=true records the IR pair as raw image_raw instead
(~20 MB/s total).

sys2 (the VLA backend), VSLAM and nvblox run separately.

    ros2 launch bringup system_launch.py bag_name:=demo_1
    ros2 launch bringup system_launch.py bag_name:=local_1 localisation:=local
    ros2 launch bringup system_launch.py bag_name:=pp_1 safety_layer:=false
    ros2 launch bringup system_launch.py bag:=false
    ros2 launch bringup system_launch.py bag_name:=vslam_1 raw_ir:=true

Arguments: safety_layer, localisation, bag, bag_name, raw_ir, camera_fps.
"""
import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition, LaunchConfigurationEquals, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

BRINGUP_LAUNCH_DIR = os.path.join(get_package_share_directory('bringup'), 'launch')
AG_VLA_LAUNCH_DIR = os.path.join(get_package_share_directory('ag_vla'), 'launch')

# One subdir per run under here; created now so `ros2 bag record -o` only makes the leaf.
BAG_ROOT = os.path.expanduser('~/testing_rosbags')

# Excluded from the otherwise-everything (-a) recording: motor-driver topics, raw images
# (bagged via their compressed transports instead), the unused theora transport and the depth
# point cloud (~90% of the bag otherwise).
BAG_EXCLUDE_REGEX = '(set_motor_|current_motor_|encoder_counts|/image_raw$|/theora$|/depth/points$)'
# raw_ir:=true -- keep raw IR, drop its now-redundant PNG copy (saves the encode CPU too).
BAG_EXCLUDE_REGEX_RAW_IR = (
    '(set_motor_|current_motor_|encoder_counts|/(color|depth)/image_raw$'
    '|_ir/image_raw/compressed$|/theora$|/depth/points$)')


def generate_launch_description():
    os.makedirs(BAG_ROOT, exist_ok=True)

    safety_layer = LaunchConfiguration('safety_layer')
    localisation = LaunchConfiguration('localisation')
    # Controller drives on the local EKF: /odometry/local in vslam mode, /odometry/filtered otherwise.
    odom_topic = PythonExpression(
        ["'/odometry/local' if '", localisation, "' == 'vslam' else '/odometry/filtered'"])

    # --- localisation (exactly one) ---
    local_localisation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(BRINGUP_LAUNCH_DIR, 'local_localisation_launch.py')),
        condition=LaunchConfigurationEquals('localisation', 'local'),
    )
    vslam_localisation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(BRINGUP_LAUNCH_DIR, 'global_vslam_localisation_launch.py')),
        condition=LaunchConfigurationEquals('localisation', 'vslam'),
    )

    # --- motor control (localisation starts the roboclaw driver) ---
    motor_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(BRINGUP_LAUNCH_DIR, 'motor_control_launch.py')),
        launch_arguments={'start_driver': 'false'}.items(),
    )

    # --- controller (exactly one), driving on whichever local EKF topic is live ---
    cbf_safety_layer = Node(
        package='control', executable='cbf_safety_layer', name='cbf_safety_layer',
        output='screen',
        remappings=[('/odometry/filtered', odom_topic)],
        condition=IfCondition(safety_layer),
    )
    pure_pursuit = Node(
        package='control', executable='pure_pursuit_controller', name='pure_pursuit_controller',
        output='screen',
        remappings=[('/odometry/filtered', odom_topic)],
        condition=UnlessCondition(safety_layer),
    )

    # --- AgVLA edge stack (owns the single Orbbec camera) ---
    ag_vla = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(AG_VLA_LAUNCH_DIR, 'ag_vla.launch.py')),
        launch_arguments={'camera_fps': LaunchConfiguration('camera_fps')}.items(),
    )

    # --- diagnostics ---
    system_monitor = Node(
        package='data_collection', executable='system_monitor', name='system_monitor',
        output='screen',
    )

    # --- recording ---
    bag_exclude = PythonExpression([
        "'", BAG_EXCLUDE_REGEX_RAW_IR, "' if '", LaunchConfiguration('raw_ir'),
        "'.lower() == 'true' else '", BAG_EXCLUDE_REGEX, "'"])
    record_bag = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('bag')),
        cmd=[
            'ros2', 'bag', 'record', '-a',
            '-x', bag_exclude,
            '-o', [BAG_ROOT + os.sep, LaunchConfiguration('bag_name')],
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('safety_layer', default_value='true',
                              description='true: run the CBF safety layer; false: pure pursuit only'),
        DeclareLaunchArgument('localisation', default_value='vslam',
                              description="Localisation stack: 'local' (wheel+IMU EKF) or 'vslam' (adds VSLAM global EKF)"),
        DeclareLaunchArgument('bag', default_value='true',
                              description='Record a rosbag of the run'),
        DeclareLaunchArgument('bag_name', default_value=datetime.now().strftime('run_%Y%m%d_%H%M%S'),
                              description='Subdir name under ~/testing_rosbags for this run\'s bag'),
        DeclareLaunchArgument('raw_ir', default_value='false',
                              description='Bag the left/right IR as raw image_raw instead of PNG-compressed'),
        DeclareLaunchArgument('camera_fps', default_value='30',
                              description='Camera frame rate [Hz] for IR, colour and depth (e.g. 30 or 60)'),
        local_localisation,
        vslam_localisation,
        motor_control,
        cbf_safety_layer,
        pure_pursuit,
        ag_vla,
        system_monitor,
        record_bag,
    ])
