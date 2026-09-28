"""Bring up all host-side odometry: sensors, VO relay, local EKF and global EKF.

VSLAM itself runs in the Isaac ROS docker container (aion_vslam vslam.launch.py)
and is launched separately.

TF tree:
    map -[global EKF]-> odom -[local EKF]-> base_link -[static]-> camera_link -> ...
    map -[VSLAM]-> vo_odom

Topics:
    /odometry/wheel     encoder_localisation (wheel odometry)
    /odometry/vo_pose_odom  vo_pose_relay (VSLAM VO pose re-based into odom)
    /odometry/filtered  local EKF: wheel + IMU + VO (differential), odom frame
    /odometry/global    global EKF: local EKF velocities + VSLAM SLAM pose, map frame
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

LOCAL_EKF_CONFIG_FILE = os.path.join(
        get_package_share_directory('localisation'),
        'config',
        'local_ekf_wheel_imu_vodom.yaml',
    )

GLOBAL_EKF_CONFIG_FILE = os.path.join(
        get_package_share_directory('localisation'),
        'config',
        'global_ekf_local_vslam.yaml',
    )


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    launch_camera = LaunchConfiguration('launch_camera')

    # --- Camera (Gemini 336) ---
    # IR pair + synced accel/gyro for VSLAM (in the docker container), depth +
    # color for nvblox. Laser interleaved: laser-on frames go to depth, laser-off
    # frames to the IR topics VSLAM tracks on.
    base_to_camera_tf = Node(
        condition=IfCondition(launch_camera),
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_camera_link',
        arguments=[
            '--x', '0.2', '--y', '0', '--z', '0.235',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_link',
        ],
    )

    gemini_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('orbbec_camera'),
                'launch',
                'gemini_330_series.launch.py',
            ])
        ),
        condition=IfCondition(launch_camera),
        launch_arguments={
            'camera_name': 'camera',
            'enable_left_ir': 'true',
            'enable_right_ir': 'true',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_frame_sync': 'true',
            'publish_tf': 'true',
            'enable_accel': 'true',
            'enable_gyro': 'true',
            'enable_sync_output_accel_gyro': 'true',
            'accel_rate': '200hz',
            'gyro_rate': '200hz',

            # laser
            'interleave_ae_mode': 'laser',
            'interleave_frame_enable': 'true',
            'interleave_skip_enable': 'true',
            'interleave_skip_index': '0',        # drop laser-on frames from IR topics
            'laser_index0_laser_control': '1',   # slot 0: laser on  -> depth
            'laser_index1_laser_control': '0',   # slot 1: laser off -> VSLAM
            'laser_index1_ir_brightness': '100',       # default 60 -- brighten passive IR
            'laser_index1_ir_ae_max_exposure': '16000',  # default 17000 -- kept under the 60fps frame period (~16.7ms); higher gets clamped anyway
            'laser_index1_depth_gain': '48',           # default 16 -- gain isn't limited by frame period like exposure is

            'left_ir_fps': '60',
            'right_ir_fps': '60',
            'depth_fps': '60',
        }.items(),
    )

    # --- Wheel odometry + IMU ---
    roboclaw_node = Node(
        package='control',
        executable='roboclaw_for_motors',
        name='roboclaw_for_motors',
        output='screen',
    )

    mavros_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('bringup'),
                'launch',
                'mavros-test.py',
            )
        )
    )

    encoder_node = Node(
        package='localisation',
        executable='encoder_localisation',
        name='encoder_localisation',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
    )

    # --- Visual odometry ---
    vo_pose_relay_node = Node(
        package='localisation',
        executable='vo_pose_relay',
        name='vo_pose_relay',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'vo_topic': '/visual_slam/tracking/vo_pose_covariance',
            'reference_topic': '/odometry/wheel',
            'output_topic': '/odometry/vo_pose_odom',
            'output_frame': 'odom',
        }],
    )

    # --- EKFs ---
    # Local EKF publishes odom -> base_link and /odometry/filtered (no remap),
    # which the safety layer consumes.
    local_ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_local',
        output='screen',
        parameters=[
            LOCAL_EKF_CONFIG_FILE,
            {
                'use_sim_time': use_sim_time,
            },
        ],
    )

    # Global EKF publishes map -> odom and /odometry/global.
    # Remap order matters: ROS applies the first matching rule only, so
    #   1. its input (odom0: /odometry/local in the config) -> the local EKF's /odometry/filtered
    #   2. its own output (odometry/filtered)               -> /odometry/global
    # Rule 2 alone would also catch the input, since both resolve to /odometry/filtered.
    global_ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_global',
        output='screen',
        parameters=[
            GLOBAL_EKF_CONFIG_FILE,
            {
                'use_sim_time': use_sim_time,
            },
        ],
        remappings=[
            ('/odometry/local', '/odometry/filtered'),
            ('odometry/filtered', '/odometry/global'),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock',
        ),
        DeclareLaunchArgument(
            'launch_camera',
            default_value='true',
            description='Launch the Gemini 336 driver and base_link -> camera_link TF',
        ),
        base_to_camera_tf,
        gemini_launch,
        roboclaw_node,
        mavros_launch,
        encoder_node,
        vo_pose_relay_node,
        local_ekf_node,
        global_ekf_node,
    ])
