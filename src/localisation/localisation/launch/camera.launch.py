from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    """
    Bring up the front Gemini 336 with left/right IR enabled.

    Publishes camera_left_ir_optical_frame / camera_right_ir_optical_frame
    (+ TF from camera_link down to them) that the isaac_ros_visual_slam
    container -- running separately in the Isaac ROS docker workspace -- subscribes to.
    Color/depth are left off here since VSLAM only needs the IR pair.

    Accel/gyro are enabled and synced into one combined IMU stream (camera/
    gyro_accel/sample, frame camera_accel_gyro_optical_frame) for VSLAM's
    enable_imu_fusion -- see vslam.launch.py.

    Also publishes the base_link -> camera_link mounting transform, measured
    from the FCU (base_link origin) to the camera housing: 0.2m forward, 0m left,
    0.05m up, mounted level and forward-facing (identity rotation). This completes
    the chain base_link -> camera_link -> ... -> camera_*_ir_optical_frame
    that isaac_ros_visual_slam looks up at startup.
    """
    base_to_camera_tf = Node(
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
        }.items(),
    )

    return LaunchDescription([base_to_camera_tf, gemini_launch])
