from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    """
    Bring up the front Gemini 336 with colour, depth, and left/right IR enabled.

    /camera/depth/image_raw is the depth image aligned to the colour viewpoint (software
    alignment), in camera_color_optical_frame; the original is on /camera/depth/image_unaligned.

    The laser is interleaved per-frame so IR frames stay laser-off (clean passive stereo for
    VSLAM) while depth frames get the laser (for ranging).
    """
    base_to_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_camera_link',
        arguments=[
            '--x', '0.2', '--y', '0', '--z', '0.225',
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
            'enable_color': 'true',
            'enable_depth': 'true',
            'depth_registration': 'true',
            'align_mode': 'SW',
            'align_target_stream': 'COLOR',
            'enable_left_ir': 'true',
            'enable_right_ir': 'true',
            'enable_frame_sync': 'true',
            'publish_tf': 'true',
            'enable_accel': 'true',
            'enable_gyro': 'true',
            'enable_sync_output_accel_gyro': 'true',
            'accel_rate': '200hz',
            'gyro_rate': '200hz',
            'enable_laser': 'true',

            # laser interleaving
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

    return LaunchDescription([base_to_camera_tf, gemini_launch])
