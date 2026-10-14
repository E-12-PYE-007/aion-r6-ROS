import os
import importlib.util

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch import LaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

# Lossless PNG (not the default JPEG) for the IR .../image_raw/compressed topics, so bagged IR
# is still usable for VSLAM replay. The leading '.' is how Humble's image_transport names these
# (topic minus the node namespace); they are read once when the camera node starts.
IR_PNG_PARAMS = {
    '.left_ir.image_raw.format': 'png',
    '.right_ir.image_raw.format': 'png',
}


def load_gemini_launch():
    """The orbbec gemini_330_series launch, with IR_PNG_PARAMS added to the camera node.

    That launch only forwards params it declares as arguments, and these names can't go through
    SetParameter (it also hands them to the container as `-p`, which rcl fails to parse), so wrap
    its load_parameters rather than patching the OrbbecSDK_ROS2 submodule.
    """
    path = os.path.join(get_package_share_directory('orbbec_camera'),
                        'launch', 'gemini_330_series.launch.py')
    spec = importlib.util.spec_from_file_location('gemini_330_series_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    load_parameters = module.load_parameters
    module.load_parameters = lambda context, args: {**load_parameters(context, args),
                                                    **IR_PNG_PARAMS}
    return module.generate_launch_description()


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
    camera_fps = LaunchConfiguration('camera_fps')

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
        LaunchDescriptionSource(load_gemini_launch()),
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

            'left_ir_fps': camera_fps,
            'right_ir_fps': camera_fps,
            'color_fps': camera_fps,
            'depth_fps': camera_fps,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('camera_fps', default_value='30',
                              description='Camera frame rate [Hz] for IR, colour and depth (e.g. 30 or 60)'),
        base_to_camera_tf,
        gemini_launch,
    ])
