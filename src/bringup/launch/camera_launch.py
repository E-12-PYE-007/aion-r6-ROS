"""RGB-only Gemini 330-series camera bringup for teleop data collection.

episode_data_collector only needs the color stream (+ odometry), so depth,
point cloud, and the VSLAM-oriented TF/IMU/laser-interleave setup used
elsewhere for isaac_ros_visual_slam are all left off here.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
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
            'enable_depth': 'false',
            'enable_point_cloud': 'false',
            'publish_tf': 'false',
            'enable_laser': 'false',
        }.items(),
    )

    return LaunchDescription([gemini_launch])
