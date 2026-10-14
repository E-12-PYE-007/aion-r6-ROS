from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('detections_topic', default_value='/weed_detections_json'),
        DeclareLaunchArgument('depth_topic', default_value='/camera/depth/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/color/camera_info'),
        DeclareLaunchArgument('odom_topic', default_value='/odometry/filtered'),
        DeclareLaunchArgument('output_csv', default_value=''),
        DeclareLaunchArgument('target_label', default_value=''),
        DeclareLaunchArgument('min_score', default_value='0.0'),
        DeclareLaunchArgument('max_depth_m', default_value='8.0'),
        DeclareLaunchArgument('bbox_shrink_ratio', default_value='0.2'),
        DeclareLaunchArgument('max_depth_samples', default_value='800'),
        DeclareLaunchArgument('min_depth_samples', default_value='20'),
        DeclareLaunchArgument('depth_mad_scale', default_value='2.5'),
        DeclareLaunchArgument('foreground_depth_percentile', default_value='20.0'),
        DeclareLaunchArgument('foreground_depth_tolerance_m', default_value='0.6'),
        DeclareLaunchArgument('merge_radius_m', default_value='0.35'),
        DeclareLaunchArgument('turn_tracking_angular_z', default_value='0.15'),
        DeclareLaunchArgument('turning_merge_radius_m', default_value='0.75'),
        DeclareLaunchArgument('new_track_max_angular_z', default_value='999.0'),
        DeclareLaunchArgument('new_track_turn_cooldown_sec', default_value='0.0'),
        DeclareLaunchArgument('min_confirmed_observations', default_value='3'),
        DeclareLaunchArgument('turning_min_confirmed_observations', default_value='6'),
        DeclareLaunchArgument('track_merge_radius_m', default_value='0.25'),
        DeclareLaunchArgument('odom_buffer_sec', default_value='10.0'),
        DeclareLaunchArgument('depth_buffer_sec', default_value='10.0'),
        DeclareLaunchArgument('max_sync_age_sec', default_value='0.1'),
        DeclareLaunchArgument('track_min_update_weight', default_value='0.10'),
        DeclareLaunchArgument('track_max_update_weight', default_value='0.45'),
        DeclareLaunchArgument('angular_slowdown_start', default_value='0.2'),
        DeclareLaunchArgument('angular_slowdown_stop', default_value='0.8'),
        DeclareLaunchArgument('camera_x', default_value='0.2'),
        DeclareLaunchArgument('camera_y', default_value='0.0'),
        DeclareLaunchArgument('camera_z', default_value='0.235'),
        DeclareLaunchArgument('camera_yaw', default_value='0.0'),
        Node(
            package='weed_geotagging',
            executable='weed_geotagger',
            name='weed_geotagger',
            output='screen',
            parameters=[{
                'detections_topic': LaunchConfiguration('detections_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'odom_topic': LaunchConfiguration('odom_topic'),
                'output_csv': LaunchConfiguration('output_csv'),
                'target_label': LaunchConfiguration('target_label'),
                'min_score': ParameterValue(LaunchConfiguration('min_score'), value_type=float),
                'max_depth_m': ParameterValue(LaunchConfiguration('max_depth_m'), value_type=float),
                'bbox_shrink_ratio': ParameterValue(LaunchConfiguration('bbox_shrink_ratio'), value_type=float),
                'max_depth_samples': ParameterValue(LaunchConfiguration('max_depth_samples'), value_type=int),
                'min_depth_samples': ParameterValue(LaunchConfiguration('min_depth_samples'), value_type=int),
                'depth_mad_scale': ParameterValue(LaunchConfiguration('depth_mad_scale'), value_type=float),
                'foreground_depth_percentile': ParameterValue(
                    LaunchConfiguration('foreground_depth_percentile'),
                    value_type=float,
                ),
                'foreground_depth_tolerance_m': ParameterValue(
                    LaunchConfiguration('foreground_depth_tolerance_m'),
                    value_type=float,
                ),
                'merge_radius_m': ParameterValue(LaunchConfiguration('merge_radius_m'), value_type=float),
                'turn_tracking_angular_z': ParameterValue(
                    LaunchConfiguration('turn_tracking_angular_z'),
                    value_type=float,
                ),
                'turning_merge_radius_m': ParameterValue(
                    LaunchConfiguration('turning_merge_radius_m'),
                    value_type=float,
                ),
                'new_track_max_angular_z': ParameterValue(
                    LaunchConfiguration('new_track_max_angular_z'),
                    value_type=float,
                ),
                'new_track_turn_cooldown_sec': ParameterValue(
                    LaunchConfiguration('new_track_turn_cooldown_sec'),
                    value_type=float,
                ),
                'min_confirmed_observations': ParameterValue(
                    LaunchConfiguration('min_confirmed_observations'),
                    value_type=int,
                ),
                'turning_min_confirmed_observations': ParameterValue(
                    LaunchConfiguration('turning_min_confirmed_observations'),
                    value_type=int,
                ),
                'track_merge_radius_m': ParameterValue(
                    LaunchConfiguration('track_merge_radius_m'),
                    value_type=float,
                ),
                'odom_buffer_sec': ParameterValue(
                    LaunchConfiguration('odom_buffer_sec'),
                    value_type=float,
                ),
                'depth_buffer_sec': ParameterValue(
                    LaunchConfiguration('depth_buffer_sec'),
                    value_type=float,
                ),
                'max_sync_age_sec': ParameterValue(
                    LaunchConfiguration('max_sync_age_sec'),
                    value_type=float,
                ),
                'track_min_update_weight': ParameterValue(
                    LaunchConfiguration('track_min_update_weight'),
                    value_type=float,
                ),
                'track_max_update_weight': ParameterValue(
                    LaunchConfiguration('track_max_update_weight'),
                    value_type=float,
                ),
                'angular_slowdown_start': ParameterValue(
                    LaunchConfiguration('angular_slowdown_start'),
                    value_type=float,
                ),
                'angular_slowdown_stop': ParameterValue(
                    LaunchConfiguration('angular_slowdown_stop'),
                    value_type=float,
                ),
                'camera_x': ParameterValue(LaunchConfiguration('camera_x'), value_type=float),
                'camera_y': ParameterValue(LaunchConfiguration('camera_y'), value_type=float),
                'camera_z': ParameterValue(LaunchConfiguration('camera_z'), value_type=float),
                'camera_yaw': ParameterValue(LaunchConfiguration('camera_yaw'), value_type=float),
            }],
        ),
    ])
