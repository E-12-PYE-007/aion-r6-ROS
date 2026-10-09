from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('image_topic', default_value='/camera/color/image_raw'),
        DeclareLaunchArgument('detections_topic', default_value='/weed_detections_json'),
        DeclareLaunchArgument('depth_topic', default_value='/camera/depth/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/color/camera_info'),
        DeclareLaunchArgument('odom_topic', default_value='/odometry/filtered'),
        DeclareLaunchArgument('output_csv', default_value=''),
        DeclareLaunchArgument('target_label', default_value='weed'),
        DeclareLaunchArgument('target_classes', default_value=''),
        DeclareLaunchArgument('confidence', default_value='0.25'),
        DeclareLaunchArgument('iou', default_value='0.45'),
        DeclareLaunchArgument('device', default_value=''),
        DeclareLaunchArgument('publish_annotated', default_value='true'),
        DeclareLaunchArgument('annotated_topic', default_value='/weed_detections/image'),
        DeclareLaunchArgument('min_score', default_value='0.0'),
        DeclareLaunchArgument('camera_x', default_value='0.2'),
        DeclareLaunchArgument('camera_y', default_value='0.0'),
        DeclareLaunchArgument('camera_z', default_value='0.235'),
        DeclareLaunchArgument('camera_yaw', default_value='0.0'),
        Node(
            package='weed_geotagging',
            executable='yolo_weed_detector',
            name='yolo_weed_detector',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'image_topic': LaunchConfiguration('image_topic'),
                'detections_topic': LaunchConfiguration('detections_topic'),
                'confidence': ParameterValue(LaunchConfiguration('confidence'), value_type=float),
                'iou': ParameterValue(LaunchConfiguration('iou'), value_type=float),
                'device': LaunchConfiguration('device'),
                'target_classes': LaunchConfiguration('target_classes'),
                'publish_annotated': ParameterValue(
                    LaunchConfiguration('publish_annotated'),
                    value_type=bool,
                ),
                'annotated_topic': LaunchConfiguration('annotated_topic'),
            }],
        ),
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
                'camera_x': ParameterValue(LaunchConfiguration('camera_x'), value_type=float),
                'camera_y': ParameterValue(LaunchConfiguration('camera_y'), value_type=float),
                'camera_z': ParameterValue(LaunchConfiguration('camera_z'), value_type=float),
                'camera_yaw': ParameterValue(LaunchConfiguration('camera_yaw'), value_type=float),
            }],
        ),
    ])
