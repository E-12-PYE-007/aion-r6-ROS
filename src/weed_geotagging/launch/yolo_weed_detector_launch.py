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
        DeclareLaunchArgument('confidence', default_value='0.25'),
        DeclareLaunchArgument('iou', default_value='0.45'),
        DeclareLaunchArgument('device', default_value=''),
        DeclareLaunchArgument('target_classes', default_value=''),
        DeclareLaunchArgument('publish_annotated', default_value='true'),
        DeclareLaunchArgument('annotated_topic', default_value='/weed_detections/image'),
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
    ])
