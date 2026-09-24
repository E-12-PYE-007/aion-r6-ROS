"""Local image decoder for viewing the robot camera on the workstation.

The Gemini camera publishes JPEG-compressed color frames on
/camera/color/image_raw/compressed. Streaming raw frames over the network is
expensive, and image_transport's compressed subscription is fiddly in rqt/RViz
(topic-switching crashes, ghost endpoints, the base-topic-vs-/compressed
confusion). This launch runs the data_collection image_decoder node *locally on
the workstation*: it pulls the compressed stream over the network with
BEST_EFFORT QoS (so the Jetson never buffers/retransmits for the viewer and its
frame rate isn't affected), decodes it, and re-publishes a plain
sensor_msgs/Image on /workstation/image_raw, so RViz and rqt just view an
ordinary raw topic with no transport plugins involved.

Usage:
    ros2 launch bringup image_decode_launch.py
    # then point RViz Image display / rqt_image_view at /workstation/image_raw

Override topics if needed:
    ros2 launch bringup image_decode_launch.py \
        compressed_topic:=/camera/color/image_raw/compressed \
        output_topic:=/workstation/image_raw
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    compressed_topic = LaunchConfiguration('compressed_topic')
    output_topic = LaunchConfiguration('output_topic')

    return LaunchDescription([
        DeclareLaunchArgument(
            'compressed_topic',
            default_value='/camera/color/image_raw/compressed',
            description='Compressed (JPEG) image topic to pull over the network.',
        ),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/workstation/image_raw',
            description='Local raw sensor_msgs/Image topic to publish for RViz/rqt.',
        ),
        # Best-effort decoder: subscribes CompressedImage best-effort (no Jetson
        # retransmit/backlog), decodes, and publishes a plain Image best-effort.
        Node(
            package='data_collection',
            executable='image_decoder',
            name='image_decoder',
            parameters=[{
                'compressed_topic': compressed_topic,
                'output_topic': output_topic,
            }],
            output='screen',
        ),
    ])
