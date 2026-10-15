"""The display bridge: reads the rover's topics, serves the page, and starts and stops the stack over ssh.

Run it on the workstation, in a shell with the usual ROS_DOMAIN_ID and ROS_DISCOVERY_SERVER, then open
http://localhost:8080 in a browser.
"""
import math
import os
import threading
from pathlib import Path

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from visualization_msgs.msg import MarkerArray

from endeavour_display.commander import Commander
from endeavour_display.jetson import DEFAULT_SCRIPT, JetsonControl
from endeavour_display.markers import parse_weeds
from endeavour_display.plants import load_plants
from endeavour_display.server import CameraBuffer, DisplayServer
from endeavour_display.state import DisplayState

ODOM_TOPIC = '/odometry/local'
MARKERS_TOPIC = '/weed_markers'
IMAGE_TOPIC = '/weed_detections/image/compressed'
FILTERING_TOPIC = '/safety_layer/filtering_active'
REVERSING_TOPIC = '/safety_layer/reversing'
DIAGNOSTICS_TOPIC = '/system_diagnostics'


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def find_web_dir():
    try:
        share = Path(get_package_share_directory('endeavour_display')) / 'web'
        if (share / 'index.html').is_file():
            return share
    except PackageNotFoundError:
        pass
    return Path(__file__).resolve().parents[1] / 'web'


class BridgeNode(Node):
    def __init__(self):
        super().__init__('display_bridge')
        self.declare_parameter('host', '127.0.0.1')
        self.declare_parameter('port', 8080)
        self.declare_parameter('jetson_host', '10.43.254.23')
        self.declare_parameter('jetson_fallback_host', '')
        self.declare_parameter('jetson_user', 'vla-cap')
        self.declare_parameter('script_path', DEFAULT_SCRIPT)
        self.declare_parameter('plants_file', '')
        self.declare_parameter('poll_period_s', 5.0)
        self.declare_parameter('discovery', 'auto')   # server, multicast, or auto: mirror this shell's environment
        self.declare_parameter('local_test', False)   # run the script in a local shell instead of over ssh

    def param(self, name):
        return self.get_parameter(name).value

    def subscribe(self, state, camera):
        sensor = qos_profile_sensor_data
        latest_snapshot = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                     durability=DurabilityPolicy.TRANSIENT_LOCAL)

        def on_odometry(msg):
            p = msg.pose.pose
            state.on_odometry(p.position.x, p.position.y, yaw_from_quaternion(p.orientation))

        def on_image(msg):
            camera.push(msg.data)
            state.on_camera_frame()

        self.create_subscription(Odometry, ODOM_TOPIC, on_odometry, sensor)
        self.create_subscription(MarkerArray, MARKERS_TOPIC, lambda m: state.on_weeds(parse_weeds(m.markers)), latest_snapshot)
        self.create_subscription(CompressedImage, IMAGE_TOPIC, on_image, sensor)
        self.create_subscription(Bool, FILTERING_TOPIC, lambda m: state.on_filtering(m.data), sensor)
        self.create_subscription(Bool, REVERSING_TOPIC, lambda m: state.on_reversing(m.data), sensor)
        self.create_subscription(DiagnosticArray, DIAGNOSTICS_TOPIC, lambda m: state.on_heartbeat(), sensor)


def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    state = DisplayState(load_plants(node.param('plants_file') or None))
    camera = CameraBuffer()
    node.subscribe(state, camera)

    discovery = node.param('discovery')
    multicast = discovery == 'multicast' or (discovery == 'auto' and not os.environ.get('ROS_DISCOVERY_SERVER'))
    control = JetsonControl([node.param('jetson_host'), node.param('jetson_fallback_host')],
                            user=node.param('jetson_user'), script=node.param('script_path'),
                            local=node.param('local_test'), multicast=multicast)
    commander = Commander(control, state, poll_s=node.param('poll_period_s'))
    server = DisplayServer(state, commander, camera, find_web_dir(), host=node.param('host'), port=node.param('port'))
    threading.Thread(target=commander.run_forever, daemon=True).start()
    server.start()
    node.get_logger().info(f'Display on http://{node.param("host")}:{server.port}, Jetson {node.param("jetson_host")}, '
                           f'discovery {"multicast" if multicast else "server " + os.environ.get("ROS_DISCOVERY_SERVER", "")}')
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        commander.close()
        server.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
