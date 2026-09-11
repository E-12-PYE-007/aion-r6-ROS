#!/usr/bin/env python3
"""Mo-Cap data collection node.

Publish tracked NatNet rigid-body poses as ROS2 messages as geometry_msgs/PoseStamped
"""

import math
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

NATNET_PYTHON_PATH = Path(
    "/home/vla-cap/vla-cap-ws/aion-r6-vla-master/aion-r6-ROS/"
    "third_party/natnet/samples/PythonClient"
)

if not (NATNET_PYTHON_PATH / "NatNetClient.py").is_file():
    raise ImportError(
        f"NatNetClient.py not found in {NATNET_PYTHON_PATH}"
    )

# if the path exists add it to python's search path
sys.path.insert(0, str(NATNET_PYTHON_PATH))
from NatNetClient import NatNetClient



SERVER_ADDRESS = "10.42.0.82"
CLIENT_ADDRESS = "10.42.0.1"
TRACKED_RIGID_BODY_ID = 1 


MOCAP_TOPIC = 'mocap/rover/pose'
MOCAP_FRAME = "mocap_world"
 

class MoCapDataCollectionNode(Node):
    def __init__(self):
        super().__init__('mocap_data_collector')

        self.pose_publisher = self.create_publisher(PoseStamped, MOCAP_TOPIC, 10)

        self.client = NatNetClient()
        self.client.set_server_address(SERVER_ADDRESS)
        self.client.set_client_address(CLIENT_ADDRESS)
        self.client.set_use_multicast(False)

        # NatNet calls this function once per rigid body per frame.
        self.client.rigid_body_listener = (
            self.receive_mocap_frame
        )

        self.get_logger().info(
            f"Starting NatNet: server={SERVER_ADDRESS}, "
            f"client={CLIENT_ADDRESS}, "
            f"unicast, rigid_body_id={TRACKED_RIGID_BODY_ID}"
        )



    def start(self):
        """Start the SDK's receiving threads."""
        if not self.client.run():
            raise RuntimeError("NatNet receiving threads failed to start")

    def stop(self):
          """Stop the receiver before destroying its ROS publisher."""
          self.client.shutdown()

    def receive_mocap_frame(self, rigid_body_id, position, rotation):
        """
        Called by the NatNet receiver thread.

        Published the timestamped pose
        """

        receipt_stamp = self.get_clock().now()

        if rigid_body_id != TRACKED_RIGID_BODY_ID:
            return


        x, y, z = (float(value) for value in position)
        qx, qy, qz, qw = (float(value) for value in rotation)

        if not all(
            math.isfinite(value)
            for value in (x, y, z, qx, qy, qz, qw)
        ):
            return

        quaternion_norm = math.sqrt(
            qx * qx + qy * qy + qz * qz + qw * qw
        )

        # division by zero guard
        if quaternion_norm < 1e-12:
            return

        
        msg = PoseStamped()
        msg.header.stamp = receipt_stamp.to_msg()
        msg.header.frame_id = MOCAP_FRAME

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z

        msg.pose.orientation.x = qx / quaternion_norm
        msg.pose.orientation.y = qy / quaternion_norm
        msg.pose.orientation.z = qz / quaternion_norm
        msg.pose.orientation.w = qw / quaternion_norm

        self.pose_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MoCapDataCollectionNode()
    try:
        node.start()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()



if __name__ == '__main__':
    main()
