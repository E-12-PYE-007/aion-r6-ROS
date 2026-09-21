import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def quaternion_from_yaw(yaw):
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_xy_yaw(msg):
    p = msg.pose.pose if isinstance(msg, Odometry) else msg.pose.pose
    return p.position.x, p.position.y, yaw_from_quaternion(p.orientation)


def rotate_covariance(covariance, angle):
    """Rotate a planar 6x6 pose covariance into the output frame."""
    c = math.cos(angle)
    s = math.sin(angle)
    jacobian = [[0.0] * 6 for _ in range(6)]
    jacobian[0][0], jacobian[0][1] = c, -s
    jacobian[1][0], jacobian[1][1] = s, c
    jacobian[2][2] = 1.0
    jacobian[3][3] = 1.0
    jacobian[4][4] = 1.0
    jacobian[5][5] = 1.0

    result = [[0.0] * 6 for _ in range(6)]
    for i in range(6):
        for j in range(6):
            result[i][j] = sum(
                jacobian[i][a] * covariance[6 * a + b] * jacobian[j][b]
                for a in range(6) for b in range(6)
            )
    return [result[i][j] for i in range(6) for j in range(6)]


class VoPoseRelay(Node):
    def __init__(self):
        super().__init__('vo_pose_relay')
        self.declare_parameter('vo_topic', '/visual_slam/tracking/vo_pose_covariance')
        self.declare_parameter('reference_topic', '/odometry/wheel')
        self.declare_parameter('output_topic', '/odometry/vo_pose_odom')
        self.declare_parameter('output_frame', 'odom')
        self.declare_parameter('max_initial_time_gap', 0.25)

        vo_topic = self.get_parameter('vo_topic').value
        reference_topic = self.get_parameter('reference_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.output_frame = self.get_parameter('output_frame').value
        self.max_initial_time_gap = float(
            self.get_parameter('max_initial_time_gap').value
        )

        self.reference_msg = None
        self.transform = None
        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped, output_topic, 10
        )
        self.create_subscription(
            Odometry, reference_topic, self.reference_callback, 20
        )
        self.create_subscription(
            PoseWithCovarianceStamped, vo_topic, self.vo_callback, 20
        )
        self.get_logger().info(
            f'Relaying {vo_topic} into {self.output_frame} using {reference_topic}'
        )

    @staticmethod
    def stamp_seconds(stamp):
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def reference_callback(self, msg):
        self.reference_msg = msg

    def vo_callback(self, msg):
        if self.reference_msg is None:
            return

        if self.transform is None:
            reference_time = self.stamp_seconds(self.reference_msg.header.stamp)
            vo_time = self.stamp_seconds(msg.header.stamp)
            if abs(reference_time - vo_time) > self.max_initial_time_gap:
                return

            ref_x, ref_y, ref_yaw = pose_xy_yaw(self.reference_msg)
            vo_x, vo_y, vo_yaw = pose_xy_yaw(msg)

            # T_odom_vo maps a point in vo_odom into odom. At initialization,
            # the VSLAM base pose and reference base pose are the same physical
            # point, so this fixed SE(2) transform is sufficient for planar data.
            angle = ref_yaw - vo_yaw
            c = math.cos(angle)
            s = math.sin(angle)
            trans_x = ref_x - (c * vo_x - s * vo_y)
            trans_y = ref_y - (s * vo_x + c * vo_y)
            self.transform = (trans_x, trans_y, angle)
            self.get_logger().info(
                'Initialized vo_odom -> odom transform: '
                f'x={trans_x:.3f}, y={trans_y:.3f}, '
                f'yaw={math.degrees(angle):.2f} deg'
            )

        trans_x, trans_y, angle = self.transform
        vo_x, vo_y, vo_yaw = pose_xy_yaw(msg)
        c = math.cos(angle)
        s = math.sin(angle)
        out_x = trans_x + c * vo_x - s * vo_y
        out_y = trans_y + s * vo_x + c * vo_y
        out_yaw = wrap_angle(angle + vo_yaw)

        output = PoseWithCovarianceStamped()
        output.header = msg.header
        output.header.frame_id = self.output_frame
        output.pose.pose.position.x = out_x
        output.pose.pose.position.y = out_y
        output.pose.pose.position.z = msg.pose.pose.position.z
        qx, qy, qz, qw = quaternion_from_yaw(out_yaw)
        output.pose.pose.orientation.x = qx
        output.pose.pose.orientation.y = qy
        output.pose.pose.orientation.z = qz
        output.pose.pose.orientation.w = qw
        output.pose.covariance = rotate_covariance(msg.pose.covariance, angle)
        self.publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = VoPoseRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
