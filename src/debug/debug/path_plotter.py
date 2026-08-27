#!/usr/bin/env python3
"""Path plotter node for the Aion R6.

Subscribes to the wheel-odometry pose estimate (nav_msgs/Odometry, default
/odometry/wheel) and plots the XY path travelled. Updates a live matplotlib
window when an interactive backend is available, and always writes a PNG of
the final path on shutdown.

Parameters
----------
odom_topic (str, default '/odometry/wheel')
    Odometry topic to subscribe to.
output_path (str, default 'path_plot.png')
    Where the final path image is written on shutdown.
redraw_period_sec (float, default 0.2)
    Live-plot refresh period.
min_point_spacing_m (float, default 0.01)
    Points closer than this to the previous one are dropped, to bound memory
    on long runs. Set to 0.0 to keep every sample.
live (bool, default True)
    Attempt a live plot window. Ignored when the matplotlib backend is
    non-interactive (e.g. headless over SSH), in which case only the
    shutdown PNG is produced.
"""

import math

import matplotlib
import matplotlib.pyplot as plt
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_from_quaternion(z, w):
    """Yaw (rad) from the z/w terms of a planar quaternion."""
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


class PathPlotterNode(Node):
    def __init__(self):
        super().__init__('path_plotter')

        self.odom_topic = self.declare_parameter('odom_topic', '/odometry/wheel').value
        self.output_path = self.declare_parameter('output_path', 'path_plot.png').value
        self.redraw_period_sec = self.declare_parameter('redraw_period_sec', 0.2).value
        self.min_point_spacing_m = self.declare_parameter('min_point_spacing_m', 0.01).value
        want_live = self.declare_parameter('live', True).value

        self.xs = []
        self.ys = []
        self.last_yaw = 0.0

        # Interactive backends contain a canvas manager; Agg and friends don't.
        self.live = bool(want_live) and matplotlib.get_backend().lower() != 'agg'

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_title('Path taken')
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.grid(True, linestyle=':', alpha=0.5)
        (self.path_line,) = self.ax.plot([], [], '-', color='tab:blue', label='path')
        (self.start_marker,) = self.ax.plot([], [], 'go', label='start')
        (self.head_marker,) = self.ax.plot([], [], 'ro', label='current')
        self.ax.legend(loc='best')

        if self.live:
            plt.ion()
            self.fig.show()
        else:
            self.get_logger().info(
                f"Non-interactive backend '{matplotlib.get_backend()}'; "
                f"live plot disabled, will save to {self.output_path} on shutdown"
            )

        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.redraw_timer = self.create_timer(self.redraw_period_sec, self.redraw)

        self.get_logger().info(f"Plotting path from {self.odom_topic}")

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.last_yaw = yaw_from_quaternion(
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        )

        if self.xs:
            if math.hypot(x - self.xs[-1], y - self.ys[-1]) < self.min_point_spacing_m:
                return
        self.xs.append(x)
        self.ys.append(y)

    def redraw(self):
        if not self.xs:
            return

        self.path_line.set_data(self.xs, self.ys)
        self.start_marker.set_data([self.xs[0]], [self.ys[0]])
        self.head_marker.set_data([self.xs[-1]], [self.ys[-1]])
        self.ax.relim()
        self.ax.autoscale_view()

        if self.live:
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except Exception as exc:  # Display went away mid-run; fall back to save-only.
                self.get_logger().warn(f"Live redraw failed ({exc}); disabling live plot")
                self.live = False

    def save(self):
        self.redraw()
        try:
            self.fig.savefig(self.output_path, dpi=150, bbox_inches='tight')
            self.get_logger().info(
                f"Saved path with {len(self.xs)} points to {self.output_path}"
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to save path plot: {exc}")

    def destroy_node(self):
        self.save()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PathPlotterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
