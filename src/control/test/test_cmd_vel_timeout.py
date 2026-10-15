import time

import rclpy
from custom_msgs.msg import LeftRightFloat32
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor

from control.cmd_vel_to_roboclaw import CMD_VEL_TIMEOUT_S, CmdVelToRoboclawNode


def spin_for(executor, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        executor.spin_once(timeout_sec=0.01)


def test_wheels_are_zeroed_when_cmd_vel_stops_and_resume_with_it():
    rclpy.init()
    try:
        converter = CmdVelToRoboclawNode()
        helper = rclpy.create_node('cmd_vel_timeout_test')
        received = []
        helper.create_subscription(LeftRightFloat32, 'set_motor_velocity',
                                   lambda m: received.append((time.monotonic(), m.left, m.right)), 10)
        publisher = helper.create_publisher(Twist, 'cmd_vel', 10)
        executor = SingleThreadedExecutor()
        executor.add_node(converter)
        executor.add_node(helper)
        spin_for(executor, 0.5)

        cmd = Twist()
        cmd.linear.x = 0.2
        sent_until = time.monotonic() + 0.4
        while time.monotonic() < sent_until:
            publisher.publish(cmd)
            spin_for(executor, 0.05)
        moving = [r for r in received if r[1] != 0.0 or r[2] != 0.0]
        assert moving, 'a cmd_vel should produce non-zero wheel velocities'

        stopped_at = time.monotonic()
        spin_for(executor, CMD_VEL_TIMEOUT_S + 0.6)
        assert received[-1][1] == 0.0 and received[-1][2] == 0.0
        first_zero = next(t for t, left, right in received if t > stopped_at and left == 0.0 and right == 0.0)
        assert first_zero - stopped_at >= CMD_VEL_TIMEOUT_S - 0.1, 'the wheels were zeroed before the timeout'

        publisher.publish(cmd)
        spin_for(executor, 0.3)
        assert received[-1][1] != 0.0, 'wheels should move again when cmd_vel resumes'
    finally:
        rclpy.shutdown()
