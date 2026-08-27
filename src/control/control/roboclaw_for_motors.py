#!/usr/bin/env python3
"""Roboclaw motor driver node for the Aion R6.

This is the only node that touches the motor controller. Upstream publishes
normalized duty-cycle messages; downstream consumes raw encoder-count deltas.

Must have basicmicro installed via pip-install basicmicro
"""

import rclpy
from aion_msgs.msg import LeftRightFloat32, LeftRightInt32
from rclpy.node import Node

PERCENT_TO_ROBOCLAW = 327.67  # = 32767 / 100; int16 duty range per percent.
ADDRESS = 128     # Roboclaw serial address.
BAUDRATE = 38400  # Roboclaw serial baud rate.
USB_PORT = "/dev/serial/by-id/usb-Basicmicro_Inc._USB_Roboclaw_2x45A-if00"  # Stable by-id path, confirmed on robot.

LEFT_MOTOR_MULTIPLIER = 1.0   # Sign/scale for left duty command + encoder.
RIGHT_MOTOR_MULTIPLIER = 1.0  # Sign/scale for right duty command + encoder.

# Wheel-to-Roboclaw-channel wiring: the physical left wheel is on M2 and the
# physical right wheel is on M1. All *M1M2 calls take (M1, M2) argument order,
# so logical right maps to the first slot and logical left to the second.

ENCODER_PERIOD_SEC = 1.0 / 30.0  # Encoder read rate.
MAX_DUTY_CYCLE = 100.0  # Max commanded duty cycle percentage.
MAX_VELOCITY_QPPS = 3000.0

DRY_RUN = False  # True = skip serial connection and motor commands.


class RoboclawForMotorsNode(Node):
    def __init__(self):
        super().__init__("roboclaw_for_motors")

        self.address = ADDRESS
        self.max_duty_cycle = MAX_DUTY_CYCLE
        self.max_velocity_qpps = MAX_VELOCITY_QPPS
        self.left_multiplier = LEFT_MOTOR_MULTIPLIER
        self.right_multiplier = RIGHT_MOTOR_MULTIPLIER
        self.prev_left_encoder = None
        self.prev_right_encoder = None
        self.seq_num = 1
        self.encoder_seq_num = 1
        self.roboclaw = None
        self.connected = False

        if not DRY_RUN:
            self.connect_roboclaw()
        else:
            self.get_logger().warn("Roboclaw node running in dry_run mode")

        self.create_subscription(
            LeftRightFloat32,
            "set_motor_duty_cycle",
            self.drive_motors_callback,
            1,
        )
        self.current_duty_publisher = self.create_publisher(
            LeftRightFloat32,
            "current_motor_duty_cycle",
            10,
        )
        self.encoder_publisher = self.create_publisher(
            LeftRightInt32,
            "encoder_counts",
            10,
        )
        self.create_subscription(
            LeftRightFloat32,
            "set_motor_velocity",
            self.drive_motors_velocity_callback,
            1,
        )
        self.current_velocity_publisher = self.create_publisher(
            LeftRightFloat32,
            "current_motor_velocity",
            10,
        )
        self.encoder_timer = self.create_timer(ENCODER_PERIOD_SEC, self.publish_encoder_delta)

    def connect_roboclaw(self):
        """Open the Basicmicro/Roboclaw serial connection."""
        try:
            from basicmicro import Basicmicro
        except ImportError as exc:
            raise RuntimeError("Install basicmicro on the robot to use Roboclaw hardware") from exc

        self.roboclaw = Basicmicro(USB_PORT, BAUDRATE)
        self.connected = bool(self.roboclaw.Open())
        if not self.connected:
            self.get_logger().warn(
                f"Failed to open Roboclaw on {USB_PORT} at {BAUDRATE}; motor commands will be ignored"
            )
            return
        self.get_logger().info(f"Connected to Roboclaw on {USB_PORT}")

    @staticmethod
    def clamp(value, low, high):
        """Clamp motor duty cycle to the configured safety limit."""
        return max(low, min(high, value))

    def drive_motors_callback(self, msg):
        """Apply sign multipliers, command Roboclaw, and report actual duty sent."""
        duty_left = self.clamp(float(msg.left) * self.left_multiplier, -self.max_duty_cycle, self.max_duty_cycle)
        duty_right = self.clamp(float(msg.right) * self.right_multiplier, -self.max_duty_cycle, self.max_duty_cycle)

        if self.connected:
            rc_left = int(duty_left * PERCENT_TO_ROBOCLAW)
            rc_right = int(duty_right * PERCENT_TO_ROBOCLAW)
            # M1 = right wheel, M2 = left wheel.
            if not self.roboclaw.DutyM1M2(self.address, rc_right, rc_left):
                self.get_logger().warn("Roboclaw did not acknowledge DutyM1M2 command")

        out = LeftRightFloat32()
        out.left = float(duty_left)
        out.right = float(duty_right)
        out.seq_num = self.seq_num
        self.current_duty_publisher.publish(out)
        self.seq_num += 1

    def drive_motors_velocity_callback(self, msg):
        """Apply sign multipliers, command Roboclaw velocity (QPPS via onboard PID), and report actual velocity sent."""
        velocity_left = self.clamp(
            float(msg.left) * self.left_multiplier, -self.max_velocity_qpps, self.max_velocity_qpps
        )
        velocity_right = self.clamp(
            float(msg.right) * self.right_multiplier, -self.max_velocity_qpps, self.max_velocity_qpps
        )

        if self.connected:
            # M1 = right wheel, M2 = left wheel.
            if not self.roboclaw.SpeedM1M2(self.address, int(velocity_right), int(velocity_left)):
                self.get_logger().warn("Roboclaw did not acknowledge SpeedM1M2 command")

        out = LeftRightFloat32()
        out.left = float(velocity_left)
        out.right = float(velocity_right)
        out.seq_num = self.seq_num
        self.current_velocity_publisher.publish(out)
        self.seq_num += 1

    def read_encoder_pair(self):
        """Read signed absolute encoder counts from Roboclaw M1 and M2.

        Uses GetEncoders so both channels are read from the same Roboclaw
        transaction, and converts unsigned 32-bit values into signed
        integers before computing deltas.
        """
        try:
            result = self.roboclaw.GetEncoders(self.address)
        except Exception as exc:
            self.get_logger().warn(f"Roboclaw encoder read failed: {exc}")
            return None

        if not result[0]:
            return None

        # result[1] = M1 = right wheel, result[2] = M2 = left wheel.
        right = result[1] if result[1] < 0x80000000 else result[1] - 0x100000000
        left = result[2] if result[2] < 0x80000000 else result[2] - 0x100000000
        return int(left), int(right)

    def publish_encoder_delta(self):
        """Publish encoder count deltas for downstream velocity/odometry consumers."""
        if not self.connected:
            return

        encoders = self.read_encoder_pair()
        if encoders is None:
            self.get_logger().warn("Failed to read Roboclaw encoders")
            return

        left, right = encoders
        if self.prev_left_encoder is None:
            # First read establishes a baseline; subsequent reads publish deltas.
            self.prev_left_encoder = left
            self.prev_right_encoder = right
            return

        msg = LeftRightInt32()
        msg.left = int((left - self.prev_left_encoder) * self.left_multiplier)
        msg.right = int((right - self.prev_right_encoder) * self.right_multiplier)
        msg.seq_num = self.encoder_seq_num
        self.encoder_publisher.publish(msg)
        self.encoder_seq_num += 1

        self.prev_left_encoder = left
        self.prev_right_encoder = right

    def destroy_node(self):
        """Stop both motors before shutting down the node."""
        if self.connected:
            try:
                self.roboclaw.DutyM1M2(self.address, 0, 0)
                self.roboclaw.SpeedM1M2(self.address, 0, 0)
                if hasattr(self.roboclaw, "close"):
                    self.roboclaw.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RoboclawForMotorsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
