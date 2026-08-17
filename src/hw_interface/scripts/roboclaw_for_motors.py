#!/usr/bin/env python3
"""Roboclaw motor driver node for the Aion R6.

This node is intentionally the only node that touches the motor controller.
Everything upstream publishes normalized duty-cycle messages; everything
downstream consumes raw encoder-count deltas. Ported from ASClinic's
asc/roboclaw_for_motors.py (async-asc repo) -- see that package for the
inner-loop PID/feedforward controller this was originally paired with,
not yet ported here.

Constants below are grouped into two blocks: values confirmed against the
Aion R1's ArduPilot reference configuration or the Roboclaw/Basicmicro
protocol defaults, and values that are wiring/hardware-specific and CANNOT
be known without checking the actual R6 -- see the NEEDS VERIFICATION
block and pid_tuning_guide.pdf (system identification section) before
trusting them.
"""

import rclpy
from aion_msgs.msg import LeftRightFloat32, LeftRightInt32
from rclpy.node import Node

# --- Roboclaw/Basicmicro protocol constants (not platform-specific) ---
PERCENT_TO_ROBOCLAW = 327.67  # = 32767 / 100; int16 duty range per percent.

# --- Confirmed against the Aion R1 ArduPilot reference config ---
# (AION_R1_Rover.param, Rover V4.5.0 -- see pid_tuning_guide.pdf for the
# full derivation). The R1 does not use Roboclaw packet-serial at all
# (it drives via RC/PWM, SERVO1/3_FUNCTION), so this only confirms the
# Roboclaw's own factory-default address/baud, not anything R1-specific.
ADDRESS = 128    # Roboclaw factory default.
BAUDRATE = 38400  # Roboclaw factory default.

# --- NEEDS VERIFICATION on the actual R6 -- do not trust these blindly ---
# USB_PORT: device path depends on what's plugged into the R6's companion
# computer; confirm once wired up.
USB_PORT = "/dev/ttyACM0"
# LEFT/RIGHT_MOTOR_MULTIPLIER: sign depends on physical motor wiring/mount,
# which is unknown for this chassis. Verify via the Motion Studio "PWM
# Settings" slider test (move each motor's slider up, confirm it turns the
# expected direction and the encoder count moves the matching way) before
# trusting these signs or disabling DRY_RUN. ASClinic's R6 used -1.0/+1.0
# (left motor reversed) -- that was specific to their build's wiring, do
# not copy it without checking this chassis independently.
LEFT_MOTOR_MULTIPLIER = 1.0
RIGHT_MOTOR_MULTIPLIER = 1.0

# ENCODER_PERIOD_SEC: 10 Hz starting point for reading raw counts. Once a
# velocity control loop is built on top of this node, re-check this against
# the identified motor time constant (pid_tuning_guide.pdf, Section 2 --
# Discretization Check) before trusting it as a control-loop rate rather
# than just a read rate.
ENCODER_PERIOD_SEC = 0.1
MAX_DUTY_CYCLE = 100.0  # Safety cap on commanded duty cycle percentage.

# Set to True to test the whole ROS stack without opening the serial port or
# sending commands to physical motors. Defaults True here (unlike the
# ASClinic original) because the wiring/port constants above are unverified
# on this platform -- flip to False only after confirming them on the bench.
DRY_RUN = True


class RoboclawForMotorsNode(Node):
    def __init__(self):
        super().__init__("roboclaw_for_motors")

        self.address = ADDRESS
        self.max_duty_cycle = MAX_DUTY_CYCLE
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
            if not self.roboclaw.DutyM1M2(self.address, rc_left, rc_right):
                self.get_logger().warn("Roboclaw did not acknowledge DutyM1M2 command")

        out = LeftRightFloat32()
        out.left = float(duty_left)
        out.right = float(duty_right)
        out.seq_num = self.seq_num
        self.current_duty_publisher.publish(out)
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

        left = result[1] if result[1] < 0x80000000 else result[1] - 0x100000000
        right = result[2] if result[2] < 0x80000000 else result[2] - 0x100000000
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
