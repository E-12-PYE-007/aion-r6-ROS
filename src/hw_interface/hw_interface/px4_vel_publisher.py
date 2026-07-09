#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, DifferentialDriveSetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus

class px4_vel_publisher(Node):
    def __init__(self):
        super().__init__('px4_vel_publisher')
        self.get_logger().info('Node has been started.')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )  

        # Publishers

        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.differential_drive_setpoint_publisher = self.create_publisher(
            DifferentialDriveSetpoint, '/fmu/in/differential_drive_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)

        # Initialise variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()

        # Timer for publishing commands
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

    # Callback functions
    def vehicle_local_position_callback(self, vehicle_local_position):
        """ Update vehicle local position when new data published to the topic """
        self.vehicle_local_position = vehicle_local_position
    
    def vehicle_status_callback(self, vehicle_status):
        """ Update vehicle status when new data published to the topic """
        self.vehicle_status = vehicle_status

    # PX4 Mode switching
    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    # Publishing functions
    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_differential_drive_setpoint(self, speed: float, yaw_rate: float):
        """Publish the differential drive setpoint (body-frame speed + yaw rate)."""
        msg = DifferentialDriveSetpoint()
        msg.speed = speed
        msg.closed_loop_speed_control = True
        msg.yaw_rate = yaw_rate
        msg.closed_loop_yaw_rate_control = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.differential_drive_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing differential drive setpoint: speed={speed}, yaw_rate={yaw_rate}")

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal()

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_differential_drive_setpoint(0.5, 0.0)  # Move forward at 0.5 m/s, no turning

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1
                

def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    node = px4_vel_publisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
