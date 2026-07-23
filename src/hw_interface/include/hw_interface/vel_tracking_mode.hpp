
#pragma once

#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/rover/speed_rate.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <px4_ros2/utils/geometry.hpp>
#include <rclcpp/rclcpp.hpp>

static const std::string kName = "Rover Velocity Rate Mode";
static const float kMaxSpeed = 2.0f; // [m/s] Set equal to RO_SPEED_LIM
static const float kMaxYawRate = 180.0f; // [deg/s] Set equal to RO_YAW_RATE_LIM

class RoverVelRateMode : public px4_ros2::ModeBase {
  public:
  explicit RoverVelRateMode(rclcpp::Node& node) : ModeBase(node, kName)
  {
    _rover_speed_rate_setpoint = std::make_shared<px4_ros2::RoverSpeedRateSetpointType>(*this);
  }

  void onActivate() override {}
  // No tasks on activation

  void onDeactivate() override {}
  // No tasks on deactivation

  void updateSetpoint(float dt_s) override
  //Updates rover setpoint to velocity and yaw rate
  {

    trackActionChunk(vel_targets);

    _rover_speed_rate_setpoint->update(vel_targets.speed_body_x, vel_targets.yaw_rate);
  }

  private: 
  std::shared_ptr<px4_ros2::RoverSpeedRateSetpointType> _rover_speed_rate_setpoint;

  struct VelTargets
  {
    float speed_body_x{0.0f}; // [m/s] Forward velocity in body frame
    float yaw_rate{0.0f};     // [rad/s] Yaw rate in NED frame
  } vel_targets;

  void trackActionChunk(VelTargets& vel_targets)
  {
    // Placeholder for actual tracking logic
    // This function should compute the desired speed and yaw rate based on the current action chunk
    // For now, we will just set to produce a default curved path with a constant speed and yaw rate
    vel_targets.speed_body_x = 0.5*kMaxSpeed;
    vel_targets.yaw_rate = 0.2*kMaxYawRate * M_PI / 180.0f; // Convert to rad/s

  }
};