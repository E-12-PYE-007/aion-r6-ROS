/****************************************************************************
 * Copyright (c) 2023 PX4 Development Team.
 * SPDX-License-Identifier: BSD-3-Clause
 ****************************************************************************/
#pragma once

#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/rover/position.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <px4_ros2/utils/geometry.hpp>
#include <rclcpp/rclcpp.hpp>

static const std::string kName = "Rover Position Mode";
// static const float kMaxSpeed = 2.5f;      // [m/s] Set equal to RO_SPEED_LIM
// static const float kMaxYawRate = 130.0f;  // [deg/s] Set equal to RO_YAW_RATE_LIM

class RoverPositionMode : public px4_ros2::ModeBase {
  public:
  explicit RoverPositionMode(rclcpp::Node& node) : ModeBase(node, kName)
  {
    _rover_position_setpoint = std::make_shared<px4_ros2::RoverPositionSetpointType>(*this);
    _vehicle_local_position = std::make_shared<px4_ros2::OdometryLocalPosition>(*this);
  }

  void onActivate() override {}

  void onDeactivate() override {}

  void updateSetpoint(float dt_s) override
  //Updates rover setpoint to a new position in NED frame
  {
    if (!_vehicle_local_position->positionXYValid()) {
      RCLCPP_WARN(node().get_logger(), "Local position not valid, cannot update setpoint");
      return;
    }
    Eigen::Vector3f position_ned = _vehicle_local_position->positionNed();
    Eigen::Vector2f position_ned_xy(position_ned(0), position_ned(1));

    Eigen::Vector2f target_position_ned = position_ned_xy + Eigen::Vector2f(10.0f, 5.0f); // Target position in NED frame, measured from current position

    _rover_position_setpoint->update(target_position_ned);
  }

  private: 
  std::shared_ptr<px4_ros2::RoverPositionSetpointType> _rover_position_setpoint;
  std::shared_ptr<px4_ros2::OdometryLocalPosition> _vehicle_local_position;
};