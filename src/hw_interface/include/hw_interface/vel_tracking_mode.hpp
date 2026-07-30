
#pragma once

#include <memory>
#include <optional>

#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/rover/speed_rate.hpp>
#include <aion_msgs/msg/action_chunk.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <px4_ros2/utils/geometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>

#include "hw_interface/action_chunk_controller.hpp"

static const std::string kName = "Rover Velocity Rate Mode";

class RoverVelRateMode : public px4_ros2::ModeBase {
  public:
  explicit RoverVelRateMode(rclcpp::Node& node) : ModeBase(node, kName)
  {
    _rover_speed_rate_setpoint = std::make_shared<px4_ros2::RoverSpeedRateSetpointType>(*this);
    _vehicle_local_position = std::make_shared<px4_ros2::OdometryLocalPosition>(*this);
    _action_chunk_subscription = node.create_subscription<aion_msgs::msg::ActionChunk>(
      "/vla/action_chunk", 10, std::bind(&RoverVelRateMode::action_callback, this, std::placeholders::_1)
    );
    _controller = std::make_unique<hw_interface::AsyncFeedForwardController>();
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
  std::shared_ptr<px4_ros2::OdometryLocalPosition> _vehicle_local_position;
  rclcpp::Subscription<aion_msgs::msg::ActionChunk>::SharedPtr _action_chunk_subscription;

  std::optional<aion_msgs::msg::ActionChunk> _current_action_chunk;
  std::unique_ptr<hw_interface::ActionChunkController> _controller;

  hw_interface::VelocityCommand vel_targets;

  void trackActionChunk(hw_interface::VelocityCommand& v)
  {
    if (!_current_action_chunk.has_value()) {
      RCLCPP_WARN(node().get_logger(), "No action chunk received yet, commanding zero velocity");
      v = hw_interface::VelocityCommand{};
      return;
    }

    const Eigen::Vector3f position_ned = _vehicle_local_position->positionNed();
    geometry_msgs::msg::Pose2D current_pose;
    current_pose.x = position_ned.x();
    current_pose.y = position_ned.y();
    current_pose.theta = _vehicle_local_position->heading();

    v = _controller->computeCommand(*_current_action_chunk, current_pose);
  }

  void action_callback(const aion_msgs::msg::ActionChunk::SharedPtr msg)
  {
    RCLCPP_INFO(node().get_logger(), "Received ActionChunk seq=%u, %zu poses",
                msg->seq_num, msg->relative_poses.size());

    for (const auto & pose : msg->relative_poses) {
      RCLCPP_INFO(node().get_logger(), "  pose: x=%.3f y=%.3f theta=%.3f",
                  pose.x, pose.y, pose.theta);
    }

    _current_action_chunk = *msg;
  }

};