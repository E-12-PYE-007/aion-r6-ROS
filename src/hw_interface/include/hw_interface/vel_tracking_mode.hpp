
#pragma once

#include <memory>

#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/rover/speed_rate.hpp>
#include <aion_msgs/msg/action_chunk.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <px4_ros2/utils/geometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>

#include "hw_interface/action_chunk_controller.hpp"

static const std::string kName = "Rover Velocity Rate Mode";
static const float kMaxSpeed = 2.0f; // [m/s] Set equal to RO_SPEED_LIM
static const float kMaxYawRate = 180.0f; // [deg/s] Set equal to RO_YAW_RATE_LIM

class RoverVelRateMode : public px4_ros2::ModeBase {
  public:
  explicit RoverVelRateMode(rclcpp::Node& node) : ModeBase(node, kName)
  {
    _rover_speed_rate_setpoint = std::make_shared<px4_ros2::RoverSpeedRateSetpointType>(*this);
    _action_chunk_subscription = node.create_subscription<aion_msgs::msg::ActionChunk>(
      "/vla/action_chunk", 10, std::bind(&RoverVelRateMode::action_callback, this, std::placeholders::_1)
    );
    _controller = std::make_unique<hw_interface::FixedWaypointFeedforwardController>();
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
  rclcpp::Subscription<aion_msgs::msg::ActionChunk>::SharedPtr _action_chunk_subscription;

   std::array<geometry_msgs::msg::Pose2D, 8> _current_action_chunk;
  int _current_action_chunk_index=0;
  std::unique_ptr<hw_interface::ActionChunkController> _controller;

  struct VelTargets
  {
    float speed_body_x{0.0f}; // [m/s] Forward velocity in body frame
    float yaw_rate{0.0f};     // [rad/s] Yaw rate in NED frame
  } vel_targets;

  void trackActionChunk(VelTargets& vel_targets)
  {
    const geometry_msgs::msg::Pose2D current_pose{}; // unused by FixedWaypointFeedforwardController
    const hw_interface::VelocityCommand command =
      _controller->computeCommand(_current_action_chunk, current_pose);
    vel_targets.speed_body_x = command.speed_body_x;
    vel_targets.yaw_rate = command.yaw_rate;
  }

  void action_callback(const aion_msgs::msg::ActionChunk::SharedPtr msg)
  {
    RCLCPP_INFO(node().get_logger(), "Received ActionChunk seq=%u, %zu poses",
                msg->seq_num, msg->relative_poses.size());

    for (const auto & pose : msg->relative_poses) {
      RCLCPP_INFO(node().get_logger(), "  pose: x=%.3f y=%.3f theta=%.3f",
                  pose.x, pose.y, pose.theta);
    }

    _current_action_chunk = msg->relative_poses;
  }

};