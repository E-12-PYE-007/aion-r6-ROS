/****************************************************************************
 * Copyright (c) 2023 PX4 Development Team.
 * SPDX-License-Identifier: BSD-3-Clause
 ****************************************************************************/

#include <hw_interface/mode.hpp>
#include <px4_ros2/components/node_with_mode.hpp>

#include "rclcpp/rclcpp.hpp"

using MyNodeWithMode = px4_ros2::NodeWithMode<RoverPositionMode>;

static const std::string kNodeName = "rover_position_mode";
static const bool kEnableDebugOutput = true;

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MyNodeWithMode>(kNodeName, kEnableDebugOutput));
  rclcpp::shutdown();
  return 0;
}