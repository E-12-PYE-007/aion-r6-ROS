
#include <px4_interface/vel_tracking_mode.hpp>
#include <px4_ros2/components/node_with_mode.hpp>

#include "rclcpp/rclcpp.hpp"

using MyNodeWithMode = px4_ros2::NodeWithMode<RoverVelRateMode>;

static const std::string kNodeName = "rover_vel_rate_mode";
static const bool kEnableDebugOutput = true;

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MyNodeWithMode>(kNodeName, kEnableDebugOutput));
  rclcpp::shutdown();
  return 0;
}