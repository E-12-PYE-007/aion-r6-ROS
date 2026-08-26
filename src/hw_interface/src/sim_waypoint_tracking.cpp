/**
 * Replicates behaviour of px4_waypoint_tracking, but publishes to generic, Isaac-sim friendly topics
 * Control rate of 30Hz
 */


#include <hw_interface/action_chunk_controller.hpp>
#include "rclcpp/rclcpp.hpp"
#include <memory>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <aion_msgs/msg/action_chunk.hpp>
#include <optional>
# include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

# define CONTROL_RATE 30 // Control rate in Hz

using std::placeholders::_1;

class SimWaypointTracker : public rclcpp::Node
{
  public:
    SimWaypointTracker()
    : Node("sim_waypoint_tracker")
    {
        // Create a subscription to the pose topic that isaac publishes
        _odom_subscription = this->create_subscription<nav_msgs::msg::Odometry>(
            "/sim_odom", 10, std::bind(&SimWaypointTracker::odom_callback, this, _1));

        _action_chunk_subscription = this->create_subscription<aion_msgs::msg::ActionChunk>(
            "/vla/action_chunk", 10, std::bind(&SimWaypointTracker::action_chunk_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        _controller = std::make_unique<hw_interface::PurePursuitController>();

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000/CONTROL_RATE),
            std::bind(&SimWaypointTracker::timer_callback, this));
    }

    private:
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _odom_subscription;
        rclcpp::Subscription<aion_msgs::msg::ActionChunk>::SharedPtr _action_chunk_subscription;
        std::optional<aion_msgs::msg::ActionChunk> _current_action_chunk;
        std::unique_ptr<hw_interface::ActionChunkController> _controller;
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
        std::optional<geometry_msgs::msg::Pose2D> _current_pose;
        hw_interface::VelocityCommand vel_targets;


        void odom_callback(const std::shared_ptr<nav_msgs::msg::Odometry> msg){

            // Keep the tracker in the same runtime odom convention as the expert node.
            geometry_msgs::msg::Pose2D pose;
            pose.x = msg->pose.pose.position.x;
            pose.y = msg->pose.pose.position.y;
            pose.theta = tf2::getYaw(msg->pose.pose.orientation);
            _current_pose = pose;
        }

        void timer_callback(){
            if (!_current_pose.has_value() || !_current_action_chunk.has_value()) {
                RCLCPP_WARN(this->get_logger(), "No pose/action chunk received yet, skipping tick");
                return;
            }

            vel_targets = _controller->computeCommand(*_current_action_chunk, *_current_pose);
            auto msg = geometry_msgs::msg::Twist();

            msg.angular.z = vel_targets.yaw_rate;
            msg.linear.x = vel_targets.speed_body_x;

            RCLCPP_INFO(
                this->get_logger(),
                "Controller: linear=%.3f angular=%.3f target_idx=%u target=(%.3f, %.3f) dist=%.3f heading_error=%.3f curvature=%.3f raw_angular=%.3f found=%s",
                vel_targets.speed_body_x,
                vel_targets.yaw_rate,
                vel_targets.target_index,
                vel_targets.target_x,
                vel_targets.target_y,
                vel_targets.target_distance,
                vel_targets.heading_error,
                vel_targets.curvature,
                vel_targets.raw_yaw_rate,
                vel_targets.target_found ? "true" : "false"
            );
            publisher_->publish(msg);
        }

        void action_chunk_callback(const aion_msgs::msg::ActionChunk::SharedPtr msg)
        {
            RCLCPP_INFO(this->get_logger(), "Received ActionChunk seq=%u, %zu poses",
                        msg->seq_num, msg->relative_poses.size());

            for (const auto & pose : msg->relative_poses) {
            RCLCPP_INFO(this->get_logger(), "  pose: x=%.3f y=%.3f theta=%.3f",
                        pose.x, pose.y, pose.theta);
            }

            _current_action_chunk = *msg;
        }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimWaypointTracker>());
  rclcpp::shutdown();
  return 0;
}

