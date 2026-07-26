
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include <geometry_msgs/msg/pose2_d.hpp>

namespace hw_interface
{

/// Body-frame velocity command produced by an ActionChunkController for one control tick.
struct VelocityCommand
{
  float speed_body_x{0.0f}; // [m/s] Forward velocity in body frame
  float yaw_rate{0.0f};     // [rad/s] Yaw rate
};

/// Generic interface for turning a buffered action chunk into a velocity command.
/// current_pose is passed explicitly rather than held as member state set via a separate
/// setter - controllers that don't need it (like FixedWaypointFeedforwardController) just
/// ignore the parameter, and controllers that do (odometry-based ones, added later) get it
/// fresh on every call with no risk of reading stale or never-set pose state.
class ActionChunkController
{
  public:
    virtual ~ActionChunkController() = default;

    virtual VelocityCommand computeCommand(
      const std::array<geometry_msgs::msg::Pose2D, 8> & action_chunk,
      const geometry_msgs::msg::Pose2D & current_pose) = 0;
};

/// Direct port of the reference controller found in NHirose/AsyncVLA. This is a memoryless
/// feedforward P controller. It always targets the fixed midpoint waypoint of
/// whichever chunk it's given and computes one command from it directly.
class FixedWaypointFeedforwardController : public ActionChunkController
{
  public:
    VelocityCommand computeCommand(
      const std::array<geometry_msgs::msg::Pose2D, 8> & action_chunk,
      const geometry_msgs::msg::Pose2D & /* current_pose */) override
    {
      constexpr std::size_t kWaypointSelect = 4;
      const geometry_msgs::msg::Pose2D & chosen = action_chunk[kWaypointSelect];

      const double dx = chosen.x;
      const double dy = chosen.y;

      constexpr double kEps = 1e-8;
      constexpr double kDt = 1.0 / 3.0; // matches the source's DT - a gain constant, not a real time-to-target

      double linear_vel;
      double angular_vel;

      if (std::abs(dx) < kEps && std::abs(dy) < kEps) {
        // Already at the target position - just correct heading to match its predicted theta.
        linear_vel = 0.0;
        angular_vel = wrapToPi(chosen.theta) / kDt;
      } else if (std::abs(dx) < kEps) {
        // Target is directly to the side - a unicycle can't strafe, so rotate toward it.
        linear_vel = 0.0;
        angular_vel = sign(dy) * M_PI / (2.0 * kDt);
      } else {
        linear_vel = dx / kDt;
        angular_vel = std::atan(dy / dx) / kDt;
      }

      linear_vel = clamp(linear_vel, 0.0, 0.5);
      angular_vel = clamp(angular_vel, -1.0, 1.0);

      constexpr double kMaxV = 0.3;
      constexpr double kMaxW = 0.3;

      double linear_limited;
      double angular_limited;

      if (std::abs(linear_vel) <= kMaxV) {
        if (std::abs(angular_vel) <= kMaxW) {
          linear_limited = linear_vel;
          angular_limited = angular_vel;
        } else {
          const double rd = linear_vel / angular_vel;
          linear_limited = kMaxW * sign(linear_vel) * std::abs(rd);
          angular_limited = kMaxW * sign(angular_vel);
        }
      } else {
        if (std::abs(angular_vel) <= 0.001) {
          linear_limited = kMaxV * sign(linear_vel);
          angular_limited = 0.0;
        } else {
          const double rd = linear_vel / angular_vel;
          if (std::abs(rd) >= kMaxV / kMaxW) {
            linear_limited = kMaxV * sign(linear_vel);
            angular_limited = kMaxV * sign(angular_vel) / std::abs(rd);
          } else {
            linear_limited = kMaxW * sign(linear_vel) * std::abs(rd);
            angular_limited = kMaxW * sign(angular_vel);
          }
        }
      }

      VelocityCommand command;
      command.speed_body_x = static_cast<float>(linear_limited);
      command.yaw_rate = static_cast<float>(angular_limited);
      return command;
    }

  private:
    static double clamp(double value, double low, double high)
    {
      return std::max(low, std::min(high, value));
    }

    static double sign(double value)
    {
      return value < 0.0 ? -1.0 : 1.0;
    }

    static double wrapToPi(double angle)
    {
      return std::atan2(std::sin(angle), std::cos(angle));
    }
};

} // namespace hw_interface
