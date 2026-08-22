
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <geometry_msgs/msg/pose2_d.hpp>
#include <aion_msgs/msg/action_chunk.hpp>

namespace hw_interface
{

/// Body-frame velocity command produced by an ActionChunkController for one control tick.
struct VelocityCommand
{
  float speed_body_x{0.0f};
  float yaw_rate{0.0f};

  int target_index{-1};
  float target_x{0.0f};
  float target_y{0.0f};
  float target_distance{0.0f};
  float heading_error{0.0f};
  float curvature{0.0f};
  float raw_yaw_rate{0.0f};
  bool target_found{false};
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
      const aion_msgs::msg::ActionChunk & action_chunk,
      const geometry_msgs::msg::Pose2D & current_pose) = 0;

  protected:
    static constexpr double kMaxV = 0.3; // [m/s] Shared rover speed limit across controllers
    static constexpr double kMaxW = 0.3; // [rad/s] Shared rover yaw rate limit across controllers

    static double sign(double value)
    {
      return value < 0.0 ? -1.0 : 1.0;
    }

    static double clamp(double value, double low, double high)
    {
      return std::max(low, std::min(high, value));
    }

    /// Combined linear/angular saturation: scales both together (preserving turning
    /// radius) rather than clamping each independently, so a command that needs to
    /// turn sharply doesn't get its curvature distorted by clamping yaw rate alone.
    static VelocityCommand saturate(double linear_vel, double angular_vel)
    {
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

    /// Independent linear/angular clamping for differential/skid-steer tracking.
    /// Unlike saturate(), this does not preserve turning radius by reducing forward
    /// speed. That is preferable for the Isaac Pure Pursuit tracker because the rover
    /// can command forward speed and yaw rate independently.
    static VelocityCommand clampIndependent(double linear_vel, double angular_vel)
    {
      VelocityCommand command;
      command.speed_body_x = static_cast<float>(clamp(linear_vel, -kMaxV, kMaxV));
      command.yaw_rate = static_cast<float>(clamp(angular_vel, -kMaxW, kMaxW));
      return command;
    }
};

/// Direct port of the reference controller found in NHirose/AsyncVLA. This is a memoryless
/// feedforward P controller. It always targets the fixed midpoint waypoint of
/// whichever chunk it's given and computes one command from it directly.
class AsyncFeedForwardController : public ActionChunkController
{
  public:
    VelocityCommand computeCommand(
      const aion_msgs::msg::ActionChunk & action_chunk,
      const geometry_msgs::msg::Pose2D & /* current_pose */) override
    {
      constexpr std::size_t kWaypointSelect = 4;
      const geometry_msgs::msg::Pose2D & chosen = action_chunk.relative_poses[kWaypointSelect];

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

      return saturate(linear_vel, angular_vel);
    }

  private:
    static double clamp(double value, double low, double high)
    {
      return std::max(low, std::min(high, value));
    }

    static double wrapToPi(double angle)
    {
      return std::atan2(std::sin(angle), std::cos(angle));
    }
};

class PurePursuitController: public ActionChunkController
{
  public:
    VelocityCommand computeCommand(
      const aion_msgs::msg::ActionChunk & action_chunk,
      const geometry_msgs::msg::Pose2D & current_pose) override
    {
      if (action_chunk.seq_num != _last_chunk_id) {
        // Anchor to current_pose here since the message doesn't carry the pose it was
        // actually conditioned on - approximates away VLA inference/transport latency.
        _anchor_pose = current_pose;
        _waypoints = generateWaypoints(action_chunk.relative_poses);
        _waypoint_idx = 0;
        _last_chunk_id = action_chunk.seq_num;
      }

      double dx = current_pose.x - _anchor_pose.x;         // Move origin of current pose to be relative to anchor pose
      double dy = current_pose.y - _anchor_pose.y;
      std::array<double, 2> relative_position{
        dx*std::cos(_anchor_pose.theta) + dy*std::sin(_anchor_pose.theta),
        -dx*std::sin(_anchor_pose.theta) + dy*std::cos(_anchor_pose.theta)      // Rotate displacement vector into anchor frame
      };
      std::array<double, 2> lookahead_vector{0.0, 0.0}; // Variable to store lookahead vector in
      double euclid_dist = 0.0; // length of lookahead vector

      bool found_lookahead = false;
      int best_forward_idx = -1;
      std::array<double, 2> best_forward_vector{0.0, 0.0};
      double best_forward_dist = 0.0;
      int best_any_idx = -1;
      std::array<double, 2> best_any_vector{0.0, 0.0};
      double best_any_dist = 0.0;
      for(size_t i = _waypoint_idx; i<_waypoints.size();i++){
        std::array<double, 2> point = _waypoints[i];

        lookahead_vector = {
          point[0]-relative_position[0],
          point[1]-relative_position[1]
        };
        euclid_dist = std::hypot(lookahead_vector[0], lookahead_vector[1]);
        if (euclid_dist > best_any_dist) {
          best_any_idx = static_cast<int>(i);
          best_any_vector = lookahead_vector;
          best_any_dist = euclid_dist;
        }
        if (lookahead_vector[0] > kMinForwardTargetX) {
          if (euclid_dist > best_forward_dist) {
            best_forward_idx = static_cast<int>(i);
            best_forward_vector = lookahead_vector;
            best_forward_dist = euclid_dist;
          }
          if (euclid_dist > _lookahead_distance){
            _waypoint_idx = i;
            found_lookahead = true;
            break;
          }
        }
      }

      if (!found_lookahead) {
        if (best_forward_idx < 0) {
          if (best_any_idx < 0 || best_any_dist <= kMinLookaheadDistanceSq) {
            // No usable target remains in the chunk.
            return VelocityCommand{};
          }

          // The current chunk has only side/behind targets. For a skid-steer rover,
          // rotate toward the nearest remaining path direction instead of freezing.
          _waypoint_idx = best_any_idx;
          double x = best_any_vector[0];
          double y = best_any_vector[1];
          double delta_theta = current_pose.theta - _anchor_pose.theta;
          std::array<double, 2> body_vector{
            x*std::cos(delta_theta) + y*std::sin(delta_theta),
            -x*std::sin(delta_theta) + y*std::cos(delta_theta)
          };
          const double heading_error = std::atan2(body_vector[1], body_vector[0]);
          VelocityCommand command = clampIndependent(0.0, kHeadingGain * heading_error);
          command.target_index = static_cast<int>(_waypoint_idx);
          command.target_x = static_cast<float>(body_vector[0]);
          command.target_y = static_cast<float>(body_vector[1]);
          command.target_distance = static_cast<float>(best_any_dist);
          command.heading_error = static_cast<float>(heading_error);
          command.curvature = 0.0f;
          command.raw_yaw_rate = static_cast<float>(kHeadingGain * heading_error);
          command.target_found = true;
          return command;
        }

        // Action chunks may be shorter than the nominal lookahead distance,
        // especially after velocity profiling or near task completion. Keep
        // tracking the farthest forward point instead of publishing zeros.
        _waypoint_idx = best_forward_idx;
        lookahead_vector = best_forward_vector;
        euclid_dist = best_forward_dist;
      }

      // Rotate lookahead vector from the anchor frame into the robot's live body frame
      double x = lookahead_vector[0];
      double y = lookahead_vector[1];
      double delta_theta = current_pose.theta - _anchor_pose.theta;
      lookahead_vector[0] =  x*std::cos(delta_theta) + y*std::sin(delta_theta);  // x_body
      lookahead_vector[1] = -x*std::sin(delta_theta) + y*std::cos(delta_theta);  // y_body

      double R = 2.0 * lookahead_vector[1] / (euclid_dist * euclid_dist);

      const double heading_error =
        std::atan2(lookahead_vector[1], lookahead_vector[0]);

      const double speed_target = trackingSpeedForTarget(
        heading_error,
        lookahead_vector[0],
        lookahead_vector[1],
        R);
      const double raw_yaw_rate = R * speed_target;

      VelocityCommand command = clampIndependent(speed_target, raw_yaw_rate);

      command.target_index = static_cast<int>(_waypoint_idx);
      command.target_x = static_cast<float>(lookahead_vector[0]);
      command.target_y = static_cast<float>(lookahead_vector[1]);
      command.target_distance = static_cast<float>(euclid_dist);
      command.heading_error = static_cast<float>(heading_error);
      command.curvature = static_cast<float>(R);
      command.raw_yaw_rate = static_cast<float>(raw_yaw_rate);
      command.target_found = true;

      return command;


      
    }
  
  private:

    /// Interpolate directly between action-chunk waypoint positions.
    /// Heading values in generated chunks describe path tangent, but using them as
    /// Hermite tangents can create loops/curls at corners and make Pure Pursuit
    /// chase artificial points behind the rover.
    std::vector<std::array<double, 2>> generateWaypoints(
      const std::array<geometry_msgs::msg::Pose2D, 8> & waypoints)
    {
      std::vector<std::array<double, 2>> result;
      
      for (size_t i =0; i < waypoints.size()-1; ++i){
        double x0 = waypoints[i].x, y0 = waypoints[i].y;
        double x1 = waypoints[i+1].x, y1 = waypoints[i+1].y;

        for (int j = 0; j < 10; ++j) {
            double t = static_cast<double>(j) / 10.0;
            result.push_back({
                x0 + t * (x1 - x0),
                y0 + t * (y1 - y0)
            });
        }
      }
      result.push_back({waypoints.back().x, waypoints.back().y});

      return result;

    }

    static constexpr double kHeadingGain = 1.2;
    static constexpr double kRotateInPlaceHeadingError = 0.5; // [rad]
    static constexpr double kMinLookaheadDistanceSq = 1e-6;
    static constexpr double kMinForwardTargetX = 0.03; // [m]
    static constexpr double kSlowdownHeadingError = 0.20; // [rad]
    static constexpr double kStopForwardHeadingError = 0.85; // [rad]
    static constexpr double kSlowdownLateralError = 0.12; // [m]
    static constexpr double kStopForwardLateralError = 0.45; // [m]
    static constexpr double kCurvatureSpeedScale = 0.30; // [m^2/s]
    static constexpr double kMinTrackingSpeed = 0.15; // [m/s]

    static double trackingSpeedForTarget(
      double heading_error,
      double target_x,
      double target_y,
      double curvature)
    {
      if (target_x <= kMinForwardTargetX) {
        return 0.0;
      }

      const double abs_heading_error = std::abs(heading_error);
      double heading_limited_speed = kMaxV;
      if (abs_heading_error >= kStopForwardHeadingError) {
        heading_limited_speed = kMinTrackingSpeed;
      } else if (abs_heading_error > kSlowdownHeadingError) {
        const double t =
          (abs_heading_error - kSlowdownHeadingError) /
          (kStopForwardHeadingError - kSlowdownHeadingError);
        heading_limited_speed = kMaxV + t * (kMinTrackingSpeed - kMaxV);
      }

      const double abs_lateral_error = std::abs(target_y);
      double lateral_limited_speed = kMaxV;
      if (abs_lateral_error >= kStopForwardLateralError) {
        lateral_limited_speed = kMinTrackingSpeed;
      } else if (abs_lateral_error > kSlowdownLateralError) {
        const double lateral_t =
          (abs_lateral_error - kSlowdownLateralError) /
          (kStopForwardLateralError - kSlowdownLateralError);
        lateral_limited_speed = kMaxV + lateral_t * (kMinTrackingSpeed - kMaxV);
      }

      double curvature_limited_speed = kMaxV;
      const double abs_curvature = std::abs(curvature);
      if (abs_curvature > 1e-6) {
        curvature_limited_speed = clamp(kCurvatureSpeedScale / abs_curvature, kMinTrackingSpeed, kMaxV);
      }

      return std::min({heading_limited_speed, lateral_limited_speed, curvature_limited_speed});
    }

    double _lookahead_distance = 0.35; // Lookahead distance in m
    std::vector<std::array<double, 2>> _waypoints; // Set of path waypoints generated from action chunk in x,y
    int _waypoint_idx = 0; // Index of current lookahead point

    uint32_t _last_chunk_id = UINT32_MAX; // Sentinel: no chunk received yet (seq_num never reaches this in practice)
    geometry_msgs::msg::Pose2D _anchor_pose; // current_pose captured when _last_chunk_id was set
};

} // namespace hw_interface
