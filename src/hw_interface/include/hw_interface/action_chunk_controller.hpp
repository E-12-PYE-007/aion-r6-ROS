
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
      for(size_t i = _waypoint_idx; i<_waypoints.size();i++){
        std::array<double, 2> point = _waypoints[i];

        lookahead_vector = {
          point[0]-relative_position[0],
          point[1]-relative_position[1]
        };
        euclid_dist = std::hypot(lookahead_vector[0], lookahead_vector[1]);
        if (lookahead_vector[0] > kMinForwardTargetX && euclid_dist > best_forward_dist) {
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

      if (!found_lookahead) {
        if (best_forward_idx < 0) {
          // No usable forward target remains in the chunk.
          return VelocityCommand{};
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

      const double speed_target = trackingSpeedForHeading(heading_error, lookahead_vector[0]);
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

  /**
   * Interpolate curve between waypoints to allow consideration of specified theta,
   * using hemite curves
   */
    std::vector<std::array<double, 2>> generateWaypoints(
      const std::array<geometry_msgs::msg::Pose2D, 8> & waypoints)
    {
      std::vector<std::array<double, 2>> result;
      
      for (size_t i =0; i < waypoints.size()-1; ++i){
        // Start and end points
        double x0 = waypoints[i].x, y0 = waypoints[i].y;
        double x1 = waypoints[i+1].x, y1 = waypoints[i+1].y;
        
        // Scale factor based on distance between points to define curve shape
        double scale = std::hypot(x1-x0, y1-y0);

        // Scaled tangent vectors at start and end, derived from heading
        double m0x = scale * std::cos(waypoints[i].theta);
        double m0y = scale * std::sin(waypoints[i].theta);
        double m1x = scale * std::cos(waypoints[i+1].theta);
        double m1y = scale * std::sin(waypoints[i+1].theta);

        // Sample segment 10 times and store points
        for (int j = 0; j < 10; ++j) {
            double t = static_cast<double>(j) / 10;
            double t2 = t*t, t3 = t2*t;

            // Hermite basis polynomials
            double h00 =  2*t3 - 3*t2 + 1;  // 1 at t=0, 0 at t=1
            double h10 =    t3 - 2*t2 + t;  // tangent influence at t=0
            double h01 = -2*t3 + 3*t2;      // 0 at t=0, 1 at t=1
            double h11 =    t3 -   t2;      // tangent influence at t=1

            result.push_back({
                h00*x0 + h10*m0x + h01*x1 + h11*m1x,
                h00*y0 + h10*m0y + h01*y1 + h11*m1y
            });
        }
      }

      return result;

    }

    static constexpr double kHeadingGain = 1.2;
    static constexpr double kRotateInPlaceHeadingError = 0.5; // [rad]
    static constexpr double kMinLookaheadDistanceSq = 1e-6;
    static constexpr double kMinForwardTargetX = 0.03; // [m]
    static constexpr double kSlowdownHeadingError = 0.30; // [rad]
    static constexpr double kStopForwardHeadingError = 1.00; // [rad]
    static constexpr double kMinTrackingSpeed = 0.08; // [m/s]

    static double trackingSpeedForHeading(double heading_error, double target_x)
    {
      if (target_x <= kMinForwardTargetX) {
        return 0.0;
      }

      const double abs_heading_error = std::abs(heading_error);
      if (abs_heading_error <= kSlowdownHeadingError) {
        return kMaxV;
      }
      if (abs_heading_error >= kStopForwardHeadingError) {
        return kMinTrackingSpeed;
      }

      const double t =
        (abs_heading_error - kSlowdownHeadingError) /
        (kStopForwardHeadingError - kSlowdownHeadingError);
      return kMaxV + t * (kMinTrackingSpeed - kMaxV);
    }

    double _lookahead_distance = 0.45; // Lookahead distance in m
    std::vector<std::array<double, 2>> _waypoints; // Set of path waypoints generated from action chunk in x,y
    int _waypoint_idx = 0; // Index of current lookahead point

    uint32_t _last_chunk_id = UINT32_MAX; // Sentinel: no chunk received yet (seq_num never reaches this in practice)
    geometry_msgs::msg::Pose2D _anchor_pose; // current_pose captured when _last_chunk_id was set
};

} // namespace hw_interface
