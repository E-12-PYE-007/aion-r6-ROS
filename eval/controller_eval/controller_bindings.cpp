/**
 * Python bindings for the ActionChunkController class and its derived classes, 
 * to enable evaluation of control strategy via Python scripts
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <hw_interface/action_chunk_controller.hpp>
#include <aion_msgs/msg/action_chunk.hpp>

namespace py = pybind11;

namespace {
// Shared conversion helpers, written once, used by every controller binding.
std::array<geometry_msgs::msg::Pose2D, 8> toChunk(const std::vector<std::array<double, 3>> & wp)
{
  std::array<geometry_msgs::msg::Pose2D, 8> chunk;
  for (std::size_t i = 0; i < 8; ++i) {
    chunk[i].x = wp[i][0];
    chunk[i].y = wp[i][1];
    chunk[i].theta = wp[i][2];
  }
  return chunk;
}

aion_msgs::msg::ActionChunk toActionChunk(const std::vector<std::array<double, 3>> & wp, uint32_t seq_num)
{
  aion_msgs::msg::ActionChunk action_chunk;
  action_chunk.seq_num = seq_num;
  action_chunk.relative_poses = toChunk(wp);
  return action_chunk;
}

geometry_msgs::msg::Pose2D toPose(const std::array<double, 3> & p)
{
  geometry_msgs::msg::Pose2D pose;
  pose.x = p[0];
  pose.y = p[1];
  pose.theta = p[2];
  return pose;
}
}

PYBIND11_MODULE(action_chunk_controller_py, m)
{
  py::class_<hw_interface::AsyncFeedForwardController>(m, "AsyncFeedForwardController")
    .def(py::init<>())
    .def("compute_command", [](hw_interface::AsyncFeedForwardController & self,
                                const std::vector<std::array<double,3>> & chunk,
                                const std::array<double,3> & pose) {
      // seq_num is irrelevant here - AsyncFeedForwardController is memoryless and
      // never looks at it, unlike PurePursuitController's chunk-change tracking.
      const auto cmd = self.computeCommand(toActionChunk(chunk, 0), toPose(pose));
      return std::make_pair(cmd.speed_body_x, cmd.yaw_rate);
    });

  py::class_<hw_interface::PurePursuitController>(m, "PurePursuitController")
    .def(py::init<>())
    .def("compute_command", [](hw_interface::PurePursuitController & self,
                                const std::vector<std::array<double,3>> & chunk,
                                uint32_t seq_num,
                                const std::array<double,3> & pose) {
      const auto cmd = self.computeCommand(toActionChunk(chunk, seq_num), toPose(pose));
      return std::make_pair(cmd.speed_body_x, cmd.yaw_rate);
    });

  // adding a new controller later is just another block like this one
}