/**
 * Python bindings for the ActionChunkController class and its derived classes, 
 * to enable evaluation of control strategy via Python scripts
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <hw_interface/action_chunk_controller.hpp>

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
      const auto cmd = self.computeCommand(toChunk(chunk), toPose(pose));
      return std::make_pair(cmd.speed_body_x, cmd.yaw_rate);
    });

  // adding a new controller later is just another block like this one
}