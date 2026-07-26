cmake_minimum_required(VERSION 3.15)
project(controller_eval_bindings)

# Standalone project - deliberately not part of the colcon workspace.
# geometry_msgs is only findable if CMAKE_PREFIX_PATH points at the ROS install:
#   cmake -B build -S . -DCMAKE_PREFIX_PATH=/opt/ros/humble
#   cmake --build build
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 REQUIRED)
find_package(geometry_msgs REQUIRED)

pybind11_add_module(action_chunk_controller_py controller_bindings.cpp)

target_include_directories(action_chunk_controller_py PRIVATE
  ${geometry_msgs_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/hw_interface/include
)
target_compile_features(action_chunk_controller_py PRIVATE cxx_std_17)
