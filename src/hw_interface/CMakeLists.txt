cmake_minimum_required(VERSION 3.8)
project(hw_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(px4_ros2_cpp REQUIRED)
find_package(aion_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_executable(px4_waypoint_tracking src/px4_waypoint_tracking.cpp)
target_include_directories(px4_waypoint_tracking PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(px4_waypoint_tracking PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  px4_waypoint_tracking
  "rclcpp"
  "px4_msgs"
  "px4_ros2_cpp"
  "aion_msgs"
  "geometry_msgs"
)

install(TARGETS px4_waypoint_tracking
  DESTINATION lib/${PROJECT_NAME})

add_executable(sim_waypoint_tracking src/sim_waypoint_tracking.cpp)
target_include_directories(sim_waypoint_tracking PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(sim_waypoint_tracking PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  sim_waypoint_tracking
  "rclcpp"
  "nav_msgs"
  "aion_msgs"
  "geometry_msgs"
  "tf2"
  "tf2_geometry_msgs"
)

install(TARGETS sim_waypoint_tracking
  DESTINATION lib/${PROJECT_NAME})


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
