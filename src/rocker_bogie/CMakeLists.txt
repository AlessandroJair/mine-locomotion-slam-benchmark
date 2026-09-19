cmake_minimum_required(VERSION 3.0.2)
project(rocker_bogie)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  gazebo_ros
  gazebo_ros_control
  xacro
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs gazebo_ros gazebo_ros_control xacro
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/twist_to_wheels.py
  scripts/rocker_differential.py
  scripts/waypoint_navigator_yaml.py
  scripts/wheel_odom.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/ config/ urdf/ meshes/ world/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
