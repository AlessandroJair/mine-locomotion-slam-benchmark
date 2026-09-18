cmake_minimum_required(VERSION 3.0.2)
project(robot_metrics)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
)

catkin_package()

# Executable entry points.  metrics_io.py, slam_metrics.py, model_parser.py and
# pointcloud_io.py are imported, not run, so they are installed with the rest of
# the scripts directory below rather than listed here.
catkin_install_python(PROGRAMS
  scripts/metrics_logger.py
  scripts/analyze_metrics.py
  scripts/aggregate_runs.py
  scripts/check_sim_parity.py
  scripts/extract_robot_specs.py
  scripts/extract_sensor_poses.py
  scripts/generate_waypoints.py
  scripts/map_vs_groundtruth.py
  scripts/plot_wheel_contact.py
  scripts/plot_ate_vs_attitude.py
  scripts/yaw_calibration.py
  scripts/analiza_calibracion.py
  scripts/wheel_odometry.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

# The whole scripts directory is installed so that the importable modules and
# run_campaign.sh travel with the package.
install(DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
  USE_SOURCE_PERMISSIONS
)
