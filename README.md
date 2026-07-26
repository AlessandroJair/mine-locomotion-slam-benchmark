# Comparative Study: Rocker-Bogie vs. Differential vs. Tracked Robots in Underground Mine Environments

Comparative analysis of three robotic locomotion platforms evaluating mechanical stability and SLAM performance during autonomous navigation in a simulated underground mine tunnel network.

## Overview

This project implements a complete simulation and evaluation pipeline to compare:

| Platform | Description |
|----------|-------------|
| **Rocker-Bogie** | 6-wheel passive suspension mechanism |
| **Differential Drive** | Husky A200 equivalent, 4-wheel skid-steer |
| **Tracked** | Continuous track system |

Each robot traverses the same **137.7 m, 17-waypoint closed loop** (`mine_loop_2026`)
through an underground mine environment while collecting IMU, odometry, and SLAM
data for post-run comparative analysis.

The route is defined **once**, in Gazebo world coordinates, in
`robot_metrics/config/route_mine.yaml`. `generate_waypoints.py` projects it into
each robot's own map frame using that platform's spawn pose from
`sim_config.yaml`, producing the per-robot `config/waypoints_mine.yaml` files.
Those are **generated artefacts — do not edit them by hand**: doing so puts one
platform on a different route from the other two and invalidates the comparison.

### Key Findings

> **These figures are pre-revision.** They come from the earlier runs, when the
> three platforms carried hand-edited, mutually inconsistent waypoint files and
> were therefore *not driving the same route* — see the rationale at the top of
> `robot_metrics/config/route_mine.yaml`. They must be regenerated against
> `mine_loop_2026` before being cited. Raw data from those runs is in
> `~/metrics_output_prerevision`.

- Rocker-bogie achieves **54% reduction in vertical shock** vs. tracked vehicle
- Rocker-bogie achieves **lowest ATE (0.72m)** — 50% lower than Husky, 59% lower than tracked
- Rocker-bogie maintains the lowest roll angular acceleration (22.69 rad/s²)

## Project Structure

```
journal_comparison/
├── src/
│   ├── rocker_bogie/           # 6-wheel rocker-bogie robot package
│   │   ├── urdf/               # URDF/XACRO robot model
│   │   ├── launch/             # Gazebo, SLAM, navigation launch files
│   │   ├── scripts/            # Control and odometry scripts
│   │   ├── config/             # Navigation and controller parameters
│   │   ├── meshes/             # STL 3D meshes
│   │   └── world/              # Gazebo mine environment
│   ├── differential/           # Husky A200 equivalent package
│   │   ├── launch/
│   │   ├── scripts/
│   │   ├── urdf_xacro/
│   │   └── config/
│   ├── tracked/                # Tracked vehicle package
│   │   ├── gazebo_continuous_track/
│   │   └── gazebo_continuous_track_example/
│   └── robot_metrics/          # Metrics logging and analysis
│       ├── config/             # route_mine.yaml, sim_config.yaml
│       └── scripts/
│           ├── metrics_logger.py
│           ├── analyze_metrics.py
│           ├── aggregate_runs.py
│           ├── slam_metrics.py
│           ├── scale_masses.py     # mass normalization across platforms
│           └── run_campaign.sh     # multi-trial campaign driver
└── .mass_backup/               # original model.sdf masses before scaling
```

The vineyard experiment lives in a separate workspace, `~/journal_vineyard_comparison`.

## Technologies

- **ROS Noetic** — Robot Operating System framework
- **Gazebo 11** — 3D physics simulation
- **RTAB-Map** — Real-Time Appearance-Based SLAM
- **URDF/Xacro** — Robot modeling
- **move_base** — Autonomous navigation stack (DWA local planner + navfn global planner)
- **Python 3** — Control scripts, data analysis (pandas, numpy, matplotlib, scipy)

### Simulated Sensors

Identical across the three platforms — `check_sim_parity.py` enforces this, and
`extract_sensor_poses.py --standardize` reports the mounting poses.

| Sensor | Specs |
|--------|-------|
| Velodyne VLP-16 LiDAR | 16 channels, 1875 samples/scan, 360° FOV, 10 Hz, 130m range |
| IMU (6-axis) | 100 Hz, angular velocity + linear acceleration |
| RGB Camera | 640×480 px, 30 Hz |
| Wheel/track contact sensors | 100 Hz (2× the 50 Hz logging rate) |

## Prerequisites

- **Ubuntu 20.04 LTS**
- **ROS Noetic**
- **Gazebo 11+**
- **Python 3.8+** with `rospy`, `pandas`, `numpy`, `matplotlib`, `scipy`, `pyyaml`, `actionlib`
- ROS packages: `gazebo_ros`, `gazebo_ros_control`, `controller_manager`, `move_base`, `rtabmap_ros`, `robot_state_publisher`

### Build

```bash
cd ~/journal_comparison
catkin_make
source devel/setup.bash
```

## Running

### Full campaign (what produces the paper tables)

The experiment is 3 platforms x 3 runs, driven by a script rather than by hand
so that no parameter can drift between runs. It refuses to start unless
`check_sim_parity.py` confirms the three platforms share the same world, physics
and terrain, and `generate_waypoints.py --check` confirms no waypoint file has
been hand-edited away from the shared route.

```bash
rosrun robot_metrics run_campaign.sh --dry-run      # print what would happen
rosrun robot_metrics run_campaign.sh                # all 3 robots, runs 1-3
rosrun robot_metrics run_campaign.sh --robots rocker_bogie --runs 5
```

Results land in `~/metrics_output/<robot>/runNN/`, with aggregated tables and
figures in `~/metrics_output/paper_tables/`. Gazebo's RNG seed for run *N* is
`seed_base + N`, recorded in each `run_meta.yaml`, so any single run can be
replayed exactly.

Map quality is still scored by hand after the campaign:

```bash
rosrun rtabmap_ros rtabmap-export --cloud map.pcd ~/.ros/rtabmap.db
rosrun robot_metrics map_vs_groundtruth.py --map map.pcd \
    --run ~/metrics_output/<robot>/run01 --output_dir ~/metrics_output/paper_tables
```

### A single run by hand

Three terminals, in order — the world, then SLAM, then navigation:

```bash
# 1. World + robot  (pick one)
roslaunch rocker_bogie                    lcmine_rocker_bogie_world.launch paused:=false
roslaunch differential                    lcmine_husky_world.launch        paused:=false
roslaunch gazebo_continuous_track_example lcmine_two_track_world.launch    paused:=false

# 2. RTAB-Map (same package as the robot)
roslaunch <pkg> rtabmap_3d_slam.launch

# 3. Waypoint navigation + metrics logging
roslaunch <pkg> waypoint_navigation.launch run_id:=1
```

### Launch Parameters

| Parameter | Where | Values | Description |
|-----------|-------|--------|-------------|
| `paused` | world | `true/false` | Start Gazebo paused (default `true`) |
| `gui` | world | `true/false` | Run with/without the Gazebo client |
| `headless` | world | `true/false` | Suppress rendering entirely |
| `seed` | world | int | Gazebo RNG seed, for reproducible sensor noise |
| `differential_enabled` | world (rocker-bogie) | `true/false` | Enforce the rocker differential coupling |
| `use_move_base` | navigation | `true/false` | Navigation stack or the fallback follower |
| `use_rviz` | navigation | `true/false` | Open RViz (set `false` for batch runs) |
| `config_file` | navigation | path | Waypoint YAML (default `config/waypoints_mine.yaml`) |
| `run_id` | navigation | int | Repetition index; selects the output directory |
| `odom_topic` | navigation | topic | Odometry source (default `/rtabmap/odom`) |

## Results

> Pre-revision — see the note under **Key Findings**. Regenerate before citing.

### Mechanical Stability

| Metric | Husky | Tracked | Rocker-Bogie |
|--------|------:|--------:|-------------:|
| Max vertical accel. (m/s²) | 13.12 | 16.85 | **7.68** |
| Max roll angle (rad) | 0.577 | 0.503 | **0.459** |
| Max roll angular accel. (rad/s²) | 42.20 | 80.65 | **22.69** |

### SLAM Performance

| Metric | Husky | Tracked | Rocker-Bogie |
|--------|------:|--------:|-------------:|
| ATE (m) | 1.46 | 1.76 | **0.72** |
| Max ATE (m) | 2.34 | 3.04 | **1.23** |
| RPE (m) | 0.034 | **0.027** | 0.034 |

## License

This project is intended for academic and research purposes.
