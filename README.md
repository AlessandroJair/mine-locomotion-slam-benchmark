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

Campaign of 2026-09-17/18, at commit `4cacede`: fixed-trajectory control with
the swept LiDAR, 5 runs per platform, two laps of the closed loop (~267 m of
ground truth per run). Figures, tables and per-run trajectories are in
[`results/`](results/); read **Known issues** before quoting any of it.

| Mean ± std, n=5 | Husky (differential) | Tracked | Rocker-bogie |
|---|---|---|---|
| ATE translational RMSE (online) [m] | 2.874 ± 0.385 | 4.107 ± 0.695 | **2.005 ± 0.486** |
| ATE optimised graph RMSE [m] | 0.141 ± 0.007 | 0.170 ± 0.007 | 0.166 ± 0.002 |
| RPE translational RMSE [m/m] | 0.0796 ± 0.0037 | 0.0772 ± 0.0009 | **0.0660 ± 0.0004** |
| Vibration RMS [m/s²] | 2.057 ± 0.189 | 4.362 ± 0.060 | **0.767 ± 0.022** |
| Peak abs(a_z − g) [m/s²] | 54.9 ± 6.2 | 78.1 ± 12.0 | **15.5 ± 9.6** |
| Pitch max abs [deg] | 33.6 ± 0.6 | 35.6 ± 0.8 | **30.1 ± 0.1** |

Three things to read off it, in the order they matter:

- **The spread lives in the odometry, not in the map.** The three optimised
  graphs land within 3 cm of each other while the online estimates differ by a
  factor of two. What this comparison measures is the pose the robot navigated
  on, not the map it ends up with.
- **The rocker-bogie is the quiet platform.** Its vibration RMS is 2.7× below
  the Husky's and 5.7× below the tracked robot's, and it carries the lowest
  online ATE and the lowest RPE.
- **No loop closure fired in any of the 15 runs.** The events are proximity
  detections only, with 0 false positives. The zeros in the loop-closure rows
  are a fact of this campaign, not a missing measurement.

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
│           ├── imu_covariance.py   # fills the IMU covariances the plugin leaves at 0
│           └── run_campaign.sh     # multi-trial campaign driver
└── results/                    # published campaign: figures, tables, trajectories
```

The vineyard experiment lives in a separate workspace, `~/journal_vineyard_comparison`.

### Which file Gazebo actually loads

Not the same for the three platforms, and getting it wrong is expensive:

| Platform | Spawned from | `robot_description` |
|----------|--------------|---------------------|
| Rocker-Bogie | `urdf/ensamblajeurdf.xacro` | same file |
| Differential | `model/differential/model.sdf` | `urdf_xacro/husky_gazebo.urdf.xacro` |
| Tracked | `model/two_track_robot/model.sdf` | `urdf_xacro/two_track_robot_gazebo.urdf.xacro` |

The two SDF-spawned platforms still load their xacro into `robot_description`,
because `gazebo_ros_control` reads the joint effort limits from there. **Both
files have to be kept in step**, and `scale_masses.py` edits the xacro, not the
SDF. `check_sim_parity.py` section 9 weighs whichever file is actually spawned,
precisely so this cannot drift unnoticed.

The tracked platform **must** be spawned from its SDF. It was briefly switched
to `spawn_model -urdf` from the xacro so that one definition would be
authoritative; the intent was right, but sdformat's URDF-to-SDF conversion
cannot express what `gazebo_continuous_track` needs and silently dropped three
things: the `<robotNamespace>` (which froze the whole simulation), the
`left_body` / `right_body` links (lumped away as fixed joints), and the
`<pose relative_to=...>` on every belt segment — which put **both tracks on the
chassis centreline**. `model/two_track_robot/` is therefore versioned, via an
exception in that package's `.gitignore`, which ignores `model/` because
upstream generates it.

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
| Velodyne VLP-16 LiDAR | 16 channels, 1875 samples/scan, 360° FOV, 10 Hz, 100 m range |
| IMU (6-axis) | 100 Hz, angular velocity + linear acceleration |
| RGB Camera | 640×480 px, 30 Hz |
| Wheel/track contact sensors | 100 Hz (2× the 50 Hz logging rate) |

The LiDAR range was 130 m until 2026-08-23. That figure was not a Velodyne
number: it is the default of `velodyne_simulator`, from which this project
inherited the plugin and the sampling pattern. The VLP-16 datasheet says 100 m.
The `<ray>` block is deliberately set one metre wider than the plugin's
`max_range`, and its `<min>` (0.3 m) is deliberately below the plugin's
`min_range` (0.9 m) — that is the reference design, not an inconsistency: the
0.9 m cutoff reproduces the near-field behaviour of the real
`velodyne_pointcloud` driver.

**IMU covariances.** `gazebo_ros_imu_sensor` fills the covariance diagonals from
its own `<gaussianNoise>` parameter — the same parameter that injects the noise.
This project needs every random draw to come from the seeded generator (the
`<noise>` blocks of the `<sensor>`, which `gzserver --seed` controls), so the
plugin's `gaussianNoise` is 0.0 and the covariances came out zero, i.e.
"unknown" per the `sensor_msgs/Imu` spec. `imu_covariance.py` closes the gap:
the plugin publishes raw on `/imu/data_raw` and that node republishes
`/imu/data` with the diagonals filled from the declared standard deviations.
The orientation variance is a modelling choice, not a measurement — Gazebo's
orientation is noiseless ground truth — and should be declared as such.

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
rosrun robot_metrics run_campaign.sh --fixed-trajectory          # see below
rosrun robot_metrics run_campaign.sh --fixed-trajectory --laps 3
```

`--fixed-trajectory` drives all three platforms along one ground-truth path
instead of navigating, so the SLAM is recorded but cannot steer. It is a second
experiment, not a replacement, and it writes to
`<output>/fixed_trajectory/`. `--laps N` repeats the closed loop N times: it
accumulates drift over *the same* geometry and multiplies the loop-closure
opportunities without inventing new terrain.

Since 2026-08-23 the trajectory follower shuts the run down when it finishes,
rather than idling until `--duration` expires. `--duration` is therefore a
safety net, not the stop condition; raise it freely for multi-lap runs. Before
that change every completed run was still stamped `timed_out: true` and burned
the remaining wall clock.

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
| `laps` | trajectory_follow | int | Times to repeat the closed loop (default 1) |
| `base_yaw_offset` | trajectory_follow | rad | Added to the model yaw to get the physical heading; π for the Rocker-Bogie, whose `base_link` faces −x |
| `viz_frame` | trajectory_follow | frame | Frame the reference and ground-truth paths are published in (default `map`) |

## Results

The published campaign is in [`results/`](results/): `paper_tables/` holds the
figures (PNG + EPS) and the summary tables, and `<robot>/runNN/` the per-run
ground truth, online estimate and optimised graph. `results/README.md` lists
what was left out of the repository — the RTAB-Map databases and the high-rate
logs, 7.0 GB in all — and gives the exact command that regenerates it.

Report the two SLAM accuracy figures separately and say which is
which. `ATE optimised graph RMSE` is the error of the final map, and is what the
SLAM literature means by ATE. The plain `ATE translational RMSE` is the online
estimate, the pose the robot actually navigated on. On a two-lap rocker_bogie
run the two differed by a factor of 5.6, because loop closure corrects the graph
behind the pose that has already been published. `aggregate_runs.py` emits both.
In the published campaign the gap is larger still: 2.0–4.1 m online against
0.14–0.17 m on the optimised graph.

## Known issues

Read this before collecting data for publication.

**Loop closure: fixed 2026-08-23, and the earlier diagnosis here was wrong.**
This section used to blame `Mem/UseOdomFeatures` and an OpenCV built without
`xfeatures2d`. Both of those warnings are real and both have been fixed
(`Kp/DetectorStrategy` and `Vis/FeatureType` are now both declared as 8,
GFTT/ORB, in all three `rtabmap_3d_slam.launch`), and fixing them changed
nothing: still zero loop closures. They were never the cause. Anyone reading
the old text was sent down a dead end, so here is what was actually measured.

The cause was **`RGBD/ProximityMaxGraphDepth`**, which shipped at 50.

Proximity detection applies two filters in series, and a candidate has to pass
both. `RGBD/LocalRadius` is spatial, in metres. `RGBD/ProximityMaxGraphDepth`
is topological, in hops along the pose graph, and a candidate beyond it is
discarded *before* its distance in metres is ever examined. One lap of this
route creates about 137 nodes, so on the second lap the node for the same
physical spot sat 140-200 hops back and was rejected on depth alone — however
precisely the robot returned, and however many laps it drove. **A limit of 50
makes closing any loop longer than ~50 nodes structurally impossible**, which
is why running two laps did not help.

Setting it to 0 — rtabmap's own value for "ignore" — cuts ATE by **79%**.
Measured by reprocessing one two-lap rocker_bogie run with `rtabmap-reprocess`,
so every row is the same recorded data:

| depth | radius | closures | ATE rmse (m) | ATE max (m) | links spanning >½ lap |
|------:|-------:|---------:|-------------:|------------:|----------------------:|
| 50 | 20 | 60 | 0.792 | 1.443 | **0** |
| 0 | 10 | 86 | 0.177 | 0.302 | 172 |
| 0 | 15 | 121 | 0.171 | 0.325 | 180 |
| **0** | **20** | 150 | **0.170** | 0.340 | 180 |

The maximum error falls with the mean, which is the signature of drift being
redistributed across the whole graph rather than smoothed locally. With the
depth filter open, `LocalRadius` is worth about 4% — inside the noise of a
single run. It is kept at 20 m only as margin, because the search runs on the
*estimated* poses and a platform that drifts more than this one needs the
revisit to still land inside the radius.

**Check it with the graph, never with the count.** rtabmap reports both kinds
of link as `Prox=N`, so the number alone cannot tell a revisit from a link to
the node that just left short-term memory. Read the `Link` table of the
database and look at `|from_id - to_id|`. At depth 50 every single link had a
gap of exactly 30 — `Mem/STMSize` — which is local stiffening and corrects no
accumulated drift; the 22% ATE gain that a `LocalRadius` sweep appeared to
show was that, and nothing more. At depth 0 the widest links are `204↔2` and
`205↔4`, tying the second lap to the first. **That gap is the difference
between local densification and loop closure, and the count is not.**

**The closures are geometric, not visual — `Prox=89, Loop=0`.** Visual loop
closure cannot work on this robot as built, and this is a property of the
sensor suite rather than a bug to fix. The camera publishes no depth and
nothing projects the LiDAR onto the image, so no keypoint ever gets a 3D
position: of 233,916 features stored across 263 nodes, **zero** carry one.
`Vis/EstimationType` defaults to 1 (3D→2D, PnP), which needs 3D geometry from
the older node, so the old side of every candidate contributed 0 and every
visual candidate was rejected with `Not enough features in images (old=0,
new=~850, min=20)` — the `min` there is `Vis/MinInliers`. Setting
`Vis/EstimationType=2` (2D→2D epipolar) does move the rejection downstream to
`Variance is too high!`, which confirms the mechanism, but still closes
nothing. Report this as LiDAR SLAM; it is what the VLP-16 is there for. If
visual loop closure is ever wanted, the cheapest route is rtabmap_util's
`pointcloud_to_depthimage` nodelet, which synthesises a depth image from the
Velodyne cloud without adding a sensor.

Still to confirm: these numbers come from reprocessing a single rocker_bogie
run. Validate the value live, and on the differential and tracked platforms,
before it goes into the paper.

**The continuous-track plugin is patched and diverges from upstream.**
`gazebo_continuous_track`'s `UpdateTrack` now repositions the idle belt variant
before disabling it, instead of only disabling it. Without that, the idle
variant stays put in the world while the vehicle drives off; the arc segments
are revolute and stay tethered, but the straight segments are prismatic with
±1e16 limits and slide unbounded — 2 of 16 belt links ended 27 m from the
chassis, visible as track-textured rectangles floating near the robot.
Reapply the patch if that submodule is ever updated.

**The physics timestep is 0.0005 s, not the Gazebo default 0.001.**
`gazebo_ros_control` integrates each drive joint's velocity PID explicitly at
the physics step, so the loop is stable only while `p < 2·I/dt`. The tracked
sprocket's 0.0152 kg·m² put the 30.4 limit below the `p = 36.975` its share of
the actuation budget demands, and the platform drove itself at ±4.5 rad/s with
zero command. Halving the timestep raises the limit to 60.8 and leaves every
declared gain intact, at about half the real-time factor. `sim_config.yaml`
carries the derivation.

**The Rocker-Bogie's `gt_yaw` is 180° from its physical heading — and it used
to destroy two of the reported metrics.** `gt_traj.tum` stores
`/gazebo/model_states` poses unmodified by design, and that platform's URDF
authors `base_link` facing −x. The trajectory follower corrects for it via
`base_yaw_offset`; the logged column is not corrected.

This section used to say "ATE is unaffected (Umeyama alignment absorbs a
constant rotation) and so is RPE (relative poses)". **Only the Umeyama ATE was
unaffected.** `evaluate()` builds `T_gt` from the logged yaw, and three metrics
take that orientation as their frame of reference: `align_origin`, which places
the estimate with `T_gt[0]`; `rpe`, which takes relative motion in the body
frame; and `final_drift_m`, which is measured against the origin-aligned
estimate. Measured on the fixed-trajectory campaign of 2026-08-25,
rocker_bogie run01:

| metric | reported | true |
|--------|---------:|-----:|
| ATE translational RMSE | 56.33 m | 2.14 m |
| RPE translational RMSE | 1.995 m/m | 0.060 m/m |

1.995 is not a coincidence: it is the error of travelling one metre backwards,
per metre travelled. The per-run table prints the origin-aligned figure as
**"ATE translational RMSE"**, so the headline SLAM number for that platform was
26× its real value, and any cross-platform comparison using it or RPE put the
Rocker-Bogie 20–40× behind on an artefact.

**Fixed 2026-08-25.** `slam_metrics.evaluate()` now measures the angle between
each trajectory's logged yaw and the direction it actually travels, rotates
both onto their own heading before anything else is computed, and records what
it removed as `gt_yaw_offset_deg` / `est_yaw_offset_deg`. The offset is
estimated from the trajectory rather than read from `base_yaw_offset`, so a
platform added later cannot inherit the bug silently. Expect ≈ 0° for the
differential and the tracked platform and ≈ 180° for the Rocker-Bogie; a value
that is neither, or one that moves between runs of the same platform, means the
estimate is not seeing a fixed convention and the run's numbers should not be
trusted. Correcting it left the other two platforms' figures within 1%.

Any plot comparing the raw `gt_yaw` column across platforms is still wrong.

**No `map` frame.** rtabmap builds its map but does not publish `map→odom`, so
RViz draws the reference and ground-truth paths — they are tagged `map` and
match the fixed frame, needing no lookup — but not the robot model or the point
cloud. The recorded data is unaffected: the SLAM pose comes from
`/rtabmap/odom`, which works.

**`ICP inliers` and `ICP matches` log as 0.0** while the inlier *ratio* (0.758)
and correspondence count (6649) are populated. Check those columns before they
go into a table.

**Peak metrics can rest on a single sample.** On the tracked run,
`alpha_x,max` was 215.9 rad/s² while the 99.9th percentile was 47.07 and the
99th was 21.0 — the maximum is 4.6× the next percentile. It occurs 47 s into the
route, so it is not the spawn transient, but report a high percentile alongside
the maximum rather than the maximum alone.

## License

MIT — see [LICENSE](LICENSE). The vendored `gazebo_continuous_track` plugin
(Yoshito Okada, MIT) keeps its own `LICENSE` file, and the mine world assets
under `world/mine/` are third-party models redistributed with the simulation.
