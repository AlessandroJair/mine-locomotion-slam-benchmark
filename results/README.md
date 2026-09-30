# Published campaign data

`control_mode: fixed_trajectory` with the swept-LiDAR distortion enabled: each
platform is driven along the same ground-truth path (SLAM observes, it does not
steer), 5 runs per platform, two laps of the 137.7 m closed loop (266–273 m of
ground truth per run). Gazebo's seed for run *N* is `seed_base + N`, so any
single run can be replayed exactly; the per-run `run_meta.yaml` records it.

| Platform | Campaign | Commit |
|----------|----------|--------|
| tracked, rocker_bogie | 2026-09-17/18 | `4cacede` |
| differential | 2026-09-29, re-run | `51cbdb5` + `wheel_radius` 0.148 in `lcmine_husky_world.launch` |

The differential runs were repeated because its twist-to-wheel node converted
`cmd_vel` with the old plugin radius (0.17775 m) while the wheels roll on
0.148 m, so the platform drove at ~83 % of the commanded speed. Every run in
`differential/` logs `wheel_radius = 0.148` in its `gazebo.log`; its own
campaign record is `differential/campaign_meta.yaml`.

```
results/
├── campaign_meta.yaml      # control mode, seeds, parity check (tracked, rocker_bogie)
├── route_mine.yaml         # the shared route, in world coordinates
├── sim_config.yaml         # spawn poses, physics, sensor standardisation
├── paper_tables/           # aggregated figures (PNG + EPS) and tables
└── <differential|tracked|rocker_bogie>/run01..run05/
                            # differential/ also holds its campaign_meta.yaml
```

## Per run

| File | What it is |
|------|------------|
| `run_meta.yaml` | Seed, sample counts, `gt_distance_m`, SLAM event totals. A run is complete when `gt_distance_m` reaches the route length — not when the files exist |
| `gt_traj.tum` | Ground truth pose, 50 Hz, TUM format (`t tx ty tz qx qy qz qw`) |
| `est_traj.tum` | Online SLAM estimate — the pose the robot actually navigated on |
| `opt_traj.tum` | Poses of the optimised RTAB-Map graph, exported after the run |
| `events.csv` | Loop closures, proximity detections, tracking losses, with the ground-truth gap at each one |
| `columns.csv` | Column dictionary (name, unit, description) for `metrics.csv` — see below |
| `*.log` | `gazebo`, `slam`, `navigation` and `export` console logs, kept as provenance |

## What is *not* here

Three artefacts per run were left out: together they are several GB, and
GitHub rejects files over 100 MB.

| Excluded | Size | Why it is not needed to read the results |
|----------|------|------------------------------------------|
| `rtabmap.db` | a few hundred MB/run | The RTAB-Map database. `opt_traj.tum` and the SLAM counts in `run_meta.yaml` are already exported from it |
| `gt_highrate.csv` | 100–155 MB/run | 1 kHz ground truth, used for the attitude/vibration statistics. Those statistics are in `paper_tables/` |
| `metrics.csv` | ~30 MB/run | The 50 Hz log of every logged channel; `columns.csv` keeps its schema |

To regenerate them, re-run the campaign — the seed rule makes each run
reproducible. This is the command that produced everything in this directory:

```bash
rosrun robot_metrics run_campaign.sh \
    --fixed-trajectory --lidar swept --laps 2 --runs 5 --output ~/metrics_VF
```

It lands in `~/metrics_VF/fixed_trajectory/swept_lidar/` (`--fixed-trajectory`
and `--lidar swept` each add a level, because each is a different experiment)
and rebuilds `paper_tables/` itself at the end. The differential re-run used the
same command with `--robots differential` and a separate `--output`, and its
five runs were then copied into `differential/`. To rebuild only the tables and
figures from data already on disk (this is how `paper_tables/` was produced):

```bash
rosrun robot_metrics run_campaign.sh --analyze-only \
    --fixed-trajectory --lidar swept --runs 5 --output ~/metrics_VF
```

## Reading the numbers

`ATE translational RMSE` is the **online** estimate (`/rtabmap/odom`), the pose
the robot navigated on. `ATE optimised graph RMSE` is the error of the **final
map**, and is what the SLAM literature calls ATE. In this campaign they differ
by more than an order of magnitude — the spread between platforms lives almost
entirely in the online odometry. Report both, and say which is which.

`Vibration RMS` and the peak accelerations come from `gt_highrate.csv`: the
1 kHz ground-truth velocity is differentiated over the nominal 1 ms interval and
rotated into the body frame (`metrics_io._accel_from_twist`). The acceleration
columns stored by the logger are not used, because they sample a per-message
derivative and catch a rigid-contact impulse only when a stored row happens to
fall on it.

No loop closure fired in any of the 15 runs; the `events.csv` entries are
proximity detections (0 false positives). The zeros in the loop-closure rows of
`table_repeatability.txt` are a fact of this campaign, not a missing column.
