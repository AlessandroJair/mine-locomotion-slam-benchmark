# Published campaign data

Campaign of **2026-09-17/18**, run at commit `4cacede`, `control_mode:
fixed_trajectory` with the swept-LiDAR distortion enabled: each platform is
driven along the same ground-truth path (SLAM observes, it does not steer),
5 runs per platform, two laps of the 137.7 m closed loop (~267 m of ground
truth per run). Gazebo's seed for run *N* is `seed_base + N`, so any single run
can be replayed exactly; the per-run `run_meta.yaml` records it.

```
results/
├── campaign_meta.yaml      # control mode, seeds, parity check
├── route_mine.yaml         # the shared route, in world coordinates
├── sim_config.yaml         # spawn poses, physics, sensor standardisation
├── paper_tables/           # aggregated figures (PNG + EPS) and tables
└── <differential|tracked|rocker_bogie>/run01..run05/
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

Three artefacts per run were left out: together they are **7.0 GB**, and GitHub
rejects files over 100 MB.

| Excluded | Size | Why it is not needed to read the results |
|----------|------|------------------------------------------|
| `rtabmap.db` | 300–410 MB/run | The RTAB-Map database. `opt_traj.tum` and the SLAM counts in `run_meta.yaml` are already exported from it |
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
and rebuilds `paper_tables/` itself at the end. To rebuild only the tables and
figures from data already on disk:

```bash
rosrun robot_metrics aggregate_runs.py       --results <dir> --output_dir <dir>/paper_tables
rosrun robot_metrics plot_results_section.py --results <dir> --output_dir <dir>/paper_tables
```

## Reading the numbers

`ATE translational RMSE` is the **online** estimate (`/rtabmap/odom`), the pose
the robot navigated on. `ATE optimised graph RMSE` is the error of the **final
map**, and is what the SLAM literature calls ATE. In this campaign they differ
by more than an order of magnitude — the spread between platforms lives almost
entirely in the online odometry. Report both, and say which is which.

No loop closure fired in any of the 15 runs; the `events.csv` entries are
proximity detections (0 false positives). The zeros in the loop-closure rows of
`table_repeatability.txt` are a fact of this campaign, not a missing column.
