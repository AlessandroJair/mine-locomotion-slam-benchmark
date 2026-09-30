# robot_metrics

Runs the campaign, logs each run and computes the tables and figures of the
paper.

## Usage

```bash
source ~/journal_comparison/devel/setup.bash

# check the three platforms share world, physics and sensors
python3 src/robot_metrics/scripts/check_sim_parity.py

# campaign: 5 runs per platform, fixed trajectory, swept LiDAR
rosrun robot_metrics run_campaign.sh --fixed-trajectory --lidar swept --laps 2 --runs 5 --output ~/metrics_VF

# rebuild tables and figures only
rosrun robot_metrics run_campaign.sh --analyze-only --fixed-trajectory --lidar swept --runs 5 --output ~/metrics_VF
```

Useful options: `--robots "differential tracked"`, `--runs N`, `--gui`,
`--dry-run`, `--seed-base N`. Run *N* uses Gazebo seed `seed_base + N`, so
every run can be repeated.

If files were edited from Windows through `\\wsl.localhost`, restore the
executable bits before launching:

```bash
chmod +x src/*/scripts/*.py src/*/scripts/*.sh src/*/*/scripts/*.py
```

## Output

```
<output>/fixed_trajectory/swept_lidar/
  <robot>/runNN/
    metrics.csv       50 Hz log (units in each column name)
    gt_highrate.csv   1 kHz ground truth
    gt_traj.tum       ground truth
    est_traj.tum      online odometry
    opt_traj.tum      optimised RTAB-Map graph
    events.csv        proximity detections, loop closures, tracking losses
    run_meta.yaml     seed and run summary
  paper_tables/
    table_*.txt                    tables
    results_section_numbers.txt    every number quoted in the text
    *.png, *.eps                   figures
```

## Metrics

- **ATE**: absolute trajectory error of the online odometry, aligned at the
  start pose and with Umeyama; also for the optimised graph.
- **RPE**: relative pose error over 1 m of travel. Each odometry estimate is
  compared with the 1 kHz ground truth at its own time.
- **Vibration RMS**: from the 1 kHz ground-truth velocity (no IMU).
- **Per-sweep coupling**: correlation between chassis agitation during a
  LiDAR sweep and the odometry error of that sweep, with block bootstrap
  confidence intervals.

## Main scripts

| Script | What it does |
|---|---|
| `run_campaign.sh` | Runs the campaign and the analysis |
| `check_sim_parity.py` | Checks that the three platforms are set up alike |
| `metrics_logger.py` | Logs each run |
| `trajectory_follower.py` | Drives the robot along the fixed route |
| `slam_metrics.py` | ATE, RPE and correlations |
| `aggregate_runs.py` | Tables and correlation figures |
| `plot_results_section.py` | Results-section figures and `results_section_numbers.txt` |
| `generate_waypoints.py` | Builds each robot's waypoints from the shared route |

Only numpy, pandas, matplotlib and PyYAML are needed (no scipy).
