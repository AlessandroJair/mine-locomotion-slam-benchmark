# Published campaign data

Fixed-trajectory campaign with the swept LiDAR: each platform follows the same
route (two laps, ~267 m per run), 5 runs per platform.

| Platform | Campaign |
|----------|----------|
| tracked, rocker_bogie | 2026-09-17/18 |
| differential | 2026-09-29 (re-run with the corrected wheel radius, 0.148 m) |

```
results/
  campaign_meta.yaml   settings and seeds
  route_mine.yaml      the shared route
  sim_config.yaml      physics and sensor setup
  paper_tables/        tables and figures of the paper
  <robot>/run01..run05/
```

## Per run

| File | Content |
|------|---------|
| `run_meta.yaml` | Seed and run summary |
| `gt_traj.tum` | Ground truth |
| `est_traj.tum` | Online odometry |
| `opt_traj.tum` | Optimised RTAB-Map graph |
| `events.csv` | Proximity detections, loop closures, tracking losses |
| `columns.csv` | Column names and units of `metrics.csv` |
| `*.log` | Console logs |

Not included because of their size: `metrics.csv`, `gt_highrate.csv` and the
RTAB-Map databases. Re-running the campaign regenerates them:

```bash
rosrun robot_metrics run_campaign.sh --fixed-trajectory --lidar swept --laps 2 --runs 5 --output ~/metrics_VF
```

To rebuild only `paper_tables/` from data on disk:

```bash
rosrun robot_metrics run_campaign.sh --analyze-only --fixed-trajectory --lidar swept --runs 5 --output ~/metrics_VF
```

## Notes

- `ATE translational RMSE` is the online odometry; `ATE optimised graph RMSE`
  is the final map. They differ by more than ten times.
- Every number quoted in the paper is in
  `paper_tables/results_section_numbers.txt`.
- No loop closure was accepted in any run; the graph uses LiDAR proximity
  detections only.
