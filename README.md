# Rocker–bogie vs. differential vs. tracked robots in an underground mine

Simulation study (ROS Noetic, Gazebo 11, RTAB-Map) of how the locomotion system
of a ground robot affects LiDAR odometry in a mine.

| Platform | Description |
|----------|-------------|
| **Differential** | 4-wheel skid-steer, based on the Clearpath Husky |
| **Tracked** | Continuous-track robot |
| **Rocker–bogie** | 6 wheels, passive rocker–bogie suspension, Ackermann steering |

All three have the same mass (45 kg), sensors, sensor height, trajectory
follower and speed limit. Each one drives the same closed route through the
mine twice per run, 5 runs per platform, with a swept LiDAR model and no
de-skewing.

## Main results

Mean ± std over 5 runs (full tables in [`results/`](results/)):

| | Differential | Tracked | Rocker–bogie |
|---|---|---|---|
| ATE online, start-aligned [m] | 4.53 ± 0.41 | 4.12 ± 0.69 | **2.00 ± 0.49** |
| ATE online, Umeyama [m] | 1.31 ± 0.19 | **0.71 ± 0.29** | 0.97 ± 0.24 |
| ATE optimised graph [m] | 0.180 ± 0.009 | 0.170 ± 0.007 | 0.166 ± 0.002 |
| RPE translational [m/m] | 0.0700 ± 0.0035 | 0.0747 ± 0.0009 | **0.0626 ± 0.0003** |
| RPE rotational [deg/m] | 1.85 ± 0.14 | **1.60 ± 0.01** | 1.74 ± 0.02 |
| Vibration RMS [m/s²] | 1.67 ± 0.13 | 4.49 ± 0.04 | **0.86 ± 0.01** |

- The rocker–bogie vibrates least and has the lowest translational error.
- The tracked robot has the lowest rotational error.
- After pose-graph optimisation the three maps are practically the same.

## Structure

```
src/
  rocker_bogie/    rocker–bogie robot (URDF, launch, control)
  differential/    differential robot (Husky-based)
  tracked/         tracked robot (gazebo_continuous_track)
  robot_metrics/   logging, campaign script and analysis
results/           published data, tables and figures
```

## Build

```bash
cd ~/journal_comparison
catkin_make
source devel/setup.bash
```

Requires Ubuntu 20.04, ROS Noetic, Gazebo 11, RTAB-Map and Python 3 with
numpy, pandas, matplotlib and PyYAML.

## Run

```bash
# full campaign (the one in results/)
rosrun robot_metrics run_campaign.sh --fixed-trajectory --lidar swept --laps 2 --runs 5 --output ~/metrics_VF

# only rebuild tables and figures from existing runs
rosrun robot_metrics run_campaign.sh --analyze-only --fixed-trajectory --lidar swept --runs 5 --output ~/metrics_VF
```

See [`src/robot_metrics/README.md`](src/robot_metrics/README.md) for the
options and outputs.

## License

MIT, see [LICENSE](LICENSE). The `gazebo_continuous_track` plugin and the mine
world models keep their own licences.
