# robot_metrics — simulation and evaluation pipeline

Evaluation toolchain for the three-platform underground-mine exploration study
(Rocker-Bogie vs Husky differential vs tracked, Gazebo + ROS Noetic + RTAB-Map).

This package was reworked in response to the major-revision review. It now
covers repeatability, run-level reproducibility, ground-truth provenance,
wheel–terrain contact, physics parity, mass and sensor-placement matching, and
SLAM metrics beyond position error.

Everything asserted here is re-measured from the model files by a script, so it
can be checked rather than taken on trust: run `check_sim_parity.py`,
`extract_robot_specs.py` and `extract_sensor_poses.py`.

---

## 1. Findings from the audit

These were found while implementing the revisions. Several invalidate results
collected before the fix, so they are listed first.

### 1.1 The three robots were not running the same experiment

| What differed | Before | Now |
|---|---|---|
| Step obstacle | 2.5 m slab at y=−27.0 for Husky/Tracked, **3.0 m at y=−27.5** for the Rocker-Bogie | 2.5 m at y=−27.0 for all three |
| Boundary-wall friction | μ=100 for Husky/Rocker-Bogie, **absent (μ=1.0 default)** for Tracked | μ=100 for all three (no effect on results — see below) |
| Physics block | **absent from every world** → Gazebo defaults, unreportable | explicit ODE block, 1 ms step, 150 solver iterations |
| Terrain friction | **undeclared** on the mine mesh and ground plane | declared explicitly (μ=1.0, unchanged in value) |
| World file | robot spawned inside the world → three different files | byte-identical; robots spawn from their launch files |
| Waypoints | three hand-edited files on different routes | generated from one shared route |

**Any step-crossing or SLAM result predating these fixes is not comparable
across platforms and must be regenerated.** `check_sim_parity.py` now fails
loudly if any of this drifts again.

A note on that μ=100, since a reviewer opening the `.world` will query a
friction coefficient two orders of magnitude above any real material pair: the
six `smaze_wall` instances are site boundary markers, not gallery walls. They
sit at (−1.1, 0.18), (−31.5, −41.1), (74.0, −40.7), (64.8, −69.0),
(108.5, −69.7) and (86.2, −52.1), while the route occupies x ∈ [18, 54.5],
y ∈ [−39.5, 0.2] — the nearest is **~19 m** off-route, so no robot ever touches
one. The walls the robots actually drive between are the `lc_mine` mesh at
μ = 1.0. The missing `<surface>` block in the tracked package was a genuine
parity defect and was worth fixing; the value itself changes nothing.

Surfaces the robots can actually contact:

| Surface | μ | Contacted on this route |
|---|---|---|
| `lc_mine` mesh (floor and gallery walls) | 1.0 | yes, throughout |
| step / `bump` slab | 1.5 | yes, when crossing |
| `ramp` (terrain patching, see §8) | 1.5 | no |
| `smaze_wall` ×6 (boundary markers) | 100 | no, ≥19 m away |
| `grass_plane` | 1.0 | no, at x = −70 |

### 1.2 The Rocker-Bogie suspension was not working

Three independent problems, all of which made the platform behave like a rigid
chassis — which is precisely what the paper claims it is not:

1. **Left and right were welded together.** `differential_bar_joint` was a
   *fixed* joint, so both rocker tubes formed one rigid body carrying all six
   wheels, hanging off the chassis by a single pivot. There was no independent
   left/right travel at all. The joints actually *named* `rocker_joint_l/r`
   were not in the load path — their children are leaf links carrying no wheel.
2. **The passive joints were being actively held still.** `differential_joint`,
   `rocker_joint_l/r`, `rev_3` and `rev_4` each carried a
   `VelocityJointInterface` transmission labelled "(passive)".
   `gazebo_ros_control` registers every transmitted joint and drives it toward
   its command every cycle; with no controller claiming them that command was a
   constant **zero velocity**. On top of that the URDF specified
   `damping=80 N·m·s/rad` and `friction=4–10 N·m`.
3. **The travel limit made the step geometrically impossible.** The rocker
   limit was ±0.15 rad. Lifting one wheel over the 0.10 m step needs
   `atan(0.10/0.285) ≈ 0.34 rad`. The joint hit its stop and levered the
   chassis up instead of absorbing the step.

Now: a proper coaxial rocker pair on `base_link`, each carrying a bogie
(`rev_3`/`rev_4`) plus a rear wheel; no transmissions on the bogies;
damping 0.5, friction 0.2; limits ±0.6 rad. Verified with
`extract_robot_specs.py --wheel-map`:

```
nav position   link        suspension chain to base_link      steer   drive
front-left     Rueda_6_1   rocker_pivot_left                  rev_5   rev_11
front-right    Rueda_1_1   rocker_pivot_right                 rev_7   rev_12
middle-left    Rueda_4_1   rev_3 -> rocker_pivot_left           -     rev_10
middle-right   Rueda_3_1   rev_4 -> rocker_pivot_right          -     rev_13
rear-left      Rueda_2_1   rev_3 -> rocker_pivot_left         rev_6   rev_9
rear-right     Rueda_5_1   rev_4 -> rocker_pivot_right        rev_8   rev_14
```

That is the standard arrangement: the front wheel hangs directly off the
rocker, and the bogie at the other end of the rocker carries the middle and
rear wheels.

### 1.3 There was no rocker differential

Nothing coupled the two sides. `rocker_differential.py` now enforces
`φ_left + φ_right = 0` with a stiff torque coupling (see §4). Disable it with
`differential_enabled:=false` for the ablation.

### 1.4 The drive controller was mirrored, and its geometry was wrong

`base_link` is authored facing −x (camera at x=−0.07 with yaw=π, robot spawns
at Y=π). With z up, **left is −y, not +y**. The wheel-velocity controllers had
been labelled in raw base_link axes, so every one was bound to the
diametrically opposite wheel. Front/rear swapping is harmless (the Ackermann
solution gives them the same speed) but **left/right swapping is not**: inner
wheel speeds went to the outer wheels in every turn, dragging all six wheels
sideways through every corner. That inflates both the slip metric and the IMU
vibration signal the paper attributes to locomotion type.

The steering map was already correct and is unchanged.

Geometry passed to the Ackermann solver was also wrong:

| Parameter | Was | Measured from URDF |
|---|---|---|
| wheelbase | 0.57 m | **0.8532 m** |
| track | 0.6775 m | **0.7821 m** rear/middle, **0.5959 m** front |

The old wheelbase was 33% short, so every commanded yaw rate produced too
small a steering angle and the robot understeered and scrubbed.

### 1.5 Two confounds, now removed

`extract_robot_specs.py` and `extract_sensor_poses.py` found two differences
between the platforms large enough to explain a SLAM result on their own. Both
have been eliminated in the models; the scripts re-measure them on every run,
so the claim is checkable rather than asserted.

**Mass — was 94.9 / 47.0 / 45.3 kg, now 45.0 kg for all three.**

The Rocker-Bogie weighed more than twice the others. At equal wheel friction
coefficient traction force scales with normal force, and at equal motor torque
the light platforms accelerate harder, so almost any locomotion result could
have been attributed to mass instead of to the suspension.

`scale_masses.py` multiplies every `<mass>` and every `<inertia>` component of
each model by one factor per platform (0.474039 / 0.958120 / 0.994392). Uniform
scaling is the point: it is the same body at a different density, so the centre
of gravity and every geometric property stay exactly where they were. Measured
before and after, the CoG heights are unchanged to 4 decimal places — 0.3412 m
Husky, 0.1795 m Tracked, 0.4555 m Rocker-Bogie. Set `mass_scale` to 1.0 (or
divide the Husky SDF literals by its factor) to recover the native models;
`.mass_backup/` holds them.

Note the tracked robot's native mass is **46.967 kg**, not the 43.967 kg an
earlier pass reported: 3.0 kg of track segments live in `<gazebo>` blocks that
`gazebo_continuous_track` instantiates, so they never appear as URDF links.
`model_parser.py` now counts them and the spec table reports the split.

Two settings had to follow the mass down, because they close effort onto
inertia and `wn = sqrt(K/J)` would otherwise have moved: the Rocker-Bogie wheel
and steering PID gains, and the rocker differential's stiffness/damping
(2000/60 → 948.08/28.44 N·m/rad). Both are commented at their definitions.

**Sensor height — was a 0.26 m spread, now under 1 mm.**

Native LiDAR heights above the contact plane were 0.8248 m (Rocker-Bogie),
0.5805 m (Husky) and 0.5657 m (Tracked), so a 1° tilt displaced the
Rocker-Bogie's LiDAR 14.4 mm against the Tracked robot's 9.9 mm.

All three now sit at the Rocker-Bogie's heights: LiDAR 0.8248 m, camera
0.7607 m, IMU 0.6969 m, each 0.3543 m ahead of the contact-polygon centre.
Standardizing **upwards** rather than downwards is forced by geometry — the
Rocker-Bogie's 0.638 m ground clearance puts the underside of its chassis above
both native low mounts, so a sensor at 0.57 m would be inside the chassis. The
masts on the two low platforms are modelled as massless with the sensor masses
retained, which raises the Husky CoG by 4.66 mm and the tracked robot's by
0.68 mm — negligible against their 341 and 180 mm CoG heights, and in the
direction that makes the low platforms slightly *less* stable, so the choice
does not favour the Rocker-Bogie.

Every edited element carries a `SENSOR STANDARDIZATION` comment with its native
value and the arithmetic. `extract_sensor_poses.py --standardize` regenerates
`sensor_standardization.yaml` from the models, so an accidental edit shows up
as a non-zero spread at the bottom of the table.

One correction to note if the earlier figures were used anywhere: the Tracked
robot's native LiDAR height is 0.5657 m, not 0.528 m — the 0.528 figure omitted
the 0.0377 m offset from the mount link to the Velodyne's own ray origin.

### 1.6 Reproducible runs, and the LiDAR noise figure

**What actually differed between the three runs of a robot: nothing.** Same
world, same launch, same waypoints, same spawn pose, same physics. `run_id` only
named the output directory. All run-to-run variation came from unseeded sensor
noise plus real-time scheduling jitter — which is the *right* thing for a
repeatability claim to measure, but it meant no run could ever be repeated.

Gazebo now gets an explicit seed: run N of every robot launches with
`gzserver --seed (seed_base + N)`, so runs 1/2/3 use seeds 1/2/3. Different
seeds still give different noise, so the reported standard deviation measures
exactly what it did before — but each run is anchored:

```bash
roslaunch rocker_bogie lcmine_rocker_bogie_world.launch seed:=2   # replay run 2
./run_campaign.sh --seed-base 100                                 # independent replication
```

The seed is written to each `run_meta.yaml` *before* the run starts, so it
survives a crash, and to `campaign_meta.yaml`.

**The LiDAR noise figure was wrong — report 0.011314 m, not 0.008.** All three
models declared `<stddev>0.008</stddev>` inside the sensor *and*
`<gaussianNoise>0.008</gaussianNoise>` inside the plugin. Those are two
independent Gaussians, so they add in quadrature: the LiDAR really saw
0.008·√2 = 0.011314 m.

Worse, only one half was seedable. `nm -D` on the compiled plugins shows
`libgazebo_ros_velodyne_gpu_laser.so` imports the C library's `rand()`, which
the gzserver seed does not reach, while `libgazebo_ros_imu_sensor.so` uses
`ignition::math::Rand`, which it does. All LiDAR noise has therefore been moved
into the sensor's own seeded `<noise>` block at 0.011314 m and the plugin's
`<gaussianNoise>` set to 0.0. **The effective noise level is unchanged** — this
does not alter the experiment, it only makes it reproducible and correctly
stated.

`check_sim_parity.py` check 7 now asserts all of this: that the three platforms
declare identical noise, that no non-zero noise comes from an unseeded
generator, and that the LiDAR value matches `sim_config.yaml`. Verified against
an injected regression — it fails as it should.

---

## 2. Running the experiment

### 2.0 Setup, once per session

```bash
cd ~/journal_comparison
catkin_make
source devel/setup.bash
```

If any file was edited from Windows through `\\wsl.localhost`, restore the
executable bits first. That path drops them, and **roslaunch refuses to start a
node that is not executable** — the campaign dies on the first run with a
`cannot launch node` error that does not obviously point at permissions:

```bash
chmod +x src/*/scripts/*.py src/*/scripts/*.sh src/*/*/scripts/*.py
```

### 2.1 Preflight — check before spending hours

```bash
python3 src/robot_metrics/scripts/check_sim_parity.py
python3 src/robot_metrics/scripts/generate_waypoints.py --check
src/robot_metrics/scripts/run_campaign.sh --dry-run
```

All three must come back clean. Variations:

```bash
# regenerate the waypoints if --check calls them stale
python3 src/robot_metrics/scripts/generate_waypoints.py

# regenerate and plot the route over the map
python3 src/robot_metrics/scripts/generate_waypoints.py --plot --output_dir ~/paper_tables

# check a different tree (a copy, a branch)
python3 src/robot_metrics/scripts/check_sim_parity.py --src /path/to/src

# model tables, no simulation needed
python3 src/robot_metrics/scripts/extract_robot_specs.py --output_dir ~/paper_tables
python3 src/robot_metrics/scripts/extract_robot_specs.py --wheel-map
python3 src/robot_metrics/scripts/extract_sensor_poses.py --output_dir ~/paper_tables --standardize

# report the mass matching without writing (already applied: says "already scaled")
python3 src/robot_metrics/scripts/scale_masses.py --src src
```

### 2.2 The campaign

Nine runs at up to 20 min each — budget three hours. Do a **single run with the
GUI first**, so a mistake costs 20 minutes instead of three hours. Start with
the Rocker-Bogie: it changed the most (mass, suspension, differential, Ackermann
controller, gains) and is where a problem is most likely to show:

```bash
src/robot_metrics/scripts/run_campaign.sh --robots rocker_bogie --runs 1 --gui
```

Then the full campaign:

```bash
src/robot_metrics/scripts/run_campaign.sh 2>&1 | tee ~/campaign.log
```

Variations:

```bash
# one platform, all 3 runs
src/robot_metrics/scripts/run_campaign.sh --robots rocker_bogie

# two named platforms (one quoted string)
src/robot_metrics/scripts/run_campaign.sh --robots "differential tracked"

# more repetitions -> a more trustworthy standard deviation
src/robot_metrics/scripts/run_campaign.sh --runs 5

# short cap, for a smoke test
src/robot_metrics/scripts/run_campaign.sh --runs 1 --duration 300

# somewhere else, so the good campaign is not overwritten
src/robot_metrics/scripts/run_campaign.sh --output ~/metrics_test

# independent replication: seeds 101,102,103 instead of 1,2,3
src/robot_metrics/scripts/run_campaign.sh --seed-base 100 --output ~/metrics_rep2

# replay ONE run by hand (three terminals)
roslaunch rocker_bogie lcmine_rocker_bogie_world.launch seed:=2
roslaunch rocker_bogie rtabmap_3d_slam.launch
roslaunch rocker_bogie waypoint_navigation.launch run_id:=2

# ablation: pivots free, no differential coupling
roslaunch rocker_bogie lcmine_rocker_bogie_world.launch differential_enabled:=false
```

Afterwards, confirm nothing hit the time cap and that all nine runs produced
data:

```bash
grep -r timed_out ~/metrics_output/*/run*/run_meta.yaml
ls ~/metrics_output/*/run*/metrics.csv | wc -l    # expect 9
```

### 2.3 Maps, and re-aggregating

The campaign writes the tables and figures itself. This is what is left by hand:

```bash
rosrun rtabmap_ros rtabmap-export --cloud map.pcd ~/.ros/rtabmap.db

python3 src/robot_metrics/scripts/map_vs_groundtruth.py \
    --map map.pcd --run ~/metrics_output/rocker_bogie/run01 \
    --output_dir ~/metrics_output/paper_tables
```

Variations:

```bash
# a different F-score threshold (default 0.20 m)
python3 src/robot_metrics/scripts/map_vs_groundtruth.py \
    --map map.pcd --run ~/metrics_output/rocker_bogie/run01 --tau 0.10

# very dense cloud: downsample first, much faster
python3 src/robot_metrics/scripts/map_vs_groundtruth.py \
    --map map.pcd --run ~/metrics_output/rocker_bogie/run01 --voxel 0.05

# limit the reference geometry to a different sensor range
python3 src/robot_metrics/scripts/map_vs_groundtruth.py \
    --map map.pcd --run ~/metrics_output/rocker_bogie/run01 --range 20

# re-aggregate without re-running anything (after deleting a bad run, say)
python3 src/robot_metrics/scripts/aggregate_runs.py \
    --results ~/metrics_output --output_dir ~/metrics_output/paper_tables

# contact figure: the step window, or the whole run
python3 src/robot_metrics/scripts/plot_wheel_contact.py \
    --run ~/metrics_output/rocker_bogie/run01 --output_dir ~/metrics_output/paper_tables
python3 src/robot_metrics/scripts/plot_wheel_contact.py \
    --run ~/metrics_output/rocker_bogie/run01 --full
```

### 2.4 Measuring the residual non-determinism

Worth doing before treating the numbers as final. Both of these use `seed=1`, so
any difference between them is what the seed does *not* control — ROS message
timing — and that is the floor under the reported standard deviation:

```bash
src/robot_metrics/scripts/run_campaign.sh --robots rocker_bogie --runs 1 --output ~/rep_a
src/robot_metrics/scripts/run_campaign.sh --robots rocker_bogie --runs 1 --output ~/rep_b
diff <(cut -d, -f2-6 ~/rep_a/rocker_bogie/run01/metrics.csv) \
     <(cut -d, -f2-6 ~/rep_b/rocker_bogie/run01/metrics.csv) | head
```

Short output means the setup is near-deterministic. Long output means that is
the irreducible variation, and it belongs in the paper.

---

## 3. Output layout

```
~/metrics_output/
  campaign_meta.yaml          git commit, date, settings, seed_base and rule
  sim_config.yaml             copy of the configuration actually used
  route_mine.yaml             copy of the route actually driven
  <robot>/run01/
      metrics.csv             50 Hz time series, units in every header
      columns.csv             data dictionary: column, unit, description
      gt_traj.tum             raw ground truth, TUM format
      est_traj.tum            SLAM estimate, TUM format
      events.csv              tracking losses, relocalizations, loop closures
      run_meta.yaml           provenance, incl. the Gazebo seed (written
                              before the run, so it survives a crash)
  paper_tables/
      table_per_run.txt              every run reported separately
      table_repeatability.txt        mean +/- std across runs
      table_stability_correlation.txt
      table_robot_specs.txt / robot_specs.csv
      table_sensor_poses.txt / sensor_poses.csv
      sensor_standardization.yaml
      per_run_metrics.csv, summary_mean_std.csv
      ate_vs_time, ate_vs_distance, stability_vs_error,
      repeatability_summary, wheel_contact_step_rocker_bogie   (.eps + .png)
```

Every CSV column is named `name[unit]` — `pitch[rad]`, `accel_x[m/s^2]`,
`normal_force_Rueda_1_1[N]`. `metrics_io.load_metrics()` strips the suffix on
load and exposes it as `df.attrs['units']`, so units propagate into the
generated tables automatically instead of being remembered by hand.

---

## 4. Ground truth — provenance

Asked for explicitly in the review; also recorded in `config/sim_config.yaml`
and in every `run_meta.yaml`.

- **Source:** `/gazebo/model_states` (`gazebo_msgs/ModelStates`).
- **Processing:** none. It is the exact pose integrated by ODE — no filter, no
  smoothing, no added noise, no estimator in between.
- **Frame:** the model's canonical link (`base_link`), the same frame the SLAM
  estimate uses, so ATE/RPE need no extrinsic compensation.
- **Timestamping:** `ModelStates` carries no header stamp, so each sample is
  stamped with the ROS sim clock (`use_sim_time:=true`) at reception. The topic
  publishes at ~1 kHz and logging runs at 50 Hz, so the error is bounded by one
  1 ms physics step — three orders of magnitude below the ATE values of
  interest.
- **Raw record:** `gt_traj.tum`, unmodified, in TUM format so the numbers can
  be cross-checked with `evo` or the TUM tools.

---

## 5. Metrics

### Trajectory error (`slam_metrics.py`)

Everything is split into translational and rotational components, because a
platform that pitches its sensor mast degrades rotation first and translation
only as a consequence:

- `ate_origin_*` — estimate placed at the GT start pose, nothing else fitted.
  This is what a navigation stack actually experiences.
- `ate_umeyama_*` — least-squares rigid SE(3) alignment (Umeyama 1991, no
  scale). The TUM/KITTI convention, comparable with the literature.
- `rpe_*` — relative pose error per **metre of ground-truth travel**, not per
  sample. Using distance keeps the comparison fair when the platforms move at
  different speeds; a slower robot evaluated over shorter relative motions
  would otherwise look better.

### Stability → SLAM correlation

`stability_error_correlation()` reduces both signals to non-overlapping 1 s
windows — predictor: `std(pitch)`, `std(roll)` inside the window; response:
mean instantaneous ATE in the same window — then reports Pearson *r* with a
2000-sample percentile bootstrap CI.

Windowing is not cosmetic. The raw ATE is a slowly drifting integral, so a
sample-wise correlation against a zero-mean vibration signal is dominated by
autocorrelation and reports a meaningless number. The bootstrap is used instead
of an analytic p-value because windowed IMU statistics are not normal (and
because it needs no scipy).

### Tracking loss and loop closures

Recorded to `events.csv` with the timestamp, the ground-truth position and the
distance travelled, so a loss can be attributed to a place on the route:

- **tracking loss** — RTAB-Map publishes a saturated pose covariance
  (≥ 9999), or `OdomInfo.lost`, or no odometry for longer than
  `tracking_loss_timeout`.
- **relocalization** — the transition back out of that state.
- **loop closure** — classified against ground truth. The logger records the GT
  position at which each map node was created (from `Info.refId`); when a
  closure onto node *X* is reported, the GT gap between the current pose and
  node *X*'s pose decides it: within `loop_true_radius` (default 3 m) is a true
  positive, beyond it a false positive.

### Map vs true geometry (`map_vs_groundtruth.py`)

Samples the collision geometry of the world — including the mine's COLLADA mesh
— and compares the RTAB-Map cloud against it:

- **accuracy** map → surface: is the mapped geometry real?
- **completeness** surface → map: was it all mapped?
- **Chamfer** = mean(accuracy) + mean(completeness). Both halves are reported
  separately, because a paper quoting only the sum cannot distinguish "mapped
  the wrong shape" from "did not finish exploring".
- **precision / recall / F-score at τ** (default 0.20 m), plus map density.

Only surface within `--range` of the ground-truth trajectory is used: a robot
cannot be blamed for a gallery it never entered, and including unvisited
geometry would turn completeness into a measure of route coverage.

The nearest-neighbour search is grid-hashed and **exact** up to the cutoff
radius (verified against brute force to 0 disagreement) — no scipy or open3d
needed.

---

## 6. The rocker differential

A real rocker-bogie ties the two rockers together so the chassis pitch is the
*average* of the two, halving the tilt that reaches the sensor mast. That is the
mechanical claim behind the paper, so it has to be in the model.

The constraint `φ_left + φ_right = 0` is a closed loop; URDF describes trees
only, and `<joint type="gearbox">` is not reachable from a URDF-spawned model.
So it is imposed the way a physical differential imposes it — with torque:

```
e   = φ_l + φ_r
τ   = −k·e − c·ė        applied to BOTH pivots with the same sign
```

Same sign is correct: for the holonomic constraint `g = φ_l + φ_r`,
`∂g/∂φ_l = ∂g/∂φ_r = 1`, so the multiplier enters both joints identically and
the reaction `−2τ` lands on the chassis, which is where a differential is
mounted.

The launch file sets `k = 948.08 N·m/rad`, `c = 28.44 N·m·s/rad` — the native
`2000 / 60` scaled by the same 0.474039 as the masses (§1.5), which leaves the
coupling dynamics exactly as tuned. The rocker subtree is 25 links and 13.48 kg
with `I = 4.046 kg·m²` about the pivot (measured from the expanded URDF by the
parallel axis theorem, not estimated), giving ω_n ≈ 15.3 rad/s (2.4 Hz, ~410
physics steps per period) — comfortably stable at a 1 ms step and well under
the node's 200 Hz rate.

ζ ≈ 0.23, so the coupling is underdamped and a step disturbance overshoots by
roughly 48%. That comes from the original tuning and is deliberately left
alone so the mass matching changes one variable only. Critical damping would be
`c = 2√(kI) = 123.9`. Both parameters are ROS params and both belong in the
paper's model table. The node reports the RMS constraint residual on shutdown;
report it, and if it is dominated by ringing rather than by steady offset,
raise the damping rather than the stiffness.

**Ablation:** `roslaunch rocker_bogie lcmine_rocker_bogie_world.launch
differential_enabled:=false` leaves the pivots at zero effort, i.e. genuinely
free — the "independent rockers" control condition.

---

## 7. Wheel contact (replaces Fig. 2)

Fig. 2 as submitted came from an idealised MATLAB model: it shows what the
mechanism is *supposed* to do, not what the simulated robot did. Six Gazebo
contact sensors (one per wheel, 100 Hz) now publish to
`/wheel_contacts/<link>`, and the logger records contact state and normal force
alongside everything else.

```bash
python3 src/robot_metrics/scripts/plot_wheel_contact.py \
    --run ~/metrics_output/rocker_bogie/run01
```

produces a three-panel figure: a per-wheel contact raster, per-wheel normal
force, and chassis pitch/roll over the same interval, with the step window
located from ground truth (not from "where the acceleration was biggest", which
would beg the question). Wheels that are touching but carrying < 1 N are drawn
pale — the rocker-bogie's advantage is not just keeping six wheels down but
keeping *load* on them, and a raster alone hides that.

### On the contact-stiffness question

`kp = 1e5 N/m`, `kd = 10`, `minDepth = 3 mm`, `maxVel = 1 m/s`. At the
mass-matched 45.0 kg the static load is 73.6 N per wheel, i.e. **0.74 mm** of
penetration — soft, not stiff (stock Husky wheels use `kp = 1e6…1e7`). Before
the mass matching it was 155 N and 1.55 mm; the conclusion only gets stronger.
**The contact model was not the cause of the observed lift-off.** The causes were the locked suspension (§1.2) and the unconverged
solver (§1.1): with 6 wheel contacts and the passive pivots in one constraint
island, Gazebo's default 50 iterations do not converge inside a 1 ms step, and
the residual appears as contact jitter. The world now pins 150 iterations; peak
vertical acceleration changes < 2% between 150 and 300.

For a sensitivity study, sweep `kp` over 1e4 / 1e5 / 1e6 with everything else
fixed and compare the contact traces.

---

## 8. Route and waypoints

The route is defined **once**, in Gazebo world coordinates, in
`config/route_mine.yaml`. `generate_waypoints.py` projects it into each robot's
map frame using that robot's spawn pose:

```
p_map = R(−yaw_spawn) · (p_world − p_spawn)
```

This is what the old Husky file's comment — *"Transformados desde rocker_bogie
(x*−1, y*−1, yaw+180)"* — was doing by hand, once, and never re-checked. It now
falls out of the spawn pose automatically. **The per-robot `waypoints_mine.yaml`
files are generated; do not edit them.** `run_campaign.sh` refuses to start if
they are stale.

The corridors are not invented — they follow the four galleries the previous
Husky run demonstrably drove. What changed is the distribution: the old file had
1.0 m gaps next to 11.2 m gaps, five of fifteen points inside a 5 m stretch of
one corner, and nothing along the 19 m north gallery. Now spacing is 6–10 m
(mean 8.6 m) around a 137.7 m loop, every corridor carries intermediate points,
the step is straddled 4 m before and after (never landed on), and the route
closes so RTAB-Map gets a genuine revisit.

> **The ramp is terrain patching, not a test obstacle.** The `ramp` model at
> (25.5, −9.5) — a 5 m × 1.5 m wedge, 16.7° — and the other added blocks were
> placed to cover holes in the `lc_mine` mesh and verified visually in Gazebo.
> It sits inside the loop rather than on any of the four corridors, and the
> previous Husky route did not cross it either, so no robot drives over it.
> That is intended. It does mean, though, that **no slope- or ramp-climbing
> claim can be supported by these runs** — if the paper makes one, it has to go,
> or it needs its own route and its own validated corridor. Do not simply add a
> waypoint at (25.5, −9.5) and assume the robot can get there: the interior of
> the loop has not been shown to be drivable.

### Follower

`waypoint_navigator_yaml.py` now respects the platform's kinematics: the yaw
rate is capped at `v / R_min` for a steered platform, speed ramps down over
`approach_distance` to `approach_speed`, and in-place heading correction is
skipped for platforms that cannot pivot. The old hard floor of 0.3 m/s (which
kept the robot driving forward while up to 90° off course) is gone, and the
`min_angular = 0.45 rad/s` spin-turn floor now defaults to **off** — it silently
made the robot rotate faster than commanded, corrupting the yaw-rate and slip
metrics.

---

## 9. Script reference

| Script | Purpose |
|---|---|
| `check_sim_parity.py` | Assert the three platforms share world, physics, terrain, obstacle, spawn and sensor noise. **Run before every campaign.** |
| `run_campaign.sh` | Drive 3 robots × 3 runs with per-run seeds, then aggregate. |
| `metrics_logger.py` | Per-run logging: time series, raw GT, events, provenance. |
| `metrics_io.py` | Shared CSV schema with units; TUM read/write. |
| `slam_metrics.py` | ATE/RPE with trans/rot split, Umeyama alignment, correlation, bootstrap CI. |
| `aggregate_runs.py` | Per-run and mean ± std tables, ATE-vs-time/distance, correlation plots. |
| `analyze_metrics.py` | Single-run quick look (superseded by `aggregate_runs.py` for the paper). |
| `plot_wheel_contact.py` | The measured contact figure that replaces Fig. 2. |
| `extract_robot_specs.py` | Mass, CoG, bounding box, track, wheelbase, speeds, controller config. |
| `extract_sensor_poses.py` | Sensor poses, lever arms, cross-platform spread, `--standardize`. |
| `scale_masses.py` | Mass-matches the three platforms by uniform inertial scaling. |
| `generate_waypoints.py` | Project the shared route into each robot's map frame. |
| `map_vs_groundtruth.py` | Chamfer / completeness / F-score against the true world geometry. |
| `model_parser.py` | Normalises URDF, xacro and SDF into one structure. |
| `pointcloud_io.py` | PCD/PLY readers, STL/DAE sampling, exact grid-hashed NN. |

Dependencies: numpy, pandas, matplotlib, PyYAML. Deliberately **not** scipy,
sklearn or open3d — everything they would provide is implemented here so the
results reproduce without installing anything extra.

---

## 10. Before resubmitting

- [ ] `catkin_make` and re-run `check_sim_parity.py` → PASS
- [ ] **Regenerate all data.** Every result predating §1.1–§1.6 is invalid —
      the mass matching alone changes every dynamic quantity in the paper.
- [ ] Report the differential stiffness/damping and the constraint residual
- [ ] Report solver iterations and step size (`sim_config.yaml`)
- [ ] State that the platforms are mass-matched at 45.0 kg, give the native
      masses (94.9 / 47.0 / 45.3 kg) and the uniform-scaling argument (§1.5)
- [ ] State that sensor placement is standardized (spread < 1 mm), that it had
      to go upwards, and that the resulting CoG rise disfavours the low
      platforms rather than the Rocker-Bogie (§1.5)
- [ ] Report the rescaled PID gains and differential stiffness, and say why
- [ ] Report LiDAR noise as **0.011314 m**, not 0.008 (§1.6), and state that
      every run is seeded (`gazebo_seed = seed_base + run_id`) so the results
      are reproducible
- [ ] Replace Fig. 2 with the measured contact figure, or present both
- [ ] Report ATE/RPE as mean ± std over three runs, trans and rot separately
- [ ] Report tracking losses, relocalizations, and true/false loop closures
- [ ] Remove any slope- or ramp-climbing claim: no robot drives over the ramp,
      which is terrain patching rather than a test obstacle (§8)
- [ ] Measure the residual non-determinism: run the same seed twice and diff
      the two `metrics.csv`. The seed fixes the sensor noise but not ROS
      message timing, so this is the floor on run-to-run variation — worth
      knowing before a reviewer asks
