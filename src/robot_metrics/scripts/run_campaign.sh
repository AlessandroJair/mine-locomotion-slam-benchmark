#!/usr/bin/env bash
#
# run_campaign.sh - drive the full 3 robots x 3 runs experiment.
#
# The repeatability requirement is three runs per platform, each logged
# separately.  Doing that by hand invites exactly the drift this revision is
# fixing: a parameter changed between runs, one run overwriting another, or a
# robot quietly driving a different route.  So the campaign is a script.
#
# Before anything runs it calls check_sim_parity.py, which refuses to continue
# unless the three platforms share the same world, physics and terrain, and
# generate_waypoints.py --check, which refuses if a waypoint file has been
# hand-edited away from the shared route.
#
# Usage:
#   ./run_campaign.sh                          all three robots, runs 1-3
#   ./run_campaign.sh --robots rocker_bogie    one platform
#   ./run_campaign.sh --runs 5                 five repetitions
#   ./run_campaign.sh --duration 600           cap each run at 600 s
#   ./run_campaign.sh --seed-base 100          an independent replication
#   ./run_campaign.sh --dry-run                print what would happen
#
# Every run is launched with gzserver --seed (seed_base + run_id), so the
# sensor noise of any run can be reproduced exactly.  Runs still differ from
# one another, which is what the reported standard deviation measures.  The
# seed is recorded in each run_meta.yaml and in campaign_meta.yaml.
#
set -u -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PKG_DIR="$( dirname "$SCRIPT_DIR" )"
SRC_DIR="$( dirname "$PKG_DIR" )"
WS_DIR="$( dirname "$SRC_DIR" )"

ROBOTS="differential tracked rocker_bogie"
RUNS=3
DURATION=1200          # hard cap per run, seconds
SETTLE=25              # seconds for Gazebo, controllers and SLAM to come up
OUTPUT_DIR="${HOME}/metrics_output"
DRY_RUN=0
GUI=false
# Gazebo's RNG seed for run N is SEED_BASE + N.  Change it only to collect an
# independent replication of the whole campaign; keep it at 0 to reproduce the
# results in the paper.
SEED_BASE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robots)   ROBOTS="$2"; shift 2 ;;
    --runs)     RUNS="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --output)   OUTPUT_DIR="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --gui)      GUI=true; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)  sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

# Per-robot launch configuration.
world_launch() {
  case "$1" in
    differential) echo "differential lcmine_husky_world.launch" ;;
    tracked)      echo "gazebo_continuous_track_example lcmine_two_track_world.launch" ;;
    rocker_bogie) echo "rocker_bogie lcmine_rocker_bogie_world.launch" ;;
  esac
}
nav_pkg() {
  case "$1" in
    differential) echo "differential" ;;
    tracked)      echo "gazebo_continuous_track_example" ;;
    rocker_bogie) echo "rocker_bogie" ;;
  esac
}

log()  { echo -e "\033[1;34m[campaign]\033[0m $*"; }
warn() { echo -e "\033[1;33m[campaign]\033[0m $*"; }
err()  { echo -e "\033[1;31m[campaign]\033[0m $*" >&2; }

# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------
if [[ -f "${WS_DIR}/devel/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${WS_DIR}/devel/setup.bash"
else
  err "no devel/setup.bash in ${WS_DIR}; build the workspace first (catkin_make)"
  exit 1
fi

log "workspace : ${WS_DIR}"
log "output    : ${OUTPUT_DIR}"
log "robots    : ${ROBOTS}"
log "runs each : ${RUNS}"
echo

log "Checking simulation parity..."
if ! python3 "${SCRIPT_DIR}/check_sim_parity.py" --src "${SRC_DIR}"; then
  err "parity check FAILED - the platforms are not running identical"
  err "conditions. Fix that before collecting data for the paper."
  exit 1
fi
echo

log "Checking the waypoint files match the shared route..."
if ! python3 "${SCRIPT_DIR}/generate_waypoints.py" --src "${SRC_DIR}" --check; then
  err "waypoint files are stale. Run generate_waypoints.py, or if a file was"
  err "hand-edited, move the change into config/route_mine.yaml."
  exit 1
fi
echo

# Record the configuration this campaign ran under, so results can be traced
# back to it even after the tree moves on.
mkdir -p "${OUTPUT_DIR}"
{
  echo "campaign_started: $(date -Is)"
  echo "git_commit: $(git -C "${SRC_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "robots: ${ROBOTS}"
  echo "runs_per_robot: ${RUNS}"
  echo "duration_cap_s: ${DURATION}"
  # Reproducibility: run N of every robot ran with gzserver --seed $((SEED_BASE+N)).
  echo "seed_base: ${SEED_BASE}"
  echo "seed_rule: 'gazebo_seed = seed_base + run_id'"
} > "${OUTPUT_DIR}/campaign_meta.yaml"
cp "${PKG_DIR}/config/sim_config.yaml" "${OUTPUT_DIR}/" 2>/dev/null || true
cp "${PKG_DIR}/config/route_mine.yaml" "${OUTPUT_DIR}/" 2>/dev/null || true

# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------
cleanup() {
  # Gazebo leaves gzserver behind often enough that not killing it explicitly
  # means the next run silently attaches to the previous world.
  pkill -f gzserver    >/dev/null 2>&1
  pkill -f gzclient     >/dev/null 2>&1
  pkill -f rtabmap      >/dev/null 2>&1
  pkill -f roslaunch    >/dev/null 2>&1
  pkill -f rosmaster    >/dev/null 2>&1
  sleep 5
}
trap 'echo; warn "interrupted"; cleanup; exit 130' INT TERM

run_one() {
  local robot="$1" run_id="$2"
  read -r wpkg wlaunch <<< "$(world_launch "$robot")"
  local npkg
  npkg="$(nav_pkg "$robot")"
  local run_dir="${OUTPUT_DIR}/${robot}/run$(printf '%02d' "$run_id")"

  log "=== ${robot} run ${run_id}/${RUNS} -> ${run_dir}"

  # The seed makes the run reproducible.  It is the repetition index, so runs
  # 1/2/3 still see different sensor noise - which is what the reported spread
  # measures - but each one can be replayed exactly.  Re-run a single one with
  #   roslaunch <pkg> <world.launch> seed:=2
  local seed=$(( SEED_BASE + run_id ))

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    roslaunch ${wpkg} ${wlaunch} gui:=${GUI} seed:=${seed}"
    echo "    roslaunch ${npkg} rtabmap_3d_slam.launch"
    echo "    roslaunch ${npkg} waypoint_navigation.launch run_id:=${run_id}"
    return 0
  fi

  cleanup
  mkdir -p "${run_dir}"

  # Written before the run so the seed survives even if the run dies.
  {
    echo "robot: ${robot}"
    echo "run_id: ${run_id}"
    echo "gazebo_seed: ${seed}"
    echo "started: $(date -Is)"
  } > "${run_dir}/run_meta.yaml"

  roslaunch "${wpkg}" "${wlaunch}" gui:="${GUI}" paused:=false \
      seed:="${seed}" \
      > "${run_dir}/gazebo.log" 2>&1 &
  local world_pid=$!
  sleep "${SETTLE}"

  if ! rostopic list >/dev/null 2>&1; then
    err "  ROS master never came up; see ${run_dir}/gazebo.log"
    cleanup
    return 1
  fi

  roslaunch "${npkg}" rtabmap_3d_slam.launch \
      > "${run_dir}/slam.log" 2>&1 &
  sleep 10

  # The logger is started by waypoint_navigation.launch, which passes run_id
  # through so each repetition lands in its own directory.
  timeout "${DURATION}" roslaunch "${npkg}" waypoint_navigation.launch \
      run_id:="${run_id}" use_rviz:=false \
      > "${run_dir}/navigation.log" 2>&1
  local rc=$?
  if [[ $rc -eq 124 ]]; then
    warn "  hit the ${DURATION}s cap; the run is kept but flagged"
    echo "timed_out: true" >> "${run_dir}/run_meta.yaml"
  fi

  # roslaunch has to exit cleanly for the logger's shutdown hook to write the
  # CSV, so give it a moment before tearing the rest down.
  sleep 8
  cleanup

  if [[ -f "${run_dir}/metrics.csv" ]]; then
    local n
    n=$(( $(wc -l < "${run_dir}/metrics.csv") - 1 ))
    log "  OK: ${n} samples logged"
    return 0
  fi
  err "  no metrics.csv was written; see ${run_dir}/navigation.log"
  return 1
}

# --------------------------------------------------------------------------
FAILED=""
for robot in ${ROBOTS}; do
  for run_id in $(seq 1 "${RUNS}"); do
    if ! run_one "${robot}" "${run_id}"; then
      FAILED="${FAILED} ${robot}/run${run_id}"
    fi
  done
done

echo
if [[ -n "${FAILED}" ]]; then
  warn "runs that produced no data:${FAILED}"
  warn "the aggregate will use whatever did succeed, but a platform with"
  warn "fewer than 3 runs has no meaningful standard deviation."
fi

if [[ $DRY_RUN -eq 1 ]]; then
  log "dry run complete"
  exit 0
fi

log "Aggregating..."
python3 "${SCRIPT_DIR}/aggregate_runs.py" \
    --results "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}/paper_tables"

log "Extracting model specifications..."
python3 "${SCRIPT_DIR}/extract_robot_specs.py" --src "${SRC_DIR}" \
    --output_dir "${OUTPUT_DIR}/paper_tables"
python3 "${SCRIPT_DIR}/extract_sensor_poses.py" --src "${SRC_DIR}" \
    --output_dir "${OUTPUT_DIR}/paper_tables" --standardize

log "Wheel-contact figure (Rocker-Bogie run 1)..."
if [[ -f "${OUTPUT_DIR}/rocker_bogie/run01/metrics.csv" ]]; then
  python3 "${SCRIPT_DIR}/plot_wheel_contact.py" \
      --run "${OUTPUT_DIR}/rocker_bogie/run01" \
      --output_dir "${OUTPUT_DIR}/paper_tables" || true
fi

echo
log "Campaign complete. Tables and figures: ${OUTPUT_DIR}/paper_tables"
log "Still to do by hand: export each RTAB-Map database and score the maps,"
log "  rosrun rtabmap_ros rtabmap-export --cloud map.pcd ~/.ros/rtabmap.db"
log "  python3 ${SCRIPT_DIR}/map_vs_groundtruth.py --map map.pcd \\"
log "      --run ${OUTPUT_DIR}/<robot>/run01 --output_dir ${OUTPUT_DIR}/paper_tables"
