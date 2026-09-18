#!/usr/bin/env bash
#
# run_yaw_calibration.sh - banco de giro para las tres plataformas.
#
# Levanta el mundo con cada robot, corre yaw_calibration.py y recoge el CSV.
# NO lanza SLAM ni el seguidor: el banco no los necesita y son lo caro.
#
# Uso:
#   ./run_yaw_calibration.sh                    las tres
#   ./run_yaw_calibration.sh --robots tracked   una
#   ./run_yaw_calibration.sh --gui              con ventana de Gazebo
#
set -u -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PKG_DIR="$( dirname "$SCRIPT_DIR" )"
SRC_DIR="$( dirname "$PKG_DIR" )"
WS_DIR="$( dirname "$SRC_DIR" )"

ROBOTS="differential tracked rocker_bogie"
OUTPUT_DIR="${HOME}/metrics_output/yaw_calibration"
GUI=false
SETTLE=25
# Tope por robot.  36 escalones x (6 s de escalon + 2 s de reposo + 2.5 s de
# asentado tras reponer la pose) = ~380 s de simulacion.  Con el factor de
# tiempo real medido (0.32-0.49) son 13-20 min de reloj.
DURATION=1800
HOLD=6.0
YAW_RATES="0.1,0.2,0.3,0.5,0.7,1.0"
LIN_SPEEDS="0.0,0.25,0.5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robots)   ROBOTS="$2"; shift 2 ;;
    --output)   OUTPUT_DIR="$2"; shift 2 ;;
    --gui)      GUI=true; shift ;;
    --duration) DURATION="$2"; shift 2 ;;
    --hold)     HOLD="$2"; shift 2 ;;
    --yaw-rates) YAW_RATES="$2"; shift 2 ;;
    --lin-speeds) LIN_SPEEDS="$2"; shift 2 ;;
    -h|--help)  sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "opcion desconocida: $1"; exit 2 ;;
  esac
done

world_launch() {
  case "$1" in
    differential) echo "differential lcmine_husky_world.launch" ;;
    tracked)      echo "gazebo_continuous_track_example lcmine_two_track_world.launch" ;;
    rocker_bogie) echo "rocker_bogie lcmine_rocker_bogie_world.launch" ;;
  esac
}
gt_model() {
  case "$1" in
    differential) echo "differential" ;;
    tracked)      echo "two_track_robot" ;;
    rocker_bogie) echo "rocker_bogie" ;;
  esac
}
cmd_topic() {
  case "$1" in
    differential) echo "/husky/cmd_vel" ;;
    tracked)      echo "/example_two_track/cmd_vel" ;;
    rocker_bogie) echo "/rocker_bogie/cmd_vel" ;;
  esac
}

log() { echo -e "\033[1;34m[calib]\033[0m $*"; }
err() { echo -e "\033[1;31m[calib]\033[0m $*" >&2; }

if [[ -f "${WS_DIR}/devel/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${WS_DIR}/devel/setup.bash"
else
  err "no hay devel/setup.bash en ${WS_DIR}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
log "salida : ${OUTPUT_DIR}"
log "robots : ${ROBOTS}"
log "rejilla: w = ${YAW_RATES} rad/s   v = ${LIN_SPEEDS} m/s"

limpia() {
  pkill -x gzserver  >/dev/null 2>&1
  pkill -x gzclient  >/dev/null 2>&1
  pkill -x roslaunch >/dev/null 2>&1
  pkill -x rosmaster >/dev/null 2>&1
  sleep 5
}
trap 'echo; err "interrumpido"; limpia; exit 130' INT TERM

for robot in ${ROBOTS}; do
  read -r wpkg wlaunch <<< "$(world_launch "$robot")"
  out="${OUTPUT_DIR}/${robot}.csv"
  logdir="${OUTPUT_DIR}/${robot}"
  mkdir -p "${logdir}"
  log "=== ${robot} -> ${out}"

  limpia
  roslaunch "${wpkg}" "${wlaunch}" gui:="${GUI}" paused:=false \
      seed:=1 lidar:=normal > "${logdir}/gazebo.log" 2>&1 &
  sleep "${SETTLE}"

  if ! rostopic list >/dev/null 2>&1; then
    err "  el master no arranco; ver ${logdir}/gazebo.log"
    limpia
    continue
  fi

  timeout --signal=INT --kill-after=30 "${DURATION}" \
    rosrun robot_metrics yaw_calibration.py \
      _model_name:="$(gt_model "$robot")" \
      _cmd_vel_topic:="$(cmd_topic "$robot")" \
      _output_csv:="${out}" \
      _hold_s:="${HOLD}" \
      _yaw_rates:="${YAW_RATES}" \
      _lin_speeds:="${LIN_SPEEDS}" \
      > "${logdir}/calibration.log" 2>&1
  rc=$?
  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    err "  ${robot}: se agoto el tope de ${DURATION}s"
  fi
  if [[ -f "${out}" ]]; then
    log "  ${robot}: $(wc -l < "${out}") filas"
  else
    err "  ${robot}: no se escribio el CSV; ver ${logdir}/calibration.log"
  fi
  limpia
done

log "banco terminado. Analiza con:"
log "  python3 ${SCRIPT_DIR}/analiza_calibracion.py --dir ${OUTPUT_DIR}"
