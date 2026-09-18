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
#   ./run_campaign.sh --skip-parity            DIAGNOSTIC ONLY, see below
#   ./run_campaign.sh --max-retries 2          repetir una corrida hasta 2
#                                              veces si el vigia la aborta
#   ./run_campaign.sh --lidar swept            use the swept LiDAR instead of
#                                              the instantaneous one.  Writes
#                                              to <output>/swept_lidar/.
#                                              See LIDAR below.
#   ./run_campaign.sh --fixed-trajectory       drive one fixed path under
#                                              ground-truth control instead of
#                                              navigating; no move_base, the
#                                              SLAM only observes.  Writes to
#                                              <output>/fixed_trajectory/.
#                                              A SECOND experiment, not a
#                                              replacement - see FIXED_TRAJ.
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
# --skip-parity runs the parity check, prints it, and continues even if it
# FAILS.  It exists so a single platform can be driven for debugging while an
# unrelated platform's parity is broken - watching the Rocker-Bogie climb does
# not require the Husky's masses to be right.  Every run it produces is
# stamped paper_usable: false in run_meta.yaml and campaign_meta.yaml, because
# a run collected under a failing parity check is a debugging artefact and
# nothing else.  Never use it to produce a number that reaches the paper.
SKIP_PARITY=0
# Gazebo's RNG seed for run N is SEED_BASE + N.  Change it only to collect an
# independent replication of the whole campaign; keep it at 0 to reproduce the
# results in the paper.
SEED_BASE=0
# --fixed-trajectory drives every platform along ONE path, taken from the
# rocker_bogie run that closed the loop, under ground-truth control.  move_base
# is not started; the SLAM runs alongside and is recorded but cannot steer.
#
# It exists because with move_base each platform steers on its own SLAM
# estimate, that estimate drifts 0.6-0.8 m over this route, and the drift moves
# the ROBOT, not just the log: measured 2026-08-21, the rocker_bogie crossed the
# 0.10 m step at x = 54.35 while the differential wedged at x = 53.41, nose-up
# 48 deg, on ground the rocker never touched.  Two platforms meeting different
# terrain are not being compared.
#
# This is a SECOND experiment, not a replacement.  Ground truth is not available
# to a real robot, so a fixed-trajectory run cannot say whether a platform can
# navigate the mine - that is what the move_base runs are for.  What it gives is
# a SLAM comparison and a locomotion comparison over identical geometry.
# Results go to <output>/fixed_trajectory/ so the two never mix.
FIXED_TRAJ=0
# LIDAR: normal | swept
#
# normal es el sensor tal cual: un gpu_ray que dispara sus 1875x16 rayos TODOS
# en el mismo instante.  Por construccion su nube no puede llevar distorsion de
# movimiento, porque no hay movimiento entre el primer rayo y el ultimo.  Un
# VLP-16 real barre durante 100 ms y toma cada azimut desde una pose distinta.
#
# swept toma esa misma nube y le mete la distorsion de barrido por formulas,
# en sweep_distortion.py:
#
#     p_medido = T_i^-1 * T_inicio * p_instantaneo
#
# con el instante de cada punto deducido de su azimut y las poses interpoladas
# a velocidad constante.  Es la correccion de LIO-SAM invertida; ver la
# cabecera del nodo para las referencias.
#
# EL SENSOR NO CAMBIA.  retopic_lidar.py solo le aparta el topic a
# /velodyne_points_raw; el modelo que se spawnea es BIT A BIT el mismo que en
# normal salvo esa cadena.  Resolucion angular, 16 anillos, rango y ruido
# sembrado son literalmente los mismos objetos.
#
# Hasta el 2026-09-09 hubo un tercer modo que montaba un rotor con una cuña de
# 21 grados y ensamblaba las rebanadas.  Se retiro: dependia de que Gazebo
# cumpliera el update_rate del gpu_ray y no lo cumple, asi que la cobertura de
# azimut salia de 350 a 356 grados en vez de 360 y bajaba con la carga de
# simulacion.  Eso hacia que la distorsion fuese una propiedad del planificador,
# distinta por plataforma.  Queda archivado en ~/arm_sliced_retirado_2026-09-09/.
#
# Los resultados van a <output>/swept_lidar/ por la misma razon que los de
# --fixed-trajectory van a su propio directorio: son dos sensores distintos y
# promediarlos juntos no significa nada.
LIDAR=normal
# Vueltas a la ruta en modo --fixed-trajectory.  El recorrido es un
# lazo cerrado: repetirlo acumula deriva sobre la misma geometria y
# multiplica las oportunidades de cierre de loop, sin cambiar la ruta
# que comparten las tres plataformas.  Subilo junto con --duration.
LAPS=1
# REINTENTOS.  run_guard.py (lanzado con required="true") aborta la corrida
# cuando la plataforma queda encajada o se vuelca, y deja guard_trip.yaml en
# el directorio de la corrida.  Una corrida asi no es un dato: es una
# corrida que no ocurrio, y promediarla con las buenas mete un cero de
# recorrido en la desviacion estandar.  Se repite hasta MAX_RETRIES veces
# mas; el intento fallido se guarda como run<NN>_failed_attempt<N> en vez
# de borrarse, porque es justo lo que hay que mirar para entender por que.
# 0 desactiva el reintento (el vigia sigue abortando y avisando).
MAX_RETRIES=2
# Intento actual, para que quede escrito en run_meta.yaml.
ATTEMPT=1

# GANANCIAS DEL SEGUIDOR QUE SE PUEDEN FORZAR DESDE LA LINEA DE ORDENES.
# Vacias por defecto: sin ellas el launch usa las suyas y la campana no
# cambia en nada.  Existen para el A/B de velocidad.  El husky va a 0.417
# m/s y el rocker y el tracked a 0.456-0.459, y esa diferencia NO se manda:
# la produce el propio seguimiento, porque k_xte*xte entra en la curvatura y
# curve_slowdown la convierte en freno.  Para medir si la velocidad explica
# la deriva de rumbo del LiDAR troceado hay que romper esa realimentacion:
#   --max-vel-x 0.42   frena al rocker hasta la velocidad del husky
#   --k-xte 0.0        suelta al husky hasta la de los otros dos
# CUALQUIERA DE LAS DOS ROMPE LA PARIDAD del estudio a proposito: la corrida
# queda marcada paper_usable: false y no se puede mezclar con la campana.
FOLLOW_MAX_VEL_X=""
FOLLOW_K_XTE=""
# TECHO DE RTF forzado.  Vacio = el que declara el mundo (0.0005 * 1000 = 0.50).
# NO es un ajuste de rendimiento: con use_sim_time, bajar el RTF le da al SLAM
# mas tiempo de PARED por segundo de simulacion, asi que procesa MAS nubes de
# las 10/s que produce el sensor.  Medido el 2026-09-03: husky a RTF 0.482
# procesa el 51 % de sus nubes, tracked a 0.246 el 96 %.  Es una variable
# experimental, y por eso marca la corrida paper_usable: false.
# Se aplica por servicio, en caliente, para no tocar lcmine.world: los tres
# mundos tienen que seguir siendo byte a byte identicos para check_sim_parity.
TARGET_RTF=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robots)   ROBOTS="$2"; shift 2 ;;
    --runs)     RUNS="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --output)   OUTPUT_DIR="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --gui)      GUI=true; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --skip-parity) SKIP_PARITY=1; shift ;;
    --fixed-trajectory) FIXED_TRAJ=1; shift ;;
    --lidar)    LIDAR="$2"; shift 2 ;;
    --laps)     LAPS="$2"; shift 2 ;;
    --max-vel-x) FOLLOW_MAX_VEL_X="$2"; shift 2 ;;
    --k-xte)     FOLLOW_K_XTE="$2"; shift 2 ;;
    --rtf)       TARGET_RTF="$2"; shift 2 ;;
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    -h|--help)  sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

# Las ganancias forzadas se montan una sola vez y se reenvian tal cual.  Un
# array, no una cadena: con set -u y comillas, un array vacio se expande a
# nada y la linea de roslaunch queda exactamente como estaba.
FOLLOW_OVERRIDES=()
FOLLOW_NOTE="none"
if [[ -n "${FOLLOW_MAX_VEL_X}" ]]; then
  FOLLOW_OVERRIDES+=("max_vel_x:=${FOLLOW_MAX_VEL_X}")
  FOLLOW_NOTE="max_vel_x=${FOLLOW_MAX_VEL_X}"
fi
if [[ -n "${FOLLOW_K_XTE}" ]]; then
  FOLLOW_OVERRIDES+=("k_xte:=${FOLLOW_K_XTE}")
  if [[ "${FOLLOW_NOTE}" == "none" ]]; then FOLLOW_NOTE=""; else FOLLOW_NOTE="${FOLLOW_NOTE} "; fi
  FOLLOW_NOTE="${FOLLOW_NOTE}k_xte=${FOLLOW_K_XTE}"
fi
if [[ -n "${TARGET_RTF}" ]]; then
  if [[ "${FOLLOW_NOTE}" == "none" ]]; then FOLLOW_NOTE=""; else FOLLOW_NOTE="${FOLLOW_NOTE} "; fi
  FOLLOW_NOTE="${FOLLOW_NOTE}rtf=${TARGET_RTF}"
fi
if [[ ${#FOLLOW_OVERRIDES[@]} -gt 0 && $FIXED_TRAJ -eq 0 ]]; then
  echo "--max-vel-x y --k-xte solo actuan con --fixed-trajectory" >&2
  exit 2
fi

# The two experiments must never share a directory.  They produce the same file
# names from the same logger, so a fixed-trajectory run landing on top of an
# autonomous one would be indistinguishable afterwards - and aggregate_runs.py
# would average them together without a word.
if [[ "$LIDAR" != "normal" && "$LIDAR" != "swept" ]]; then
  echo "--lidar solo acepta 'normal' o 'swept', no '${LIDAR}'" >&2
  exit 2
fi

if [[ $FIXED_TRAJ -eq 1 ]]; then
  OUTPUT_DIR="${OUTPUT_DIR}/fixed_trajectory"
fi
# Despues de fixed_trajectory, no antes: las cuatro combinaciones existen y
# cada una tiene que caer en su propio sitio.
if [[ "$LIDAR" == "swept" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR}/swept_lidar"
fi

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
# The three things waypoint_navigation.launch hardcodes per platform, needed
# again by trajectory_follow.launch.  Kept here rather than duplicated in a
# second set of per-platform launch files: one copy, one place to get wrong.
# gt_model must match what Gazebo spawns - check_sim_parity.py section 12
# enforces that, after a run was lost to the logger looking up "husky" while
# Gazebo had spawned "differential".
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

# What each platform is called in a figure legend.  analyze_metrics.py derives
# these from the FILENAME when it is not told otherwise, and every file it is
# given here is called metrics.csv - so without this the three series would all
# come out with the same label.
display_name() {
  case "$1" in
    differential) echo "Husky" ;;
    tracked)      echo "Tracked" ;;
    rocker_bogie) echo "Rocker-bogie" ;;
    *)            echo "$1" ;;
  esac
}
wheel_links() {
  case "$1" in
    rocker_bogie) echo "Rueda_6_1,Rueda_4_1,Rueda_2_1,Rueda_1_1,Rueda_3_1,Rueda_5_1" ;;
    *)            echo "" ;;
  esac
}
# Articulaciones PASIVAS de la suspension: los dos rockers y los dos bogies.
# Sin ellas no se puede separar "la suspension articula y el terreno le gana"
# de "la suspension esta contra sus topes de +-0.6 rad", que es la pregunta que
# quedo abierta el 2026-09-10 con el 15.2 % de tiempo a menos de seis ruedas.
suspension_joints() {
  case "$1" in
    rocker_bogie) echo "rocker_pivot_left,rocker_pivot_right,rev_3,rev_4" ;;
    *)            echo "" ;;
  esac
}
# Par de las ruedas motrices.  Es la reaccion de estos motores la que enrolla
# el brazo del balancin: dos ruedas por bogie, hasta 20 N*m cada una, contra
# los 15-24 N*m de momento que dan las cargas.
effort_joints() {
  case "$1" in
    rocker_bogie) echo "rev_9,rev_10,rev_11,rev_12,rev_13,rev_14" ;;
    # Husky, 2026-09-14: para separar en curva lazo de velocidad blando
    # (rueda por debajo de consigna con par < 29.958), techo de par, o
    # contacto (rueda a consigna y el chasis no gira).
    differential) echo "front_left_joint,front_right_joint,back_left_joint,back_right_joint" ;;
    *)            echo "" ;;
  esac
}
# Velocidad de esas mismas ruedas, del mismo velocity[].  Va SIEMPRE con
# effort_joints: con el par solo, el reparto desigual de traccion tiene tres
# explicaciones indistinguibles.  Ver velocity_columns() en metrics_io.py.
velocity_joints() { effort_joints "$1"; }
# Donde publica cada joint_state_controller.  El logger escucha ahi par,
# velocidad y los pivotes pasivos; su default es el del rocker.
joint_states_topic() {
  case "$1" in
    differential) echo "/husky/joint_states" ;;
    *)            echo "/rocker_bogie/joint_states" ;;
  esac
}
# Only for drawing: the reference path is in world coordinates and the map frame
# is anchored at the spawn, so the follower needs the spawn yaw to put the path
# where the robot is.  It does NOT affect control, which is done in world
# coordinates.  Values from sim_config.yaml spawn:.
spawn_yaw() {
  case "$1" in
    rocker_bogie) echo "0.0" ;;
    *)            echo "0.0" ;;
  esac
}

# Cuanto sumarle al yaw que /gazebo/model_states reporta para obtener el
# rumbo FISICO.  El URDF del Rocker-Bogie tiene base_link mirando a -x, asi
# que su modelo reporta pi cuando el robot apunta a 0.  Coincide en valor
# con spawn_yaw porque el spawn se eligio justamente para cancelarlo, pero
# son cosas distintas.
#
# COPIA.  El original es base_yaw_offset_rad en sim_config.yaml, de donde lo
# lee generate_waypoints.py; aqui se repite porque este script no sabe leer
# YAML desde bash, igual que ya repite spawn x/y.  Si cambia uno, cambia el
# otro: que estas dos discrepen es exactamente el fallo que se arreglo el
# 2026-09-10, cuando world_to_viz() y to_map_frame() proyectaban al frame
# `map` con spawn_yaw a secas y dibujaban la ruta y el ground truth del
# rocker 180 deg fuera de sitio.
base_yaw_offset() {
  case "$1" in
    rocker_bogie) echo "0.0" ;;
    *)            echo "0.0" ;;
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
PARITY_OK=1
if ! python3 "${SCRIPT_DIR}/check_sim_parity.py" --src "${SRC_DIR}"; then
  PARITY_OK=0
  if [[ $SKIP_PARITY -eq 0 ]]; then
    err "parity check FAILED - the platforms are not running identical"
    err "conditions. Fix that before collecting data for the paper."
    err "To drive one platform anyway for debugging: --skip-parity"
    exit 1
  fi
  warn "================================================================"
  warn "PARITY CHECK FAILED AND WAS BYPASSED WITH --skip-parity."
  warn "This campaign is a DEBUGGING artefact. Its numbers are NOT"
  warn "comparable between platforms and MUST NOT reach the paper."
  warn "Every run_meta.yaml it writes says paper_usable: false."
  warn "================================================================"
  sleep 3
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
  # Which experiment produced these numbers.  Without this the two are
  # indistinguishable in the output, and they answer different questions.
  echo "control_mode: $([[ $FIXED_TRAJ -eq 1 ]] \
        && echo 'fixed_trajectory (ground-truth pursuit, SLAM observes only)' \
        || echo 'autonomous (move_base navigating on the SLAM estimate)')"
  echo "git_commit: $(git -C "${SRC_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "robots: ${ROBOTS}"
  echo "runs_per_robot: ${RUNS}"
  echo "duration_cap_s: ${DURATION}"
  # Reproducibility: run N of every robot ran with gzserver --seed $((SEED_BASE+N)).
  echo "seed_base: ${SEED_BASE}"
  echo "seed_rule: 'gazebo_seed = seed_base + run_id'"
  echo "follower_overrides: ${FOLLOW_NOTE}"
  # Que sensor produjo estas nubes.  Sin esto, dos campañas con LiDAR distinto
  # son indistinguibles en la salida y responden a preguntas distintas.
  echo "lidar: $([[ "$LIDAR" == "swept" ]] \
        && echo 'swept (distorsion de barrido metida por formulas sobre la nube instantanea)' \
        || echo 'normal (gpu_ray instantaneo: sin distorsion intra-barrido)')"
  echo "parity_check_passed: $([[ $PARITY_OK -eq 1 ]] && echo true || echo false)"
  # Una corrida con ganancias forzadas no es de la campana: no se promedia.
  echo "paper_usable: $([[ $PARITY_OK -eq 1 && ${#FOLLOW_OVERRIDES[@]} -eq 0 && -z "${TARGET_RTF}" ]] && echo true || echo false)"
} > "${OUTPUT_DIR}/campaign_meta.yaml"
cp "${PKG_DIR}/config/sim_config.yaml" "${OUTPUT_DIR}/" 2>/dev/null || true
cp "${PKG_DIR}/config/route_mine.yaml" "${OUTPUT_DIR}/" 2>/dev/null || true

# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------
cleanup() {
  # Order matters, and so does the signal.
  #
  # rtabmap goes first, and with SIGINT rather than the default SIGTERM, so it
  # gets to close its database instead of being cut off mid-write.  The file
  # is around 2 GB after two laps, so closing it is not instant - hence the
  # wait rather than a fixed sleep.  The README's map-scoring step runs
  # rtabmap-export over this database after the campaign, so it is worth
  # leaving in a consistent state.
  #
  # NOTE: an empty Word table in the saved database is NOT a symptom of this.
  # Checked 2026-08-23: Mem/RawDescriptorsKept is true, so each feature
  # carries its own descriptor in the Feature table and Word is simply not
  # used.  rtabmap's own statistics show the dictionary healthy in memory
  # (89472 words by the end of a two-lap run).
  pkill -INT -f rtabmap >/dev/null 2>&1
  local waited=0
  while pgrep -f rtabmap >/dev/null 2>&1 && [[ ${waited} -lt 60 ]]; do
    sleep 2
    waited=$(( waited + 2 ))
  done
  if pgrep -f rtabmap >/dev/null 2>&1; then
    warn "  rtabmap did not close its database in ${waited}s; forcing it"
    pkill -f rtabmap >/dev/null 2>&1
  fi

  # Gazebo leaves gzserver behind often enough that not killing it explicitly
  # means the next run silently attaches to the previous world.
  pkill -f gzserver    >/dev/null 2>&1
  pkill -f gzclient     >/dev/null 2>&1
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

  # La salida del robot se VACIA al empezar su primera corrida, intentos
  # fallidos incluidos.  Sin esto, cada campaña sobre el mismo directorio
  # apilaba los run<NN>_failed_attempt<N> de las anteriores: al mirarlos
  # despues no habia forma de saber de que campaña era cada uno -misma ruta?
  # mismas ganancias?- y el mv de archivado acababa anidandolos.  Los datos
  # de una campaña anterior que interesen hay que moverlos ANTES de relanzar.
  if [[ ${run_id} -eq 1 && ${ATTEMPT} -eq 1 ]]; then
    local viejos
    viejos=$(ls -d "${OUTPUT_DIR}/${robot}"/* 2>/dev/null | wc -l)
    if [[ ${viejos} -gt 0 ]]; then
      warn "  vaciando ${OUTPUT_DIR}/${robot} (${viejos} directorio(s) de una campaña anterior)"
      rm -rf "${OUTPUT_DIR:?}/${robot:?}"
    fi
  fi

  log "=== ${robot} run ${run_id}/${RUNS} -> ${run_dir}"

  # The seed makes the run reproducible.  It is the repetition index, so runs
  # 1/2/3 still see different sensor noise - which is what the reported spread
  # measures - but each one can be replayed exactly.  Re-run a single one with
  #   roslaunch <pkg> <world.launch> seed:=2
  local seed=$(( SEED_BASE + run_id ))

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    roslaunch ${wpkg} ${wlaunch} gui:=${GUI} seed:=${seed} lidar:=${LIDAR}"
    [[ -n "${TARGET_RTF}" ]] && echo "    python3 ${PKG_DIR}/scripts/set_rtf.py ${TARGET_RTF}"
    echo "    roslaunch ${npkg} rtabmap_3d_slam.launch world:=false"
    if [[ $FIXED_TRAJ -eq 1 ]]; then
      echo "    roslaunch robot_metrics trajectory_follow.launch" \
           "model_name:=$(gt_model "$robot") cmd_vel_topic:=$(cmd_topic "$robot")" \
           "run_id:=${run_id} ${FOLLOW_OVERRIDES[*]-}" \
           "  [no move_base: SLAM observes only]"
    else
      echo "    roslaunch ${npkg} waypoint_navigation.launch run_id:=${run_id}"
    fi
    return 0
  fi

  cleanup

  # El directorio se VACIA antes de cada corrida.  Sin esto, una corrida que
  # muere a mitad deja sus artefactos y la siguiente sobrescribe unos si y
  # otros no, de modo que run01/ termina conteniendo un metrics.csv de una
  # corrida y un navigation.log de otra.  Eso ya provoco un diagnostico
  # equivocado durante el debug, y en produccion es peor: aggregate_runs.py
  # leeria un metrics.csv viejo y lo meteria en las tablas del paper sin que
  # nada lo advierta.  Es seguro borrarlo: la corrida que viene a continuacion
  # regenera todo el contenido de este directorio.
  rm -rf "${run_dir}"
  mkdir -p "${run_dir}"

  # Written before the run so the seed survives even if the run dies.
  {
    echo "robot: ${robot}"
    echo "run_id: ${run_id}"
    echo "gazebo_seed: ${seed}"
    echo "lidar: ${LIDAR}"
    echo "follower_overrides: ${FOLLOW_NOTE}"
    echo "started: $(date -Is)"
    echo "parity_check_passed: $([[ $PARITY_OK -eq 1 ]] && echo true || echo false)"
    echo "paper_usable: $([[ $PARITY_OK -eq 1 && ${#FOLLOW_OVERRIDES[@]} -eq 0 && -z "${TARGET_RTF}" ]] && echo true || echo false)"
    echo "attempt: ${ATTEMPT}"
  } > "${run_dir}/run_meta.yaml"

  roslaunch "${wpkg}" "${wlaunch}" gui:="${GUI}" paused:=false \
      seed:="${seed}" lidar:="${LIDAR}" \
      > "${run_dir}/gazebo.log" 2>&1 &
  local world_pid=$!
  sleep "${SETTLE}"

  if ! rostopic list >/dev/null 2>&1; then
    err "  ROS master never came up; see ${run_dir}/gazebo.log"
    cleanup
    return 1
  fi

  # El techo de RTF se fija AQUI: Gazebo ya esta arriba con el robot dentro y
  # el SLAM todavia no ha arrancado, asi que la corrida entera transcurre al
  # RTF pedido.  Si falla se aborta: una corrida al RTF equivocado responde a
  # otra pregunta y no se distingue luego por los archivos.
  if [[ -n "${TARGET_RTF}" ]]; then
    if ! python3 "${PKG_DIR}/scripts/set_rtf.py" "${TARGET_RTF}" > "${run_dir}/set_rtf.log" 2>&1; then
      err "  no se pudo fijar el techo de RTF en ${TARGET_RTF}; ver ${run_dir}/set_rtf.log"
      cleanup
      return 1
    fi
    log "  techo de RTF fijado en ${TARGET_RTF}"
  fi

  # world:=false because the world is already up, above, with this run's seed
  # and paused:=false.  Without it the SLAM launch brings its own Gazebo and
  # its own copy of every robot node, which unloads the wheel controllers.
  roslaunch "${npkg}" rtabmap_3d_slam.launch world:=false \
      > "${run_dir}/slam.log" 2>&1 &
  sleep 10

  # The logger is started by waypoint_navigation.launch, which passes run_id
  # through so each repetition lands in its own directory.
  # RViz follows --gui.  The runs that produce the paper's numbers are headless,
  # so this is off by default; --gui is for watching a run, and watching it
  # without RViz means seeing the robot but not the map or the plan.
  # --signal=INT, not the default TERM.  roslaunch treats SIGINT as a Ctrl-C:
  # it shuts its nodes down in order and their shutdown hooks run, which is the
  # only way metrics_logger writes metrics.csv (see the note below).  On SIGTERM
  # it dies without running them, so EVERY run that reached the cap produced no
  # data at all - the run was flagged "kept but flagged" and then discarded two
  # lines later by the "no metrics.csv was written" branch.  Measured
  # 2026-08-20: a full-length rocker_bogie run hit the cap and lost all 1200 s.
  # --kill-after still guarantees termination if the clean shutdown hangs.
  # output_dir has to be passed explicitly.  metrics_logger.launch defaults it
  # to $(env HOME)/metrics_output, and waypoint_navigation.launch used not to
  # override it, so --output moved only THIS script's logs: metrics.csv,
  # columns.csv, events.csv and both .tum files kept going to ~/metrics_output.
  # A --output run therefore split itself across two directories, and since the
  # "rm -rf ${run_dir}" above only empties the campaign's directory, the
  # logger's half survived from the previous run - the mixed-run corruption
  # that rm -rf exists to prevent.  Measured 2026-08-20.
  local rc
  if [[ $FIXED_TRAJ -eq 1 ]]; then
    # Ground-truth trajectory following.  No move_base: the SLAM is an observer.
    timeout --signal=INT --kill-after=45 "${DURATION}" \
        roslaunch robot_metrics trajectory_follow.launch \
        model_name:="$(gt_model "$robot")" \
        robot_name:="${robot}" \
        cmd_vel_topic:="$(cmd_topic "$robot")" \
        wheel_links:="$(wheel_links "$robot")" \
        suspension_joints:="$(suspension_joints "$robot")" \
        effort_joints:="$(effort_joints "$robot")" \
        velocity_joints:="$(velocity_joints "$robot")" \
        suspension_topic:="$(joint_states_topic "$robot")" \
        spawn_yaw:="$(spawn_yaw "$robot")" \
        base_yaw_offset:="$(base_yaw_offset "$robot")" \
        laps:="${LAPS}" \
        run_id:="${run_id}" use_rviz:="${GUI}" \
        output_dir:="${OUTPUT_DIR}" \
        use_run_guard:=true \
        guard_trip_file:="${run_dir}/guard_trip.yaml" \
        "${FOLLOW_OVERRIDES[@]}" \
        > "${run_dir}/navigation.log" 2>&1
    rc=$?
  else
    timeout --signal=INT --kill-after=45 "${DURATION}" \
        roslaunch "${npkg}" waypoint_navigation.launch \
        run_id:="${run_id}" use_rviz:="${GUI}" \
        output_dir:="${OUTPUT_DIR}" \
        > "${run_dir}/navigation.log" 2>&1
    rc=$?
  fi
  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    warn "  hit the ${DURATION}s cap; the run is kept but flagged"
    echo "timed_out: true" >> "${run_dir}/run_meta.yaml"
  fi

  # roslaunch has to exit cleanly for the logger's shutdown hook to write the
  # CSV, so give it a moment before tearing the rest down.
  sleep 8
  cleanup

  # CUANTAS NUBES ENTREGO DE VERDAD EL LIDAR TROCEADO.
  #
  # sweep_distortion.py forma una vuelta cada 100 ms de flujo de rebanadas y
  # publica las que puede.  Hasta el 2026-09-04 descartaba en silencio entre el
  # 7 y el 18.5% de ellas -comprobaba la cobertura del encoder DESPUES de haber
  # consumido las cunas- y la campaña no tenia forma de enterarse: ni el
  # run_meta, ni la cola del nodo, ni el log, que solo se escribe en las
  # vueltas que SI se publican.
  #
  # La tasa efectiva es lo que importa: la deriva por metro es
  # error/nube x nubes/m, asi que dos corridas con distinto Hz efectivo no son
  # comparables aunque el resto coincida.  Por eso va al run_meta y no solo al
  # log.

  # Preserve the graph.  The NEXT run deletes it - rtabmap starts with
  # --delete_db_on_start - so if it is not copied here it is gone.  Two
  # separate things depend on it:
  #
  #  * The optimised graph is the ONLY place the loop closures are visible.
  #    metrics_logger writes est_traj.tum from the pose the SLAM published at
  #    each instant, and loop closure corrects the graph retroactively, so
  #    that file never receives the correction.  Measured 2026-08-23 on
  #    rocker_bogie, one two-lap run: the online trajectory scored ATE
  #    0.998 m rmse while the optimised graph of the SAME run scored 0.177 m.
  #    Reporting only the online figure hides the entire benefit of closing
  #    the loop, and calls it "ATE", which in the SLAM literature means the
  #    optimised one.
  #  * The README's map scoring runs rtabmap-export over each platform's
  #    database after the campaign.  Only the last run's database used to
  #    survive, so that step could not be carried out as documented.
  #
  # Costs about 2 GB per run - 18 GB for a full 3x3 campaign.
  if [[ -f "${HOME}/.ros/rtabmap_3d.db" ]]; then
    cp "${HOME}/.ros/rtabmap_3d.db" "${run_dir}/rtabmap.db"
    rtabmap-export --poses --poses_format 10 \
        --output_dir "${run_dir}" --output opt_traj \
        "${run_dir}/rtabmap.db" > "${run_dir}/export.log" 2>&1
    # rtabmap-export decorates the file name; normalise it.
    local produced
    produced=$(ls -t "${run_dir}"/opt_traj*.txt 2>/dev/null | head -1)
    if [[ -n "${produced}" ]]; then
      mv "${produced}" "${run_dir}/opt_traj.tum"
      log "  optimised graph: $(wc -l < "${run_dir}/opt_traj.tum") poses"
    else
      warn "  could not export the optimised poses; see ${run_dir}/export.log"
    fi
  else
    warn "  no rtabmap database to preserve"
  fi

  # El vigia gana sobre todo lo demas: una corrida abortada por encaje o
  # vuelco puede haber escrito un metrics.csv perfectamente valido de los
  # metros que alcanzo a hacer, y ese archivo es una trampa.
  if [[ -f "${run_dir}/guard_trip.yaml" ]]; then
    err "  corrida abortada: $(sed -n 's/^reason: //p' "${run_dir}/guard_trip.yaml")"
    sed -n 's/^detail: //p' "${run_dir}/guard_trip.yaml" | sed 's/^/    /'
    # El run_meta se escribio al ARRANCAR, cuando todavia decia
    # paper_usable: true.  Corregirlo aca es lo que impide que un intento
    # abortado quede en disco marcado como bueno - que es exactamente lo
    # que paso con differential/run01 el 2026-08-24.
    sed -i 's/^paper_usable: .*/paper_usable: false/' "${run_dir}/run_meta.yaml"
    {
      echo "aborted: true"
      echo "aborted_reason: $(sed -n 's/^reason: //p' "${run_dir}/guard_trip.yaml")"
      echo "aborted_declared_by: $(sed -n 's/^declared_by: //p' "${run_dir}/guard_trip.yaml" || echo run_guard)"
    } >> "${run_dir}/run_meta.yaml"
    return 2
  fi

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
RETRIED=""
for robot in ${ROBOTS}; do
  for run_id in $(seq 1 "${RUNS}"); do
    ATTEMPT=1
    run_dir_top="${OUTPUT_DIR}/${robot}/run$(printf '%02d' "${run_id}")"
    while : ; do
      rc=0
      run_one "${robot}" "${run_id}" || rc=$?
      # rc 2 = el vigia la aborto.  Cualquier otro codigo es un resultado
      # (0 con datos, 1 sin ellos) y no se reintenta: repetir una corrida
      # que simplemente no produjo CSV esconderia el bug que lo causo.
      [[ $rc -eq 2 ]] || break
      if [[ ${ATTEMPT} -gt ${MAX_RETRIES} ]]; then
        err "  ${robot} run ${run_id}: abortada ${ATTEMPT} vez(ces), me rindo"
        break
      fi
      # mv A B mete A DENTRO de B si B ya existe, en vez de fallar.  Con
      # los intentos fallidos sin borrar entre campañas, eso anidaba una
      # corrida dentro de otra: el 2026-09-01 quedo un
      # run01_failed_attempt1/run01 con datos de DOS campañas distintas, y
      # el mv que fallo dejo la corrida sin archivar justo antes de que el
      # reintento la borrase con rm -rf.  Se perdio entera.
      destino="${run_dir_top}_failed_attempt${ATTEMPT}"
      rm -rf "${destino}"
      if ! mv "${run_dir_top}" "${destino}"; then
        err "  no pude archivar el intento fallido en ${destino}"
      fi
      RETRIED="${RETRIED} ${robot}/run${run_id}"
      ATTEMPT=$(( ATTEMPT + 1 ))
      warn "  reintentando ${robot} run ${run_id} - intento ${ATTEMPT} de $(( MAX_RETRIES + 1 ))"
    done
    if [[ $rc -ne 0 ]]; then
      FAILED="${FAILED} ${robot}/run${run_id}"
    fi
  done
done

echo
if [[ -n "${RETRIED}" ]]; then
  warn "corridas repetidas por el vigia (encaje o vuelco):${RETRIED}"
  warn "el intento fallido quedo en run<NN>_failed_attempt<N> con su"
  warn "guard_trip.yaml, sus logs y su metrics.csv parcial."
fi

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

# Per-run figures, the three platforms on the same axes.
#
# WHY THIS IS NOT REDUNDANT WITH aggregate_runs.py.  That one reports the
# numbers averaged over the repetitions, and an average hides the shape.  These
# draw what a single run did along the way - where the error entered, whether
# it accumulated or arrived in one event, and what the estimated path looks
# like next to the ground truth.  On the 2026-08-26 sliced-LiDAR campaign that
# was the difference between "the tracked platform has 3x the ATE" and seeing
# that all three estimates keep the shape of the circuit and simply ROTATE,
# i.e. the error is accumulated heading drift and not a registration failure -
# which the ATE column alone cannot tell you.
#
# run01 because it is the run every campaign has, whatever --runs is set to.
log "Per-run figures (run01 of each platform)..."
AM_FILES=()
AM_NAMES=()
for robot in $ROBOTS; do
  am_csv="${OUTPUT_DIR}/${robot}/run01/metrics.csv"
  if [[ -f "${am_csv}" ]]; then
    AM_FILES+=("${am_csv}")
    AM_NAMES+=("$(display_name "${robot}")")
  fi
done
if [[ ${#AM_FILES[@]} -gt 0 ]]; then
  python3 "${SCRIPT_DIR}/analyze_metrics.py" \
      --files "${AM_FILES[@]}" \
      --names "${AM_NAMES[@]}" \
      --output_dir "${OUTPUT_DIR}/paper_tables" || true
else
  warn "no run01 metrics.csv anywhere; skipping the per-run figures"
fi

# Does chassis agitation cost heading accuracy?  Per-platform correlation
# between how much the chassis was shaken over a window and how much heading
# error the SLAM accumulated across it.  Needs nothing but the runs
# themselves, so it goes on every campaign.
log "Agitation against SLAM heading error..."
python3 "${SCRIPT_DIR}/plot_agitation_heading.py" \
    --results "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}/paper_tables" || true

# El NIVEL del error contra los dos predictores, no su crecimiento.  La de
# arriba pregunta cuanto error NUEVO genero cada ventana, que es la pregunta
# causal; esta pregunta cuanto vale el error mientras el chasis va sacudido,
# que es lo primero que uno mira.  Van juntas a proposito: la segunda es mas
# legible pero confunde coincidencia con causa -la rampa del oeste sacude Y
# descoloca al SLAM, sin que lo uno cause lo otro-.
log "Mean instantaneous ATE against attitude and vibration..."
python3 "${SCRIPT_DIR}/plot_ate_vs_attitude.py" \
    --results "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}/paper_tables" || true

# The same question asked properly, and ONLY on a fixed-trajectory campaign.
#
# The correlation above pools windows along the route, and the route is not
# homogeneous: the step, the corners and the straights differ both in how much
# they shake a chassis AND in how well a LiDAR ICP is constrained there, so a
# pooled correlation partly measures where on the route the window sits.  This
# script removes that by matching windows BY POSITION across platforms, which
# turns it into a paired question - at this metre of mine, did the platform
# that shook more here accumulate more heading error?
#
# That only means anything if the three drove the same ground, which is what
# --fixed-trajectory guarantees and an autonomous campaign does not: there each
# platform steers on its own drifting SLAM estimate and they end up on
# different ground.  Hence the guard.
if [[ $FIXED_TRAJ -eq 1 ]]; then
  log "Position-matched agitation vs heading error..."
  # --path explicitly: the script's own default is an absolute path into one
  # developer's home directory, which is fine when it is run by hand and wrong
  # from a script that is meant to be self-contained.
  python3 "${SCRIPT_DIR}/plot_position_matched.py" \
      --results "${OUTPUT_DIR}" \
      --output_dir "${OUTPUT_DIR}/paper_tables" \
      --path "${SRC_DIR}/robot_metrics/config/reference_path_mine.yaml" || true
fi

# The three figures the Results section needs that nothing above draws: the
# step crossing at 1 kHz, the attitude and rate boxplots, and the plan view of
# run01 per platform.  It also PRINTS the scalars the text quotes - the
# crossing peaks, the attitude percentiles and the window sweep - which is why
# its stdout is kept: those numbers are in the paper and in no table.
#
# It was run by hand until 2026-09-18, which is exactly how a figure ends up
# drawn from one campaign and the table beside it from another.
log "Results-section figures (step crossing, attitude, trajectories)..."
python3 "${SCRIPT_DIR}/plot_results_section.py" \
    --results "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}/paper_tables" \
    --config "${SRC_DIR}/robot_metrics/config/sim_config.yaml" \
    2>&1 | tee "${OUTPUT_DIR}/paper_tables/results_section_numbers.txt" || true

echo
log "Campaign complete. Every table and figure: ${OUTPUT_DIR}/paper_tables"
log "  tables over the repetitions ... table_*.txt, summary_mean_std.csv"
log "  per-run figures ............. trajectory_comparison, ate_over_time,"
log "                                gt_vs_estimated_trajectory, and the rest"
log "  agitation vs heading ........ agitation_vs_heading.{png,eps}"
log "  results section ............. dynamic_profiles_step, attitude_distribution,"
log "                                trajectories_gt_vs_slam; the scalars the"
log "                                text quotes in results_section_numbers.txt"
if [[ $FIXED_TRAJ -eq 1 ]]; then
  log "  position-matched ............ position_matched.{png,eps}"
fi
log "Still to do by hand: export each RTAB-Map database and score the maps,"
log "  rosrun rtabmap_ros rtabmap-export --cloud map.pcd ~/.ros/rtabmap.db"
log "  python3 ${SCRIPT_DIR}/map_vs_groundtruth.py --map map.pcd \\"
log "      --run ${OUTPUT_DIR}/<robot>/run01 --output_dir ${OUTPUT_DIR}/paper_tables"
