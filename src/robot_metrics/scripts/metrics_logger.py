#!/usr/bin/env python3
"""Per-run logger for the three-platform mine-exploration comparison.

WHAT CHANGED FOR THE REVISION
=============================
The previous version wrote a single <robot>.csv that was overwritten by every
run, so only the last of the three repetitions survived and no per-run spread
could be reported.  It also kept no raw ground-truth trajectory, no record of
tracking losses, and no wheel-contact data.

Each run now gets its own directory

    <output_dir>/<robot_name>/run<NN>/
        metrics.csv          time series, one row per ~rate sample
        gt_highrate.csv      ground-truth pose and twist, one row per
                             ~gt_rate_hz sample (see SAMPLING RATES)
        gt_traj.tum          raw ground-truth poses, TUM format
        est_traj.tum         SLAM pose estimates, TUM format
        events.csv           tracking losses, relocalizations, loop closures
        columns.csv          data dictionary: column, unit, description
        run_meta.yaml        provenance for this run

so the three repetitions coexist and aggregate_runs.py can report mean +/- std.

GROUND TRUTH - PROVENANCE
=========================
Ground truth is the pose of the robot model as integrated by the physics
engine, read from /gazebo/model_states (gazebo_msgs/ModelStates).

  * It is the exact state vector.  Gazebo does not filter, smooth or add noise
    to it; there is no estimator between ODE and this topic.
  * It refers to the model's canonical link (base_link), the same frame the
    SLAM estimate is expressed in, so ATE and RPE need no extrinsic
    compensation.  Sensor mounting offsets are documented separately by
    extract_sensor_poses.py.
  * ModelStates carries NO header stamp.  Every sample is therefore stamped
    with the ROS sim clock at reception (use_sim_time:=true).  The topic is
    published once per physics iteration - 2000 Hz at the 0.5 ms step this
    study runs - so the stamping error is bounded by one step, 0.5 ms.  At
    50 Hz that is 2.5% of a sample and irrelevant; at the high-rate stream
    below it is a large fraction of one, which is why that stream is
    decimated in the CALLBACK and not driven by a rospy.Rate: see
    SAMPLING RATES.
  * The unmodified poses are written to gt_traj.tum.  The columns in
    metrics.csv are the same numbers, not a processed version of them.

SAMPLING RATES
==============
Two streams, because they answer different questions and one rate cannot serve
both.

    metrics.csv      ~rate, default 50 Hz
    gt_highrate.csv  ~gt_rate_hz, default 800 Hz -> 1000 Hz delivered

metrics.csv carries IMU (100 Hz), SLAM internals (~1 Hz) and commands (20 Hz)
alongside ground truth.  Logging THAT at 800 Hz would repeat every slow column
16 times and hand the analyst a file that looks like it has 400 Hz of IMU
bandwidth when it has 50.  So it stays at 50 Hz and the high-rate data goes to
its own file with only the columns that genuinely carry that bandwidth.

Why the ground truth needs a second stream at all: the distortion this study
models happens DURING a LiDAR sweep, and the sliced LiDAR is asked for 400 Hz
(rotor 10 Hz x 40 slices).  A 50 Hz log resolves 25 Hz.  Correlating a 25 Hz
observation against a 400 Hz phenomenon cannot work, and energy above 25 Hz is
not absent from the number - it is folded into it by aliasing.

WHY THE DELIVERED RATE IS NOT THE REQUESTED ONE.  The samples are decimated
from the ModelStates callback, one every N messages, so every interval is
exactly N physics steps and there is no rospy.Rate jitter - which at these
periods would be a large fraction of a sample, since /clock only advances in
0.5 ms ticks.  N = floor(physics_rate / requested), so the delivered rate is
the smallest exact divisor of the physics rate that is NOT SLOWER than what was
asked for.  At the default that is floor(2000/800) = 2, i.e. 1000 Hz, uniform
1.000 ms.  If you want 800 Hz exactly you cannot have it at a 0.5 ms step; the
divisors around it are 1000 and 666.7.  run_meta.yaml records requested and
delivered separately, plus the sample count and span, so the count can be
checked against sim time as an independent verification that nothing was
dropped.

COST.  gt_highrate.csv is ~180 bytes a row: about 140 MB for a 780 s two-lap
run at 1000 Hz.  Set gt_rate_hz:=0 to turn the stream off.

This is also stated in robot_metrics/config/sim_config.yaml under
`ground_truth`, which is the version the paper should cite.
"""

import math
import os
import subprocess
import sys
import csv as csvmod
import threading
try:
    import queue as queuemod
except ImportError:          # py2
    import Queue as queuemod

import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from tf.transformations import euler_from_quaternion

# Every other script that imports one of the sibling modules does this first.
# This one is the only one launched as a ROS node, through the catkin wrapper in
# devel/lib/robot_metrics/, which execs the source file without putting its
# directory on sys.path - so the import below raised ModuleNotFoundError and the
# logger died at startup.  Silently: roslaunch reports the traceback in the
# node's log and carries on, so every run completed with no metrics.csv.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics_io import (METRIC_COLUMNS, contact_columns, csv_header,  # noqa: E402
                        effort_columns, suspension_columns,
                        velocity_columns, write_tum, write_units_reference)

def _rotate_world_to_body(v_world, quat):
    """Pasa un vector del marco del mundo al del cuerpo.

    quat es [x, y, z, w], la orientacion del cuerpo EN el mundo, asi que lo
    que se aplica es su conjugada.  Escrito a mano en vez de con
    tf.transformations porque esto corre en el callback de ModelStates, a
    2000 Hz nominales: construir una matriz 4x4 por mensaje se nota.
    """
    x, y, z, w = quat
    vx, vy, vz = v_world
    # v_body = q* (v) q, desarrollado
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [vx - w * tx + (y * tz - z * ty),
            vy - w * ty + (z * tx - x * tz),
            vz - w * tz + (x * ty - y * tx)]


try:
    from gazebo_msgs.msg import ModelStates
    HAS_GAZEBO_MSGS = True
except ImportError:
    HAS_GAZEBO_MSGS = False

try:
    from gazebo_msgs.msg import ContactsState
    HAS_CONTACT_MSGS = True
except ImportError:
    HAS_CONTACT_MSGS = False

try:
    from rtabmap_msgs.msg import Info as RtabmapInfo
    from rtabmap_msgs.msg import OdomInfo
    HAS_RTABMAP_MSGS = True
except ImportError:
    try:
        from rtabmap_ros.msg import Info as RtabmapInfo
        from rtabmap_ros.msg import OdomInfo
        HAS_RTABMAP_MSGS = True
    except ImportError:
        HAS_RTABMAP_MSGS = False

# A pose covariance at or above this value is RTAB-Map's way of publishing
# "I do not know where I am".
LOST_COVARIANCE = 9999.0


class MetricsLogger(object):

    def __init__(self):
        rospy.init_node('metrics_logger', anonymous=False)

        self.robot_name = rospy.get_param('~robot_name', 'robot')
        self.run_id = int(rospy.get_param('~run_id', 1))
        self.model_name = rospy.get_param('~model_name', self.robot_name)
        cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel')
        odom_topic = rospy.get_param('~odom_topic', '/rtabmap/odom')
        base_output = rospy.get_param(
            '~output_dir', os.path.expanduser('~/metrics_output'))
        self.tracking_loss_timeout = float(
            rospy.get_param('~tracking_loss_timeout', 1.0))
        # Two GT poses closer than this are considered the same place, which
        # is how a reported loop closure is judged true or false.
        self.loop_true_radius = float(
            rospy.get_param('~loop_true_radius', 3.0))
        # Proximity links need their OWN radius, and it is not 3 m.
        #
        # A visual loop closure claims "this is the same place", so 3 m is the
        # right question to ask of it.  A proximity detection claims something
        # weaker and geometric: "these two nodes are near enough in space that
        # their scans overlap".  RTAB-Map searches for them inside
        # RGBD/LocalRadius, which this project sets to 20 m, and in a straight
        # mine gallery with a 100 m LiDAR two nodes 14 m apart genuinely do see
        # the same walls - that link is correct and useful, and those links are
        # what lower ATE by 22%.
        #
        # Measured 2026-08-23: judging proximity by loop_true_radius marked all
        # 60 detections of a clean two-lap rocker_bogie run as false positives,
        # with gt_gap 14.3-14.5 m, while the mechanism was working correctly.
        # Keep this equal to RGBD/LocalRadius in rtabmap_3d_slam.launch: a link
        # to a node genuinely farther away than the search radius means the
        # pose estimate was wrong, and THAT is the false positive worth
        # counting.  gt_gap is in every event row, so any other threshold can
        # be applied afterwards without re-running.
        self.proximity_true_radius = float(
            rospy.get_param('~proximity_true_radius', 20.0))

        # SAMPLING RATES.  Both of these used to be the literal 50: one in
        # the run loop and one in the run_meta field, which is how the file
        # came to claim a rate nothing enforced.  See SAMPLING RATES above.
        self.log_rate = float(rospy.get_param('~rate', 50.0))
        # 1 / physics.max_step_size_s in sim_config.yaml = 2000 Hz.  NOT
        # real_time_update_rate_hz: ModelStates arrives once per physics step,
        # every 0.5 ms of SIMULATION time, however fast the wall clock runs.
        # The two coincided until real_time_update_rate went to 1000 on
        # 2026-08-28.  Only used to work out the decimation, so a wrong value
        # here shows up as a delivered rate that does not match the sample
        # count in run_meta.
        self.physics_rate = float(rospy.get_param('~physics_rate_hz', 2000.0))
        self.gt_rate_req = float(rospy.get_param('~gt_rate_hz', 800.0))
        if self.gt_rate_req > 0.0:
            # floor, so the delivered rate is never SLOWER than requested.
            self.gt_decim = max(1, int(math.floor(self.physics_rate
                                                  / self.gt_rate_req)))
            self.gt_rate = self.physics_rate / self.gt_decim
        else:
            self.gt_decim = 0
            self.gt_rate = 0.0

        wheel_links = rospy.get_param('~wheel_links', [])
        if isinstance(wheel_links, str):
            wheel_links = [w.strip() for w in wheel_links.split(',') if w.strip()]
        self.wheel_links = wheel_links

        # Articulaciones pasivas de la suspension (los dos rockers y los dos
        # bogies del Rocker-Bogie).  Vacio en las plataformas que no tienen:
        # las columnas simplemente no salen, igual que con wheel_links.
        susp = rospy.get_param('~suspension_joints', [])
        if isinstance(susp, str):
            susp = [j.strip() for j in susp.split(',') if j.strip()]
        self.suspension_joints = susp
        # /rocker_bogie/joint_states, no "suspension_joint_states": el
        # <topicName> del plugin es inerte, ver la nota en
        # ensamblajeurdf.gazebo.  En ese topic se alternan los mensajes del
        # plugin (los 4 pasivos) con los del joint_state_controller (los 12
        # actuados), y por eso suspension_cb busca por nombre.
        self.suspension_topic = rospy.get_param(
            '~suspension_topic', '/rocker_bogie/joint_states')

        # Articulaciones MOTRICES de las que se quiere el par.  Salen del mismo
        # mensaje que los angulos pasivos -- el joint_state_controller rellena
        # effort[] -- asi que no hay suscriptor nuevo ni topic nuevo.
        eff = rospy.get_param('~effort_joints', [])
        if isinstance(eff, str):
            eff = [j.strip() for j in eff.split(',') if j.strip()]
        self.effort_joints = eff

        # Y su velocidad, del mismo velocity[] del mismo mensaje.  Sin ella el
        # par no distingue "se le manda mas rapido" de "no sigue la consigna"
        # de "patina": ver velocity_columns() en metrics_io.
        vel = rospy.get_param('~velocity_joints', [])
        if isinstance(vel, str):
            vel = [j.strip() for j in vel.split(',') if j.strip()]
        self.velocity_joints = vel

        self.run_dir = os.path.join(
            base_output, self.robot_name, 'run{:02d}'.format(self.run_id))
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.columns = (list(METRIC_COLUMNS)
                        + contact_columns(self.wheel_links)
                        + suspension_columns(self.suspension_joints)
                        + effort_columns(self.effort_joints)
                        + velocity_columns(self.velocity_joints))

        self.rows = []
        self.gt_poses = []      # (t, x, y, z, qx, qy, qz, qw)
        self.est_poses = []
        self.events = []        # (t, type, detail, gt_x, gt_y, gt_distance)

        self._init_state()
        self._open_gt_highrate()
        self._init_subscribers(odom_topic, cmd_vel_topic)

        self.rate = rospy.Rate(self.log_rate)
        rospy.on_shutdown(self.save)

        rospy.loginfo('MetricsLogger: robot=%s run=%d', self.robot_name,
                      self.run_id)
        rospy.loginfo('  output    : %s', self.run_dir)
        rospy.loginfo('  odom      : %s', odom_topic)
        rospy.loginfo('  cmd_vel   : %s', cmd_vel_topic)
        rospy.loginfo('  gt model  : %s (via /gazebo/model_states, unfiltered)',
                      self.model_name)
        if self.wheel_links:
            rospy.loginfo('  contacts  : %d wheels', len(self.wheel_links))
        if self.suspension_joints:
            rospy.loginfo('  suspension: %d joints on %s',
                          len(self.suspension_joints), self.suspension_topic)
        if self.effort_joints:
            rospy.loginfo('  efforts   : %d driven joints',
                          len(self.effort_joints))
        if self.velocity_joints:
            rospy.loginfo('  velocities: %d driven joints',
                          len(self.velocity_joints))
        rospy.loginfo('  metrics   : %.1f Hz', self.log_rate)
        if self.gt_decim:
            rospy.loginfo('  gt hi-rate: %.1f Hz (asked %.1f; every %d-th '
                          'ModelStates at %.0f Hz physics) -> gt_highrate.csv',
                          self.gt_rate, self.gt_rate_req, self.gt_decim,
                          self.physics_rate)
            if abs(self.gt_rate - self.gt_rate_req) > 1e-6:
                rospy.logwarn('  gt_rate_hz %.1f is not a divisor of the %.0f Hz '
                              'physics rate; delivering %.1f Hz instead, which '
                              'is the next exact divisor that is not slower',
                              self.gt_rate_req, self.physics_rate, self.gt_rate)
        else:
            rospy.loginfo('  gt hi-rate: off (gt_rate_hz = 0)')

    # ------------------------------------------------------------------
    def _open_gt_highrate(self):
        """Stream, not buffer.  A 1000 Hz two-lap run is ~780 000 rows; held
        as a list of Python floats that is about a gigabyte of objects, and it
        would all be lost if the run died before the shutdown hook."""
        self._gt_hr_file = None
        self._gt_hr_writer = None
        self._gt_hr_count = 0
        self._gt_hr_msgs = 0
        self._gt_hr_first_t = None
        self._gt_hr_last_t = None
        # Antes del return: save() y _gt_highrate_meta() los leen aunque el
        # stream este apagado.
        self._gt_hr_dropped_q = 0
        self._gt_hr_thread = None
        if not self.gt_decim:
            return
        path = os.path.join(self.run_dir, 'gt_highrate.csv')
        self._gt_hr_file = open(path, 'w')
        self._gt_hr_writer = csvmod.writer(self._gt_hr_file)
        # La fila se ENCOLA en el callback y la escribe un hilo aparte.  Hacer
        # el writerow dentro del callback lo bloqueaba a 1000 Hz y era la otra
        # mitad de la perdida de mensajes; ver la nota del suscriptor.
        self._gt_hr_q = queuemod.Queue(maxsize=200000)
        self._gt_hr_stop = threading.Event()
        self._gt_hr_thread = threading.Thread(target=self._gt_hr_drain)
        self._gt_hr_thread.daemon = True
        self._gt_hr_thread.start()
        self._gt_hr_writer.writerow([
            'timestamp[s]',
            'gt_x[m]', 'gt_y[m]', 'gt_z[m]',
            'gt_qx[-]', 'gt_qy[-]', 'gt_qz[-]', 'gt_qw[-]',
            'gt_roll[rad]', 'gt_pitch[rad]', 'gt_yaw[rad]',
            'gt_vx[m/s]', 'gt_vy[m/s]', 'gt_vz[m/s]',
            'gt_wx[rad/s]', 'gt_wy[rad/s]', 'gt_wz[rad/s]',
            'gt_ax[m/s^2]', 'gt_ay[m/s^2]', 'gt_az[m/s^2]',
        ])

    def _gt_hr_drain(self):
        """Vacia la cola de gt_highrate en el fichero.

        Corre en su propio hilo: el callback de ModelStates solo encola, que
        es lo unico que puede hacer a 2000 Hz sin perder mensajes.
        """
        while True:
            try:
                fila = self._gt_hr_q.get(timeout=0.2)
            except queuemod.Empty:
                if self._gt_hr_stop.is_set():
                    return
                continue
            if fila is None:
                return
            try:
                self._gt_hr_writer.writerow(fila)
            except (ValueError, IOError):
                # fichero ya cerrado en el apagado: no hay a donde escribir
                return

    # ------------------------------------------------------------------
    def _init_state(self):
        self.pitch = self.roll = self.yaw = 0.0
        self.accel = [0.0, 0.0, 0.0]
        self.ang_vel = [0.0, 0.0, 0.0]
        self.ang_acc_x = self.ang_acc_z = 0.0
        self._prev_wx = self._prev_wz = None
        self._prev_imu_t = None

        self.odom = [0.0] * 3
        self.odom_rpy = [0.0] * 3
        self.odom_vx = self.odom_vyaw = 0.0
        self.cmd_vx = self.cmd_vyaw = 0.0

        self.gt = [0.0] * 3
        self.gt_rpy = [0.0] * 3
        self.gt_v = [0.0, 0.0]
        self.gt_quat = [0.0, 0.0, 0.0, 1.0]
        self.gt_distance = 0.0
        # Aceleracion propia de la verdad-terreno, en ejes del cuerpo.  Se
        # deriva del twist de ModelStates sobre CADA mensaje recibido, no
        # sobre los diezmados, porque derivar sobre un intervalo diezmado
        # perderia justo el ancho de banda por el que existe gt_highrate.
        self.gt_a = [0.0, 0.0, 0.0]
        self._prev_gt_v3 = None
        self._prev_gt_v_t = None
        self._prev_gt_xy = None
        self.gt_ready = False

        self.loop_closures = 0
        self.proximity = 0
        self.inliers = self.matches = 0
        self.icp_ratio = 0.0
        self.icp_corr = 0
        self.proc_time = 0.0

        self.tracking_lost = 0
        self.tracking_loss_count = 0
        self.relocalization_count = 0
        self._last_odom_time = None

        # node id -> ground-truth position, used to classify loop closures
        self._node_gt = {}

        self.contact = {l: 0 for l in self.wheel_links}
        self.normal_force = {l: 0.0 for l in self.wheel_links}
        # NaN, no 0.0: un pivote a cero es una postura valida, asi que un cero
        # por "no ha llegado nada" se promediaria como si fuese una medida.
        self.suspension = {j: float('nan') for j in self.suspension_joints}
        self.effort = {j: float('nan') for j in self.effort_joints}
        self.velocity = {j: float('nan') for j in self.velocity_joints}
        self._susp_seen = 0

        self.tf_dx = self.tf_dy = self.tf_dyaw = 0.0
        self._prev_tf = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def _init_subscribers(self, odom_topic, cmd_vel_topic):
        rospy.Subscriber('/imu/data', Imu, self.imu_cb)
        rospy.Subscriber(odom_topic, Odometry, self.odom_cb)
        rospy.Subscriber(cmd_vel_topic, Twist, self.cmd_cb)

        if HAS_GAZEBO_MSGS:
            # queue_size=1 PERDIA EL 43 % DE LOS PASOS DE FISICA.  Medido en
            # el run01 del 2026-08-31: 233 355 filas en 403.8 s, o sea 577 Hz
            # de los 1000 que declaraba, con huecos de 8-10 ms repitiendose a
            # 105 Hz -- la cadencia del IMU, que es de 100 Hz.  Mientras el
            # hilo del callback esta ocupado, todo lo que llega con la cola en
            # 1 se descarta, y como la diezmacion cuenta MENSAJES RECIBIDOS,
            # cada perdida rompe la premisa de "cada intervalo son exactamente
            # gt_decim pasos".  La cola profunda absorbe la rafaga y
            # tcp_nodelay quita el retardo de Nagle, que a 2000 Hz pesa.
            rospy.Subscriber('/gazebo/model_states', ModelStates,
                             self.model_states_cb, queue_size=2000,
                             tcp_nodelay=True)
        else:
            rospy.logerr('gazebo_msgs missing: NO GROUND TRUTH will be '
                         'recorded and ATE/RPE cannot be computed.')

        if HAS_CONTACT_MSGS:
            for link in self.wheel_links:
                rospy.Subscriber('/wheel_contacts/{}'.format(link),
                                 ContactsState, self.contact_cb,
                                 callback_args=link, queue_size=1)
        elif self.wheel_links:
            rospy.logwarn('gazebo_msgs.ContactsState unavailable: wheel '
                          'contact columns will stay at zero.')

        if self.suspension_joints or self.effort_joints:
            rospy.Subscriber(self.suspension_topic, JointState,
                             self.suspension_cb, queue_size=1)

        if HAS_RTABMAP_MSGS and 'rtabmap' in odom_topic:
            rospy.Subscriber('/rtabmap/info', RtabmapInfo, self.info_cb)
            rospy.Subscriber('/rtabmap/odom_info', OdomInfo, self.odom_info_cb)

    # ---------------- callbacks ----------------
    def imu_cb(self, msg):
        q = msg.orientation
        self.roll, self.pitch, self.yaw = euler_from_quaternion(
            [q.x, q.y, q.z, q.w])
        self.accel = [msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z]
        wx, wy, wz = (msg.angular_velocity.x, msg.angular_velocity.y,
                      msg.angular_velocity.z)
        t = msg.header.stamp.to_sec()
        if self._prev_imu_t is not None:
            dt = t - self._prev_imu_t
            if dt > 0:
                self.ang_acc_x = (wx - self._prev_wx) / dt
                self.ang_acc_z = (wz - self._prev_wz) / dt
        self._prev_wx, self._prev_wz, self._prev_imu_t = wx, wz, t
        self.ang_vel = [wx, wy, wz]

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom = [p.x, p.y, p.z]
        self.odom_rpy = list(euler_from_quaternion([q.x, q.y, q.z, q.w]))
        self.odom_vx = msg.twist.twist.linear.x
        self.odom_vyaw = msg.twist.twist.angular.z
        self._last_odom_time = rospy.get_time()

        # RTAB-Map signals a lost track with a saturated covariance.
        lost = msg.pose.covariance[0] >= LOST_COVARIANCE
        self._set_tracking(lost, 'covariance')

        self.est_poses.append((rospy.get_time(), p.x, p.y, p.z,
                               q.x, q.y, q.z, q.w))

    def cmd_cb(self, msg):
        self.cmd_vx = msg.linear.x
        self.cmd_vyaw = msg.angular.z

    def model_states_cb(self, msg):
        try:
            i = msg.name.index(self.model_name)
        except ValueError:
            rospy.logwarn_throttle(
                10.0, 'model "%s" not in /gazebo/model_states; names=%s',
                self.model_name, list(msg.name))
            return

        pose = msg.pose[i]
        tw = msg.twist[i]
        q = pose.orientation
        self.gt = [pose.position.x, pose.position.y, pose.position.z]
        self.gt_quat = [q.x, q.y, q.z, q.w]
        self.gt_rpy = list(euler_from_quaternion([q.x, q.y, q.z, q.w]))
        self.gt_v = [tw.linear.x, tw.linear.y]

        if self._prev_gt_xy is not None:
            self.gt_distance += math.hypot(self.gt[0] - self._prev_gt_xy[0],
                                           self.gt[1] - self._prev_gt_xy[1])
        self._prev_gt_xy = (self.gt[0], self.gt[1])
        self.gt_ready = True

        # d(v_mundo)/dt rotada a ejes del cuerpo.  Gazebo entrega la
        # velocidad, no la aceleracion, asi que hay que derivarla; se hace
        # con el dt REAL entre mensajes y no con el nominal, porque la cola
        # de este suscriptor pierde mensajes y el intervalo no es constante.
        now_cb = rospy.get_time()
        v3 = (tw.linear.x, tw.linear.y, tw.linear.z)
        if self._prev_gt_v3 is not None and self._prev_gt_v_t is not None:
            dt_cb = now_cb - self._prev_gt_v_t
            if 1e-9 < dt_cb < 0.1:
                aw = [(v3[k] - self._prev_gt_v3[k]) / dt_cb for k in range(3)]
                self.gt_a = _rotate_world_to_body(aw, self.gt_quat)
        self._prev_gt_v3 = v3
        self._prev_gt_v_t = now_cb

        # High-rate stream.  Decimated by message COUNT, not by a clock: every
        # interval is then exactly gt_decim physics steps.  Angular velocity is
        # included because it is the half the IMU cannot give at this bandwidth
        # - its gyro quantises at 0.25 mrad/s and carries a noise floor that
        # would sit on top of whatever is being looked for above 100 Hz.
        if self._gt_hr_writer is not None:
            self._gt_hr_msgs += 1
            if self._gt_hr_msgs % self.gt_decim == 0:
                now = rospy.get_time()
                if self._gt_hr_first_t is None:
                    self._gt_hr_first_t = now
                self._gt_hr_last_t = now
                self._gt_hr_count += 1
                try:
                    self._gt_hr_q.put_nowait(['%.6f' % v for v in (
                        now,
                        pose.position.x, pose.position.y, pose.position.z,
                        q.x, q.y, q.z, q.w,
                        self.gt_rpy[0], self.gt_rpy[1], self.gt_rpy[2],
                        tw.linear.x, tw.linear.y, tw.linear.z,
                        tw.angular.x, tw.angular.y, tw.angular.z,
                        self.gt_a[0], self.gt_a[1], self.gt_a[2],
                    )])
                except queuemod.Full:
                    # El disco no da abasto.  Se cuenta y se sigue: perder una
                    # fila es mejor que bloquear el callback, que es
                    # exactamente el fallo que esto viene a arreglar.
                    self._gt_hr_dropped_q += 1

    def contact_cb(self, msg, link):
        if not msg.states:
            self.contact[link] = 0
            self.normal_force[link] = 0.0
            return
        self.contact[link] = 1
        # CORRECTED 2026-08-18, after measuring what this sensor actually
        # reports.  The original was
        #     for st in msg.states: total += |force|      # no division
        # and an intermediate "fix" changed |force| to force.z, which was worse.
        #
        # (1) THE REAL BUG: no division.  The sensors run at 100 Hz against
        #     1000 Hz physics, so each message carries exactly 10 ContactState
        #     entries - one per step, measured, uniform on all six wheels.
        #     Summing them reported TEN TIMES the load, which is what the middle
        #     panel of the wheel-contact figure has been showing.
        #
        # (2) IT MUST BE THE MAGNITUDE, NOT force.z.  The bumper plugin does not
        #     report in the world frame despite frameName being world: the wheel
        #     link frames come out of CAD with different orientations per side,
        #     so on the right-hand wheels the load lands in Fx, not Fz.  Reading
        #     force.z gave left/right 20:1 and a total of 31% of the weight.
        #     The magnitude gives 1.26:1 and 451.0 N against a 441.5 N weight -
        #     a 2% match, and the same equal penetration the contacts show
        #     (1.96 mm left, 1.84 mm right).  So the magnitude is the load.
        #
        #     It does include the tangential friction component, so this is a
        #     contact force rather than strictly a normal force.  Labelling it
        #     "normal force" is only accurate while the wheel is not pushing
        #     hard along the ground; the figure caption should say so.
        n = len(msg.states)
        total = 0.0
        for st in msg.states:
            f = st.total_wrench.force
            total += math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z)
        self.normal_force[link] = total / n if n else 0.0

    def suspension_cb(self, msg):
        """Angulo de cada pivote pasivo, del publicador de Gazebo a 200 Hz.

        Se toma por NOMBRE y no por indice: el orden de JointState no esta
        garantizado, y leerlo por posicion es como se cuelan las permutaciones
        silenciosas entre plataformas.  Una articulacion que no venga en el
        mensaje se queda en NaN, que es lo que era antes de arrancar.
        """
        self._susp_seen += 1
        for name, pos in zip(msg.name, msg.position):
            if name in self.suspension:
                self.suspension[name] = pos
        # effort[] puede venir vacio segun quien publique: en ESTE topic se
        # alternan el plugin de suspension (solo posicion) y el
        # joint_state_controller (posicion, velocidad y par).  zip() corta por
        # el mas corto, asi que un mensaje sin par simplemente no toca nada.
        for name, tau in zip(msg.name, msg.effort):
            if name in self.effort:
                self.effort[name] = tau
        # velocity[] viene del joint_state_controller igual que effort[]; el
        # plugin de suspension publica mensajes sin ella y zip() los ignora.
        for name, w in zip(msg.name, msg.velocity):
            if name in self.velocity:
                self.velocity[name] = w

    def info_cb(self, msg):
        # Remember where each map node was created, so a later loop closure
        # onto that node can be judged against ground truth.
        if getattr(msg, 'refId', 0) > 0 and self.gt_ready:
            self._node_gt[msg.refId] = (self.gt[0], self.gt[1])

        lc_id = getattr(msg, 'loopClosureId', 0)
        if lc_id > 0:
            self.loop_closures += 1
            self._classify_closure('loop_closure', lc_id)

        # Proximity detections get an event too, and the same ground-truth
        # verdict.  Until 2026-08-23 they were only counted, and the counter
        # reached nothing that aggregate_runs.py reads - it builds the paper
        # tables out of events.  That mattered because of what the closures
        # on this sensor suite actually are: the camera publishes no depth,
        # no keypoint ever gets a 3D position, so Vis/EstimationType=1 (PnP)
        # can never run and the VISUAL mechanism reports zero forever.  Every
        # closure that does happen is a LiDAR proximity detection - 89 of
        # them on a two-lap rocker_bogie run.  Counting only loop_closure
        # events therefore published "Loop closures reported: 0" while the
        # real closures went unreported and, worse, unvalidated: nothing
        # measured how many of them were false positives.
        prox_id = getattr(msg, 'proximityDetectionId', 0)
        if prox_id > 0:
            self.proximity += 1
            self._classify_closure('proximity_detection', prox_id)

    def _classify_closure(self, kind, node_id):
        """True positive if the two linked nodes really are the same place.

        Used for both mechanisms - see the note in info_cb about why
        proximity detections have to go through it as well.
        """
        radius = (self.proximity_true_radius
                  if kind == 'proximity_detection' else self.loop_true_radius)
        ref = self._node_gt.get(node_id)
        if ref is None or not self.gt_ready:
            verdict, dist = 'unknown', float('nan')
        else:
            dist = math.hypot(self.gt[0] - ref[0], self.gt[1] - ref[1])
            verdict = 'true' if dist <= radius else 'false'
        self.log_event(kind,
                       'id={} verdict={} gt_gap={:.3f}m'.format(
                           node_id, verdict, dist))

    def odom_info_cb(self, msg):
        self.inliers = msg.inliers
        self.matches = msg.matches
        self.icp_ratio = getattr(msg, 'icpInliersRatio', 0.0)
        self.icp_corr = getattr(msg, 'icpCorrespondences', 0)
        self.proc_time = getattr(msg, 'timeEstimation', 0.0)
        if hasattr(msg, 'lost'):
            self._set_tracking(bool(msg.lost), 'odom_info')

    # ---------------- tracking-loss bookkeeping ----------------
    def _set_tracking(self, lost, source):
        if lost and not self.tracking_lost:
            self.tracking_lost = 1
            self.tracking_loss_count += 1
            self.log_event('tracking_loss', 'source={}'.format(source))
        elif not lost and self.tracking_lost:
            self.tracking_lost = 0
            self.relocalization_count += 1
            self.log_event('relocalization', 'source={}'.format(source))

    def log_event(self, kind, detail):
        """Record an event with where on the route it happened."""
        self.events.append((rospy.get_time(), kind, detail,
                            self.gt[0], self.gt[1], self.gt_distance))
        rospy.loginfo('[event] %-15s %s  at gt=(%.2f, %.2f) s=%.1fm',
                      kind, detail, self.gt[0], self.gt[1], self.gt_distance)

    def _check_odom_timeout(self):
        if self._last_odom_time is None:
            return
        if rospy.get_time() - self._last_odom_time > self.tracking_loss_timeout:
            self._set_tracking(True, 'timeout')

    def _update_tf_correction(self):
        try:
            tr = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0),
                                                 rospy.Duration(0.05))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return
        t = tr.transform.translation
        q = tr.transform.rotation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        if self._prev_tf is not None:
            self.tf_dx = t.x - self._prev_tf[0]
            self.tf_dy = t.y - self._prev_tf[1]
            self.tf_dyaw = yaw - self._prev_tf[2]
        self._prev_tf = (t.x, t.y, yaw)

    # ---------------- main loop ----------------
    def run(self):
        while not rospy.is_shutdown():
            self._update_tf_correction()
            self._check_odom_timeout()

            t = rospy.get_time()
            row = [
                t,
                self.pitch, self.roll, self.yaw,
                self.accel[0], self.accel[1], self.accel[2],
                self.ang_vel[0], self.ang_vel[1], self.ang_vel[2],
                self.ang_acc_x, self.ang_acc_z,
                self.odom[0], self.odom[1], self.odom[2],
                self.odom_rpy[0], self.odom_rpy[1], self.odom_rpy[2],
                self.odom_vx, self.odom_vyaw,
                self.cmd_vx, self.cmd_vyaw,
                self.gt[0], self.gt[1], self.gt[2],
                self.gt_rpy[0], self.gt_rpy[1], self.gt_rpy[2],
                self.gt_v[0], self.gt_v[1],
                self.gt_a[0], self.gt_a[1], self.gt_a[2],
                self.gt_distance,
                self.loop_closures, self.proximity,
                self.inliers, self.matches,
                self.icp_ratio, self.icp_corr, self.proc_time,
                self.tracking_lost, self.tracking_loss_count,
                self.relocalization_count,
                math.hypot(self.tf_dx, self.tf_dy), self.tf_dyaw,
            ]
            for link in self.wheel_links:
                row.append(self.contact[link])
                row.append(self.normal_force[link])
            for j in self.suspension_joints:
                row.append(self.suspension[j])
            for j in self.effort_joints:
                row.append(self.effort[j])
            for j in self.velocity_joints:
                row.append(self.velocity[j])

            self.rows.append(row)

            if self.gt_ready:
                self.gt_poses.append((t,) + tuple(self.gt) + tuple(self.gt_quat))

            self.rate.sleep()

    # ---------------- output ----------------
    def save(self):
        # Before the early return below: an aborted run that never filled
        # metrics.csv can still have written a usable high-rate stream, and
        # leaving the handle open loses whatever is in the buffer.
        if self._gt_hr_file is not None:
            # Vaciar la cola antes de cerrar: lo que quede dentro son filas ya
            # medidas que se perderian, y en una corrida abortada es
            # justamente el final lo que interesa.
            if getattr(self, '_gt_hr_thread', None) is not None:
                self._gt_hr_stop.set()
                try:
                    self._gt_hr_q.put_nowait(None)
                except queuemod.Full:
                    pass
                self._gt_hr_thread.join(timeout=30.0)
                if self._gt_hr_thread.is_alive():
                    rospy.logwarn('gt_highrate: el hilo escritor no termino en '
                                  '30 s; quedan %d filas sin escribir',
                                  self._gt_hr_q.qsize())
                self._gt_hr_thread = None
            if self._gt_hr_dropped_q:
                rospy.logwarn('gt_highrate: %d filas descartadas por cola '
                              'llena (el disco no siguio el ritmo)',
                              self._gt_hr_dropped_q)
            self._gt_hr_file.close()
            self._gt_hr_file = None

        if not self.rows:
            rospy.logwarn('No data collected; nothing written.')
            return

        path = os.path.join(self.run_dir, 'metrics.csv')
        with open(path, 'w') as f:
            w = csvmod.writer(f)
            w.writerow(csv_header(self.columns))
            w.writerows(self.rows)

        write_units_reference(self.columns,
                              os.path.join(self.run_dir, 'columns.csv'))
        write_tum(os.path.join(self.run_dir, 'gt_traj.tum'), self.gt_poses)
        write_tum(os.path.join(self.run_dir, 'est_traj.tum'), self.est_poses)

        with open(os.path.join(self.run_dir, 'events.csv'), 'w') as f:
            w = csvmod.writer(f)
            w.writerow(['timestamp[s]', 'event[-]', 'detail[-]',
                        'gt_x[m]', 'gt_y[m]', 'gt_distance[m]'])
            w.writerows(self.events)

        self._write_meta()

        rospy.loginfo('Run %s/%02d saved to %s',
                      self.robot_name, self.run_id, self.run_dir)
        rospy.loginfo('  %d samples, %d GT poses, %d events',
                      len(self.rows), len(self.gt_poses), len(self.events))
        if self.gt_decim:
            rospy.loginfo('  %d high-rate GT samples from %d messages '
                          '(%.1f Hz over %.1f s)', self._gt_hr_count,
                          self._gt_hr_msgs, self.gt_rate,
                          self._gt_hr_span())
        rospy.loginfo('  distance travelled (GT): %.2f m', self.gt_distance)
        rospy.loginfo('  tracking losses: %d, relocalizations: %d',
                      self.tracking_loss_count, self.relocalization_count)
        rospy.loginfo('  loop closures: %d (visual), %d (proximity)',
                      self.loop_closures, self.proximity)

    def _write_meta(self):
        duration = (self.rows[-1][0] - self.rows[0][0]) if self.rows else 0.0
        meta = {
            'robot': self.robot_name,
            'run_id': self.run_id,
            'gazebo_model': self.model_name,
            'samples': len(self.rows),
            'duration_s': round(duration, 3),
            'gt_distance_m': round(self.gt_distance, 3),
            'log_rate_hz': self.log_rate,
            'ground_truth': {
                'topic': '/gazebo/model_states',
                'type': 'gazebo_msgs/ModelStates',
                'filtering': 'none',
                'added_noise': 'none',
                'frame': 'gazebo world frame, model canonical link',
                'timestamping': 'ROS sim clock at reception (no header stamp '
                                'on ModelStates); error bounded by one '
                                '0.5 ms physics step',
            },
            # Recorded separately from what was asked for, and with the raw
            # counts, so the delivered rate can be verified against sim time
            # instead of trusted: samples should be span * rate + 1, and
            # messages should be span * physics_rate.
            'gt_highrate': self._gt_highrate_meta(),
            'slam': {
                'loop_closures': self.loop_closures,
                'proximity_detections': self.proximity,
                'tracking_losses': self.tracking_loss_count,
                'relocalizations': self.relocalization_count,
            },
            'wheel_links': self.wheel_links,
            'suspension_joints': self.suspension_joints,
            'suspension_topic': (self.suspension_topic
                                 if self.suspension_joints else None),
            'suspension_messages': self._susp_seen,
            'effort_joints': self.effort_joints,
            'velocity_joints': self.velocity_joints,
            'ros_time_is_sim_time': rospy.get_param('/use_sim_time', False),
            'git_commit': self._git_commit(),
        }
        with open(os.path.join(self.run_dir, 'run_meta.yaml'), 'w') as f:
            yaml.safe_dump(meta, f, default_flow_style=False, sort_keys=False)

    def _gt_hr_span(self):
        if self._gt_hr_first_t is None or self._gt_hr_last_t is None:
            return 0.0
        return self._gt_hr_last_t - self._gt_hr_first_t

    def _gt_highrate_meta(self):
        if not self.gt_decim:
            return {'enabled': False}
        span = self._gt_hr_span()
        return {
            'enabled': True,
            'file': 'gt_highrate.csv',
            'requested_hz': self.gt_rate_req,
            # nominal_hz es lo que SALDRIA si llegase un ModelStates por paso
            # de fisica.  No es una medida, es una division: physics_hz entre
            # la diezmacion.  Se llamaba delivered_hz y eso la hacia pasar por
            # un dato observado -- el run01 del 31-ago declaraba 1000 Hz
            # habiendo entregado 577.  La cifra que hay que leer es
            # measured_hz.
            'nominal_hz': round(self.gt_rate, 4),
            'measured_hz': round((self._gt_hr_count - 1) / span, 4)
                           if span > 0 and self._gt_hr_count > 1 else 0.0,
            'physics_hz': self.physics_rate,
            'decimation': self.gt_decim,
            'samples': self._gt_hr_count,
            'messages_seen': self._gt_hr_msgs,
            'messages_expected': int(round(span * self.physics_rate))
                                 if span > 0 else 0,
            'rows_dropped_queue_full': self._gt_hr_dropped_q,
            'span_s': round(span, 4),
            'note': 'decimated by RECEIVED message count, so an interval is '
                    '`decimation` physics steps only while nothing is lost. '
                    'Compare messages_seen against messages_expected: a '
                    'shortfall means ModelStates was dropped before the '
                    'callback, and measured_hz falls below nominal_hz.',
        }

    @staticmethod
    def _git_commit():
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            out = subprocess.check_output(
                ['git', '-C', here, 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL)
            return out.decode().strip()
        except Exception:
            return 'unknown'


if __name__ == '__main__':
    try:
        MetricsLogger().run()
    except rospy.ROSInterruptException:
        pass
