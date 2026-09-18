#!/usr/bin/env python3
"""Abort a run that is wedged or has rolled over, so the campaign can redo it.

WHY THIS EXISTS
===============
Two failure modes waste a whole run and are not detectable from the metrics
afterwards without reading them by hand:

  wedged    a velocity is commanded and the platform does not go anywhere.
            trajectory_follower has its own stall guard, but that one is a
            RECOVERY: it backs off and tries again, and only gives up after
            max_stalls (8).  It is meant to get past a bad patch, and it
            usually does.  This node is the layer above it - the one that
            says the recovery itself has failed and the run is not worth
            finishing.  Its window is therefore deliberately much longer than
            the follower's stall_time, so a normal back-off-and-retry never
            trips it.

  rolled    the platform goes over.  NOTHING watched for this before.
            rearing_guard watches |pitch| only, and only exists in the
            move_base runs, because it works by cancelling a move_base goal -
            there is no move_base in a fixed-trajectory run.  Roll was never
            watched anywhere, and a platform on its side keeps producing
            perfectly well-formed metrics.csv rows.

WHAT IT USES
============
Ground truth, from /gazebo/model_states - not the SLAM odometry and not the
wheels.  Both of the other two are wrong in exactly the situations this node
exists for: the wheels spin while the vehicle is wedged, and the SLAM estimate
drifts.  The decision to throw a run away has to be made on the state vector,
not on an estimate of it.

WHAT IT DOES
============
Writes ~trip_file and shuts itself down.  It is launched required="true", so
roslaunch then tears the run down in order - which matters, because that is
what lets metrics_logger's shutdown hook write its CSV, so the failed attempt
is still on disk to look at.  run_campaign.sh sees the trip file, files the
attempt under run<NN>_failed_attempt<N>, and runs it again.

It never publishes to cmd_vel.  Steering belongs to the follower and to
rearing_guard; this node only observes and decides.
"""

import math
import os
from collections import deque

import rospy
from geometry_msgs.msg import Twist

try:
    from gazebo_msgs.msg import ModelStates
except ImportError:                                        # pragma: no cover
    ModelStates = None


def roll_pitch_yaw(q):
    """Roll y pitch en GRADOS, yaw en RADIANES.

    Roll y pitch son para el vuelco; el yaw hace falta para saber si un
    robot al que se le manda girar esta girando de verdad.

    El base_link del Rocker-Bogie mira a -x, que es un desfase CONSTANTE de
    yaw: no toca ni roll ni pitch, y aca solo se usan DIFERENCIAS de yaw,
    asi que tampoco importa.
    """
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    sinp = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return (math.degrees(math.atan2(sinr, cosr)),
            math.degrees(math.asin(sinp)),
            math.atan2(siny, cosy))


def wrap(a):
    """Diferencia de angulos al intervalo (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class RunGuard(object):

    def __init__(self):
        self.model_name = rospy.get_param('~model_name')
        cmd_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel')
        self.trip_file = rospy.get_param('~trip_file', '')

        # Nothing is judged until the platform has settled and the follower has
        # actually started driving.  A run spends its first seconds dropping
        # onto the terrain with zero command, which is neither stuck nor tipped
        # but would look like both.
        self.grace = rospy.get_param('~grace', 30.0)              # s

        # --- wedged ---------------------------------------------------------
        # "a speed is being asked for"
        self.cmd_min = rospy.get_param('~cmd_min', 0.10)          # m/s
        self.ang_min = rospy.get_param('~ang_min', 0.20)          # rad/s
        # ... for this long ...
        self.stuck_time = rospy.get_param('~stuck_time', 25.0)    # s
        # ... and never got further than this from where it started the
        # window.  CORREGIDO 2026-08-24: esto media la SUMA de la longitud
        # de arco entre muestras, y con eso el back-off del seguidor
        # (-0.25 m/s durante 2 s, despues adelante otra vez, cada ~7 s)
        # acumulaba varios metros de camino sin ir a ninguna parte, muy
        # por encima del umbral.  El vigia leia eso como avance y no
        # disparaba: medido en differential/run01, encajado en el escalon
        # 44 s con ocho stalls y 'run_guard: no trip'.  Sumar arco PREMIA
        # la sacudida, que es justo lo contrario de lo que hace falta.
        # La excursion maxima respecto del inicio de la ventana no: un
        # back-off de +/-0.5 m da 0.5 m de excursion y nada mas.
        # 1.5 m con 25 s de ventana deja margen de sobra - conduciendo
        # normal a 0.46 m/s la excursion son ~11 m - y sigue muy por
        # encima de los 0.5 m del back-off.
        self.progress_min = rospy.get_param('~progress_min', 1.5)  # m
        # Fraction of the window that must have carried a command.  Below 1.0
        # so that the follower's own back-off - which briefly commands reverse
        # and then zero - does not reset the window and mask a real wedge.
        self.cmd_fraction = rospy.get_param('~cmd_fraction', 0.7)
        # AVANCE ANGULAR minimo, cuando lo que se comanda es un giro.
        # AGREGADO 2026-08-24 por un falso positivo: en la prueba de
        # determinismo el vigia disparo con el comando en 0.00 m/s y
        # 1.00 rad/s - o sea girando en el sitio, donde la excursion
        # LINEAL es ~0 legitimamente y no dice nada.  Medir el avance en
        # la misma moneda del comando es lo que arregla eso: si se pide
        # girar, lo que tiene que crecer es el rumbo.
        # 0.5 rad (~29 deg) en 25 s: girando a 1.0 rad/s se cubre en medio
        # segundo, y un robot trabado que cabecea +/-5 deg no llega.
        # Por debajo de pi para que el envolvimiento no lo haga inalcanzable.
        self.yaw_progress_min = rospy.get_param('~yaw_progress_min', 0.5)

        # --- rolled ---------------------------------------------------------
        # Roll is the tip-over axis.  Pitch gets a wider limit because a large
        # nose-up angle against an obstacle is a WEDGE, and the wedge is the
        # other rule's business - tripping it here would report the wrong
        # reason.
        self.roll_limit = rospy.get_param('~roll_limit', 50.0)    # deg
        self.pitch_limit = rospy.get_param('~pitch_limit', 60.0)  # deg
        # Held for this long, so a single bad contact solve cannot trip it.
        self.tip_time = rospy.get_param('~tip_time', 1.5)         # s

        self.rate_hz = rospy.get_param('~rate', 10.0)

        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.pose = None                 # (x, y, z, roll, pitch, yaw)
        self.first_seen = None
        self.tipped_since = None
        self.track = deque()             # (t, x, y, yaw, trans?, rot?)
        self.tripped = False

        if ModelStates is None:
            rospy.logfatal('run_guard: gazebo_msgs is missing, so there is no '
                           'ground truth to guard on.  Refusing to pretend to '
                           'watch this run.')
            raise SystemExit(1)

        rospy.Subscriber(cmd_topic, Twist, self.cmd_cb)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.ms_cb)

        rospy.loginfo('run_guard: model=%s cmd=%s', self.model_name, cmd_topic)
        rospy.loginfo('run_guard: wedged if commanded (>%.2f m/s or >%.2f rad/s) '
                      'for >=%.0f%% of %.0fs while getting neither >%.2f m '
                      'from the window start nor >%.2f rad of heading',
                      self.cmd_min, self.ang_min, 100.0 * self.cmd_fraction,
                      self.stuck_time, self.progress_min,
                      self.yaw_progress_min)
        rospy.loginfo('run_guard: rolled if |roll|>%.0f or |pitch|>%.0f deg '
                      'held %.1fs', self.roll_limit, self.pitch_limit,
                      self.tip_time)
        if not self.trip_file:
            rospy.logwarn('run_guard: no ~trip_file set, so a trip will abort '
                          'the run but the campaign will not know to redo it')

    # ---------------- inputs ----------------
    def cmd_cb(self, msg):
        self.cmd_v = msg.linear.x
        self.cmd_w = msg.angular.z

    def ms_cb(self, msg):
        try:
            i = msg.name.index(self.model_name)
        except ValueError:
            rospy.logwarn_throttle(
                10.0, 'run_guard: model "%s" not in /gazebo/model_states; '
                'names=%s', self.model_name, msg.name)
            return
        p = msg.pose[i]
        r, pi, yw = roll_pitch_yaw(p.orientation)
        self.pose = (p.position.x, p.position.y, p.position.z, r, pi, yw)
        if self.first_seen is None:
            self.first_seen = rospy.get_time()

    # ---------------- decision ----------------
    def translating(self):
        return abs(self.cmd_v) > self.cmd_min

    def rotating(self):
        return abs(self.cmd_w) > self.ang_min

    def check_rolled(self, now):
        _, _, _, roll, pitch, _ = self.pose
        over = abs(roll) >= self.roll_limit or abs(pitch) >= self.pitch_limit
        if not over:
            self.tipped_since = None
            return None
        if self.tipped_since is None:
            self.tipped_since = now
            return None
        if now - self.tipped_since < self.tip_time:
            return None
        return ('rolled_over',
                'roll %.1f deg, pitch %.1f deg held %.1f s (limits %.0f / %.0f)'
                % (roll, pitch, now - self.tipped_since,
                   self.roll_limit, self.pitch_limit))

    def check_wedged(self, now):
        x, y, yaw = self.pose[0], self.pose[1], self.pose[5]
        self.track.append((now, x, y, yaw,
                           self.translating(), self.rotating()))
        while self.track and now - self.track[0][0] > self.stuck_time:
            self.track.popleft()

        if len(self.track) < 2:
            return None
        span = now - self.track[0][0]
        if span < self.stuck_time:
            return None

        # EXCURSION MAXIMA respecto del primer punto de la ventana, no suma
        # de arco ni desplazamiento neto.  El arco premia la sacudida (ver
        # progress_min).  El desplazamiento neto tiene el defecto opuesto:
        # daria cero para una vuelta completa, que es movimiento legitimo.
        # La excursion los evita a los dos - solo crece si el robot llega a
        # estar lejos de donde estaba, que es exactamente lo que un robot
        # encajado nunca consigue.
        _, x0, y0, yaw0, _, _ = self.track[0]
        moved = max(math.hypot(bx - x0, by - y0)
                    for (_, bx, by, _, _, _) in self.track)
        # Excursion ANGULAR, por la misma razon que la lineal: un robot que
        # cabecea +/-5 deg no ha girado, aunque la suma de sus vaivenes sea
        # grande.  El envolvimiento la limita a pi, y el umbral esta muy
        # por debajo, asi que no estorba.
        turned = max(abs(wrap(byaw - yaw0))
                     for (_, _, _, byaw, _, _) in self.track)

        n = float(len(self.track))
        asked_trans = sum(1 for s in self.track if s[4]) / n >= self.cmd_fraction
        asked_rot = sum(1 for s in self.track if s[5]) / n >= self.cmd_fraction
        frac = sum(1 for s in self.track if s[4] or s[5]) / n

        # Encajado = se le pidio moverse de ALGUNA forma durante casi toda
        # la ventana y no consiguio NINGUNA de las dos.  Basta con que una
        # prospere para que no sea un encaje: girar en el sitio es avance
        # legitimo aunque la posicion no cambie, y avanzar en linea recta lo
        # es aunque el rumbo no cambie.
        if ((asked_trans or asked_rot)
                and moved < self.progress_min
                and turned < self.yaw_progress_min):
            return ('wedged',
                    'commanded for %.0f%% of %.1f s (linear=%s angular=%s) '
                    'but never got further than %.3f m from (%.2f, %.2f) '
                    'nor turned more than %.3f rad; last command %.2f m/s, '
                    '%.2f rad/s'
                    % (100.0 * frac, span, asked_trans, asked_rot, moved,
                       x0, y0, turned, self.cmd_v, self.cmd_w))
        return None

    def trip(self, reason, detail):
        self.tripped = True
        # El aviso PRIMERO.  Estaba despues del desempaquetado de abajo, que
        # pedia 5 valores de una pose que tiene 6 (x, y, z, roll, pitch, yaw -
        # ver linea 148), asi que trip() lanzaba ValueError antes de emitir
        # nada.  Como el vigia va con required="true", roslaunch mataba la
        # corrida igual, pero SIN el mensaje y SIN guard_trip.yaml: la campaña
        # no veia ningun aborto, no reintentaba, y archivaba como buena una
        # corrida truncada.  Es justo la trampa que describe run_campaign.sh
        # en su comprobacion de guard_trip.yaml.  MEDIDO 2026-08-28: el Husky
        # quedo en 144.8 m de una ruta de 288.1 m y se guardo como valida.
        rospy.logerr('run_guard: TRIP (%s) - %s', reason, detail)
        x, y, z, roll, pitch = self.pose[:5]
        if self.trip_file:
            try:
                d = os.path.dirname(self.trip_file)
                if d and not os.path.isdir(d):
                    os.makedirs(d)
                with open(self.trip_file, 'w') as f:
                    f.write('tripped: true\n')
                    f.write('reason: %s\n' % reason)
                    f.write('detail: "%s"\n' % detail.replace('"', "'"))
                    f.write('model: %s\n' % self.model_name)
                    f.write('sim_time_s: %.2f\n' % rospy.get_time())
                    f.write('pose: {x: %.3f, y: %.3f, z: %.3f, '
                            'roll_deg: %.2f, pitch_deg: %.2f}\n'
                            % (x, y, z, roll, pitch))
            except (IOError, OSError) as exc:
                rospy.logerr('run_guard: could not write %s: %s',
                             self.trip_file, exc)
        rospy.signal_shutdown('run_guard: %s' % reason)

    def step(self):
        if self.tripped or self.pose is None or self.first_seen is None:
            return
        now = rospy.get_time()
        if now - self.first_seen < self.grace:
            return

        hit = self.check_rolled(now)
        if hit is None:
            hit = self.check_wedged(now)
        if hit is not None:
            self.trip(*hit)

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self.step()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        if not self.tripped:
            rospy.loginfo('run_guard: no trip; the run ended on its own terms')


if __name__ == '__main__':
    rospy.init_node('run_guard')
    try:
        RunGuard().run()
    except rospy.ROSInterruptException:
        pass
