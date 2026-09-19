#!/usr/bin/env python3
"""Drive a fixed path using ground truth, so the SLAM is an observer.

WHY
===
With move_base the robot steers on its own SLAM estimate.  Measured over this
route, that estimate drifts 0.6-0.8 m, and the drift does not merely add error
to the log - it changes WHERE THE ROBOT GOES.  The rocker_bogie crossed the
0.10 m step at x = 54.35; the differential, 0.4-0.9 m further west, wedged at
x = 53.41 nose-up 48 deg on ground the rocker never touched.  Two platforms
meeting different terrain are not being compared.

Under this node the platform follows one fixed path taken from the
rocker_bogie run that closed the whole loop.  Every platform then meets exactly
the same geometry, and the SLAM estimate is recorded without being able to
influence the outcome - which is what makes its error a clean measurement.

WHAT THIS IS AND IS NOT
=======================
It is NOT autonomy: ground truth is not available to a real robot, and this
experiment cannot answer "which platform can navigate the mine".  That question
needs the move_base runs.  This one answers "given the same path, how does each
platform's SLAM perform and how does its chassis behave".  Report them as two
experiments, not one.

CONTROL
=======
Pure pursuit.  Find the nearest point on the path, look ahead a fixed distance,
steer at the curvature that reaches it.  Speed is capped by the same limits the
planner uses so the platforms are still driven alike, and is reduced on tight
curvature and while the heading error is large.
"""

import collections
import math
import os

import rospy
import yaml
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def fuera_de_rumbo(th, pts, i, umbral):
    """True si el rumbo del robot se aparta mas de `umbral` del TRAMO i de la ruta.

    Es lo que decide el arco de alineacion.  Antes se miraba el rumbo al punto
    objetivo, y con el lookahead corto (0.30 + 0.8 v) ese rumbo pasa de 0.8 rad
    en cuanto el error lateral supera el lookahead aunque el robot vaya paralelo
    a la ruta.  Medido el 2026-09-14 (~/metrics_final_version): en la bajada de
    14 deg de s = 95-110 m el Husky iba a 6 deg de la ruta y 1.3 m al lado, el
    arco le fijaba v = 0.25, las ruedas frenaban, patinaban (desliz -0.29) y el
    chasis giraba el 5 % de lo que giraban las ruedas: sin giro no bajaba el
    error, y sin bajar el error no salia del arco.  El error lateral ya lo
    corrige k_xte; el arco queda para cuando el RUMBO esta mal de verdad.
    """
    ax, ay = pts[i]
    bx, by = pts[min(i + 1, len(pts) - 1)]
    if (bx - ax) ** 2 + (by - ay) ** 2 < 1e-12:
        return False
    return abs(wrap(th - math.atan2(by - ay, bx - ax))) > umbral


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class Follower(object):

    def __init__(self):
        path_file = rospy.get_param('~path_file')
        self.model = rospy.get_param('~model_name')
        cmd_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel')

        # Same limits the planner is given, so the platforms are still driven
        # under equal conditions (see planner: in sim_config.yaml).
        self.v_max = rospy.get_param('~max_vel_x', 0.5)
        self.w_max = rospy.get_param('~max_vel_theta', 1.0)
        self.goal_tol = rospy.get_param('~goal_tolerance', 0.5)  # m
        # CIERRE DEL LAZO.  Hay que alcanzar el ultimo punto de verdad, y con
        # una tolerancia propia mas apretada que la de navegacion: con la de
        # goal_tolerance el robot podia quedarse a medio metro del final y la
        # vuelta cerraba abierta.
        self.close_tol = rospy.get_param('~close_tolerance', 0.15)   # m
        # Salvavidas: si el robot ronda el final sin entrar en close_tol, la
        # corrida acabaria en el tope de --duration y se perderia entera.  Se
        # cierra igual y se DEJA ESCRITO el hueco que quedo, que es el dato.
        #
        # 15 s, POR DEBAJO de los 25 s de stuck_time del vigia: por encima no
        # llega a actuar nunca -el vigia aborta primero y la corrida se pierde
        # entera en vez de cerrarse con su hueco anotado-.
        self.close_timeout = rospy.get_param('~close_timeout', 15.0)  # s
        self._close_since = None
        # Hueco con el que se cerro cada vuelta, para el informe final.
        self.cierre_m = float('nan')
        # Slow down when the path bends or the heading is wrong.
        self.k_curve = rospy.get_param('~curve_slowdown', 0.6)
        self.align_thresh = rospy.get_param('~align_threshold', 0.8)  # rad
        # ---- correccion de rumbo SIN girar sobre el sitio -----------------
        # Medido sobre el banco de rampa cruzada, con la misma actitud de
        # partida y el mismo terreno:
        #
        #     arco (v 0.25, w 0.35)          5.59 ruedas de 6
        #     giro sobre el sitio (w 0.50)   3.55 ruedas, rev_3 barriendo
        #                                    66.1 de los 68.8 deg disponibles
        #
        # y en suelo plano, sin terreno de por medio, el circulo de R=2.0 y el
        # de R=1.0 mantienen las seis ruedas mientras que el de R=0.5 pierde
        # una y solo consigue el 77 % de la guiñada pedida.  Lo que rompe el
        # apoyo no es girar: es girar con radio pequeño o nulo.  Y no se
        # arregla yendo despacio, porque la fuerza que enrolla el bogie la fija
        # el rozamiento -mu*N- y no la velocidad: el mismo giro a 0.125 y a
        # 0.5 rad/s enrolla igual.
        #
        # Asi que en vez de frenar en seco y mandar w_max -que es justo el
        # comando que en el metro 115 de la ruta se emite con el robot
        # encaramado a la arista- se corrige el rumbo describiendo un arco de
        # radio acotado.  Cuesta espacio de maniobra: hay que comprobar que la
        # ruta lo tiene.
        #
        # align_min_radius = 0.0 restaura el giro sobre el sitio de antes.
        self.align_min_radius = rospy.get_param('~align_min_radius', 1.0)  # m
        self.align_v = rospy.get_param('~align_v', 0.25)  # m/s durante el arco
        # Por debajo de esta distancia el vector al objetivo es ruido y no se
        # apunta a el; ver el bloque de control.  0.25 m es holgado frente al
        # ruido de /gazebo/model_states y corto frente al lookahead minimo
        # (0.60 m), asi que solo entra en juego al final del recorrido.
        self.min_aim_dist = rospy.get_param('~min_aim_dist', 0.25)   # m
        self.reverse_ok = rospy.get_param('~allow_reverse', False)

        # ---- tracking ----------------------------------------------------
        # The error of a naive pure pursuit concentrates in the corners - it
        # cuts them - and that has two structural causes, not tuning ones:
        #
        #  * Pure pursuit aims at a point a fixed distance ahead ON the path
        #    and steers at the curvature that reaches it.  On a curve that
        #    leaves a steady-state offset: the vehicle settles INSIDE the arc
        #    and stays there, because the geometry is satisfied there.  The
        #    cross-track error has to enter the control law to close it, which
        #    is what ~k_xte does.
        #
        #  * A fixed lookahead cannot suit both cases.  At 1.0 m it was long
        #    for a 2-3 m radius corner (the chord cuts) and short for a
        #    straight (the vehicle weaves at it).  It now scales with the
        #    speed the vehicle is actually being given, which is short in the
        #    corners - where it slows down - and long on the straights.
        #
        # Set k_xte to 0.0 and lookahead_min = lookahead_max = 1.0 for a plain
        # fixed-lookahead pure pursuit.
        #
        # EL LOOKAHEAD ES LA PALANCA, y hacia ABAJO.  Banco frio
        # (test/banco_seguidor3.py): planta de guiñada identificada de las
        # corridas -ganancia en permanente y constante de tiempo- y lazo
        # cerrado contra la ruta de referencia real.  Error lateral medio:
        #
        #     L = min + k*v (tope)   differential   tracked   rocker
        #     0.60 + 1.6 v (1.6)        0.103        0.025     0.054
        #     0.40 + 1.0 v (1.2)        0.054        0.017     0.023
        #     0.30 + 0.8 v (1.0)        0.041        0.012     0.017   <- este
        #     0.25 + 0.6 v (0.9)        0.026        0.009     0.014
        #     0.20 + 0.5 v (0.8)        0.024        0.006     0.017
        #
        # Es la palanca que ordena el seguimiento y MEJORA LAS TRES, asi que no
        # hay que elegir plataforma ni meter una constante por robot: sigue
        # siendo el mismo controlador en las tres, que es de lo que vive la
        # comparacion.
        #
        # POR QUE FUNCIONA.  Pure pursuit corta la curva con un error de
        # asiento que va como L^2/(8R).  Las curvas de la galeria tienen R de
        # 2-3 m y el lookahead efectivo era 1.26 m a la velocidad de crucero:
        # la cuerda se comia el arco.  Acortarlo ataca la causa geometrica; el
        # termino k_xte solo curaba el sintoma, y en la plataforma con menos
        # autoridad de guiñada no le llegaba.
        #
        # SE PARA EN 0.30 Y NO EN 0.20 aunque el banco siga bajando: entre
        # 0.25 y 0.20 la oscilacion despega -cruces de signo de la curvatura
        # por metro: tracked 0.33 -> 0.58, rocker 0.46 -> 0.74- y el banco no
        # tiene terreno irregular ni suspension, asi que su margen de
        # oscilacion es optimista.  0.30 deja el error del differential en la
        # liga en la que estaban los otros dos y no toca la oscilacion
        # (0.22 -> 0.24).
        #
        # LO QUE SE PROBO Y NO SIRVE: accion integral sobre el error lateral.
        # Barrida k_i de 0.1 a 1.2 en test/banco_seguidor.py, EMPEORA LAS TRES
        # de forma monotona (differential 0.103 -> 0.174 con k_i 1.2).  El
        # error de esta ruta no es un sesgo constante que una integral pueda
        # anular: son curvas que alternan de signo, y la integral solo mete
        # retardo de fase en un lazo que ya va justo.
        #
        # La velocidad casi no se mueve, que era el riesgo: el lookahead corto
        # pide mas curvatura y curve_slowdown frena con ella.  Medido en el
        # banco, la horquilla entre plataformas se ESTRECHA del 11 % al 8 %,
        # que va en la direccion buena para las nubes/m.
        self.lookahead = rospy.get_param('~lookahead', 1.0)     # m, fallback
        self.la_min = rospy.get_param('~lookahead_min', 0.30)   # m
        self.la_max = rospy.get_param('~lookahead_max', 1.0)    # m
        self.la_k = rospy.get_param('~lookahead_k', 0.8)        # s (L = min + k*v)
        # Curvature added per metre of cross-track error, 1/m^2.  Pure pursuit
        # supplies the damping - it is already looking ahead - so this stays
        # small; it is clamped below so a transient cannot saturate the turn.
        # 2.5, medido: barrido en frio contra las curvas alfa(w) de cada
        # plataforma, arrancando 0.90 m fuera -un desvio real, no uno de
        # juguete-.  Metros hasta volver a +-10 cm:
        #
        #         k_xte      differential   tracked   rocker
        #         0.8            25.9 m      8.6 m    3.4 m
        #         2.5            10.9 m      6.4 m    2.8 m
        #
        # Mejora las tres y la sobreoscilacion BAJA en todas, asi que no hay
        # que elegir entre plataformas.
        #
        # xte_curv_max se queda en 0.5 A PROPOSITO.  Subirlo empeora: con
        # k_xte 2.5, pasar el tope a 2.0 sube el recorrido de 10.9 a 12.6 m y
        # la sobreoscilacion de 0.30 a 0.43.  El clamp amortigua, no estorba.
        self.k_xte = rospy.get_param('~k_xte', 2.5)
        self.xte_curv_max = rospy.get_param('~xte_curv_max', 0.5)   # 1/m
        # w = curv * v is the arc the vehicle is asked to trace.  A floor much
        # above zero over-rotates whenever the curve slowdown takes v below it,
        # i.e. exactly in the corners; it is kept small so the vehicle still
        # turns when nearly stopped, and exposed so it can be checked.
        self.w_v_floor = rospy.get_param('~w_v_floor', 0.10)
        self.last_v = 0.0

        # STALL GUARD.  Thresholds deliberately identical to rearing_guard.py's
        # (see robot_metrics/launch/rearing_guard.launch), because that node
        # protects the move_base runs and this one protects the fixed-trajectory
        # runs: different thresholds would mean the two experiments stop the
        # vehicle at different points, and the comparison would be measuring the
        # guard rather than the locomotion.
        #
        # Why this exists.  rearing_guard cancels a move_base GOAL, and there
        # is no move_base here, so nothing else is watching.  A wedged vehicle
        # whose drive keeps pushing loads the suspension against its joint
        # stops until the constraint releases explosively: measured on the step
        # descent, 12.35 rad/s (708 deg/s) and 190.9 rad/s^2 of roll against
        # 1.09 rad/s and 15.6 rad/s^2 for the rest of that run, and the vehicle
        # ended inverted.  The stall is the precondition: back off before the
        # energy accumulates and the release does not happen.
        self.commanded_min = rospy.get_param('~commanded_min', 0.15)
        self.speed_max = rospy.get_param('~speed_max', 0.05)
        self.stall_time = rospy.get_param('~stall_time', 1.0)
        self.backoff_speed = rospy.get_param('~backoff_speed', -0.25)
        self.backoff_time = rospy.get_param('~backoff_time', 2.0)
        self.cooldown = rospy.get_param('~cooldown', 5.0)
        # A vehicle that stalls this many times is not going to finish, and
        # letting it run out --duration wastes an hour of wall clock and
        # produces a run nobody can use.  0 disables the abort.
        self.max_stalls = rospy.get_param('~max_stalls', 8)
        # Donde avisar que la corrida se aborto.  Lo pone run_campaign.sh.
        # Sin esto el abort es INVISIBLE para la campana: metrics_logger
        # alcanza a escribir un metrics.csv perfectamente valido de los metros
        # que se hicieron y la corrida queda en disco marcada como buena, lista
        # para que aggregate_runs.py la promedie con las completas.
        self.trip_file = rospy.get_param('~trip_file', '')

        with open(path_file) as fh:
            doc = yaml.safe_load(fh)
        self.pts = [(p['x'], p['y']) for p in doc['points']]
        if len(self.pts) < 2:
            raise RuntimeError('reference path has %d points' % len(self.pts))
        rospy.loginfo('trajectory_follower: %d points, %.2f m, from %s',
                      len(self.pts), doc.get('path_length_m', float('nan')),
                      path_file)
        rospy.loginfo('trajectory_follower: model=%s cmd=%s v<=%.2f w<=%.2f '
                      'lookahead=%.2f', self.model, cmd_topic, self.v_max,
                      self.w_max, self.lookahead)

        self.pose = None
        self.idx = 0
        self.done = False
        # Vueltas a la ruta.  El recorrido es un lazo cerrado, asi que
        # repetirlo acumula deriva sobre LA MISMA geometria: no inventa
        # terreno nuevo ni cambia la ruta que comparten las tres plataformas.
        self.laps = max(1, int(rospy.get_param('~laps', 1)))
        self.lap = 0
        self.max_xte = 0.0
        self.sum_xte = 0.0
        self.n_xte = 0

        # stall-guard state.  A window of (t, x, y), not a velocity: see
        # window_motion() for why Gazebo's reported twist cannot be used.
        self.track = collections.deque()
        self.cmd_v = 0.0
        self.backoff_until = 0.0
        self.cooldown_until = 0.0
        self.stall_count = 0

        # The reference path is in WORLD coordinates; the map frame is anchored
        # at the spawn pose.  Publishing world coordinates tagged "map" would
        # draw the path 33.5 m away from the robot in RViz, so transform it.
        self.spawn_x = rospy.get_param('~spawn_x', 33.5)
        self.spawn_y = rospy.get_param('~spawn_y', 0.2)
        self.spawn_yaw = rospy.get_param('~spawn_yaw', 0.0)
        # Rumbo fisico = yaw del modelo + este offset.  El URDF del
        # Rocker-Bogie tiene base_link mirando a -x, asi que su modelo
        # reporta yaw = pi cuando el robot apunta a 0.  Sin corregirlo,
        # el control ve un error de rumbo de 180 grados que no existe y
        # gira en el lugar para siempre.  NO es lo mismo que spawn_yaw
        # (donde se puso el robot, usado solo para dibujar el path).
        self.base_yaw_offset = rospy.get_param('~base_yaw_offset', 0.0)
        self.viz_frame = rospy.get_param('~viz_frame', 'map')

        self.cmd = rospy.Publisher(cmd_topic, Twist, queue_size=1)
        self.viz = rospy.Publisher('~reference_path', Path, queue_size=1, latch=True)
        # Ground truth, drawn in the same frame as the estimate.  Watching the
        # two separate IS the experiment: the reference and the truth stay
        # together because the controller uses truth, so everything the SLAM
        # estimate does on its own is its own error.
        self.truth_pub = rospy.Publisher('~ground_truth_path', Path, queue_size=1)
        self.truth = Path()
        self.truth.header.frame_id = self.viz_frame
        self._last_truth = None

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.ms_cb)
        self.publish_path()
        rospy.on_shutdown(self.report)

    def world_to_viz(self, x, y):
        """World -> the frame RViz is showing (map, anchored at the spawn).

        El giro es spawn_yaw + base_yaw_offset, no spawn_yaw.  `map` lo ancla
        rtabmap a su frame_id, que en el Rocker-Bogie es base_link_nav --
        base_link girado 180 deg-- y no base_link.  Con spawn_yaw a secas la
        ruta de referencia y el ~ground_truth_path salian en RViz girados 180
        deg respecto de la estimada, que es justo la comparacion que el panel
        existe para ensenar.
        """
        dx, dy = x - self.spawn_x, y - self.spawn_y
        a = -(self.spawn_yaw + self.base_yaw_offset)
        c, s = math.cos(a), math.sin(a)
        return c * dx - s * dy, s * dx + c * dy

    def publish_path(self):
        m = Path()
        m.header.frame_id = self.viz_frame
        for x, y in self.pts:
            vx, vy = self.world_to_viz(x, y)
            ps = PoseStamped()
            ps.header.frame_id = self.viz_frame
            ps.pose.position.x = vx
            ps.pose.position.y = vy
            ps.pose.orientation.w = 1.0
            m.poses.append(ps)
        self.viz.publish(m)
        rospy.loginfo('trajectory_follower: reference path published on '
                      '~reference_path in frame "%s" (%d poses)',
                      self.viz_frame, len(m.poses))

    def publish_truth(self, x, y):
        if (self._last_truth is not None
                and math.hypot(x - self._last_truth[0],
                               y - self._last_truth[1]) < 0.10):
            return
        self._last_truth = (x, y)
        vx, vy = self.world_to_viz(x, y)
        ps = PoseStamped()
        ps.header.frame_id = self.viz_frame
        ps.header.stamp = rospy.Time.now()
        ps.pose.position.x = vx
        ps.pose.position.y = vy
        ps.pose.orientation.w = 1.0
        self.truth.poses.append(ps)
        self.truth.header.stamp = ps.header.stamp
        self.truth_pub.publish(self.truth)

    def ms_cb(self, msg):
        try:
            i = msg.name.index(self.model)
        except ValueError:
            rospy.logwarn_throttle(
                10.0, 'trajectory_follower: model "%s" not in /gazebo/model_states;'
                      ' names=%s', self.model, list(msg.name))
            return
        p = msg.pose[i]
        self.pose = (p.position.x, p.position.y,
                     wrap(yaw_of(p.orientation) + self.base_yaw_offset))
        now = rospy.Time.now().to_sec()
        self.track.append((now, p.position.x, p.position.y))
        while self.track and now - self.track[0][0] > self.stall_time:
            self.track.popleft()
        self.publish_truth(p.position.x, p.position.y)

    # ---------------- control ----------------
    def nearest(self, x, y):
        """Closest point ON the path, searched forward only so the robot cannot
        be captured by an earlier part of the loop when it comes back round
        near the start.

        Returns (index, signed cross-track error).  The distance is to the
        nearest SEGMENT, not the nearest vertex: the path is stored at 0.25 m
        spacing, so measuring to vertices quantised the error by up to 0.125 m
        - an eighth of the mean that run01 reported.

        The sign is positive when the robot is to the LEFT of the direction of
        travel, which is the sign convention the curvature correction in step()
        needs; the magnitude alone is what goes into the report."""
        best_i, best_d2, best_sign = self.idx, 1e18, 0.0
        hi = min(len(self.pts) - 1, self.idx + 200)
        for i in range(self.idx, max(self.idx + 1, hi)):
            ax, ay = self.pts[i]
            bx, by = self.pts[i + 1]
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            if L2 < 1e-12:
                t = 0.0
            else:
                t = ((x - ax) * dx + (y - ay) * dy) / L2
                t = max(0.0, min(1.0, t))
            cx, cy = ax + t * dx, ay + t * dy
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
                # z of (segment direction) x (vehicle offset): >0 means the
                # vehicle is to the left of the path.
                best_sign = 1.0 if (dx * (y - ay) - dy * (x - ax)) >= 0.0 else -1.0
        return best_i, best_sign * math.sqrt(best_d2)

    def lookahead_now(self):
        """Lookahead for the speed the vehicle is currently being given.

        Short when it is slow, which is when it is in a corner, so the chord to
        the target cuts less of the curve; long on the straights, where a short
        lookahead makes a pure-pursuit vehicle weave."""
        return max(self.la_min,
                   min(self.la_max, self.la_min + self.la_k * abs(self.last_v)))

    def target(self, i, x, y, lookahead=None):
        if lookahead is None:
            lookahead = self.lookahead
        acc = 0.0
        j = i
        while j + 1 < len(self.pts) and acc < lookahead:
            acc += math.hypot(self.pts[j + 1][0] - self.pts[j][0],
                              self.pts[j + 1][1] - self.pts[j][1])
            j += 1
        return j

    def window_motion(self):
        """How far the vehicle actually got, over the guard's window.

        Displacement, deliberately, NOT the twist on /gazebo/model_states.
        Measured 2026-08-23 on the differential with a wheel sunk into the
        terrain: Gazebo reported a steady 0.0737 m/s while the model moved
        0.000 m in 30 s of sim.  The contact solver publishes a velocity for a
        vehicle that is going nowhere.  The guard had been comparing that
        number against speed_max, so it never fired, and the run sat in place
        commanding 0.443 m/s until the duration cap would have killed it.
        Displacement over a window answers the question the guard is actually
        asking - did it get anywhere - and over a window this long the
        differentiation noise that the twist was meant to avoid is nothing.
        """
        if len(self.track) < 2:
            return 0.0, 0.0
        t0, x0, y0 = self.track[0]
        t1, x1, y1 = self.track[-1]
        return math.hypot(x1 - x0, y1 - y0), t1 - t0

    def stall_guard(self):
        """True when the guard is driving, so step() must not also command.

        Commanded but not moving -> reverse briefly to put the wheels back
        down, then let the pursuit resume.  See the note in __init__.
        """
        now = rospy.Time.now().to_sec()

        if now < self.backoff_until:
            tw = Twist()
            tw.linear.x = self.backoff_speed
            self.cmd.publish(tw)
            return True

        moved, span = self.window_motion()
        if span < self.stall_time:
            return False          # not enough history yet
        speed = moved / span
        if self.cmd_v < self.commanded_min or speed > self.speed_max:
            return False
        if now < self.cooldown_until:
            return False

        self.stall_count += 1
        x, y = (self.pose[0], self.pose[1]) if self.pose else (float('nan'),) * 2
        rospy.logwarn('trajectory_follower: STALL %d at (%.2f, %.2f): '
                      'commanded %.2f m/s but moved %.3f m in %.1f s '
                      '(%.4f m/s); backing off %.2f m/s for %.1f s',
                      self.stall_count, x, y, self.cmd_v, moved, span, speed,
                      self.backoff_speed, self.backoff_time)
        self.backoff_until = now + self.backoff_time
        self.cooldown_until = now + self.backoff_time + self.cooldown

        if self.max_stalls and self.stall_count >= self.max_stalls:
            rospy.logerr('trajectory_follower: %d stalls, giving up at '
                         '(%.2f, %.2f) after %.1f m of %.1f m. Shutting the '
                         'run down rather than running out the clock.',
                         self.stall_count, x, y,
                         self.lap * 137.31, self.laps * 137.31)
            self.cmd.publish(Twist())
            self.write_trip(
                'stalled_out',
                '%d stalls; reached point %d of %d on lap %d of %d, %.1f m '
                'of %.1f m, stopped at (%.2f, %.2f)'
                % (self.stall_count, self.idx, len(self.pts), self.lap,
                   self.laps, self.lap * 137.31, self.laps * 137.31, x, y),
                x, y)
            rospy.signal_shutdown('stalled %d times' % self.stall_count)
            return True
        return True

    def write_trip(self, reason, detail, x, y):
        """Dejar constancia de que esta corrida se aborto, para que
        run_campaign.sh no la cuente como buena.  Mismo formato y mismo
        archivo que run_guard.py: quien lo lee no necesita saber cual de
        los dos lo escribio, solo que la corrida no vale."""
        if not self.trip_file:
            rospy.logwarn('trajectory_follower: sin ~trip_file, la campana '
                          'no se va a enterar de que esta corrida se aborto')
            return
        try:
            d = os.path.dirname(self.trip_file)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            with open(self.trip_file, 'w') as fh:
                fh.write('tripped: true\n')
                fh.write('reason: %s\n' % reason)
                fh.write('detail: "%s"\n' % detail.replace('"', "'"))
                fh.write('declared_by: trajectory_follower\n')
                fh.write('sim_time_s: %.2f\n' % rospy.get_time())
                fh.write('pose: {x: %.3f, y: %.3f}\n' % (x, y))
        except (IOError, OSError) as exc:
            rospy.logerr('trajectory_follower: no pude escribir %s: %s',
                         self.trip_file, exc)

    def step(self):
        if self.pose is None or self.done:
            return
        if self.stall_guard():
            return
        x, y, th = self.pose
        i, xte_signed = self.nearest(x, y)
        xte = abs(xte_signed)
        self.idx = i
        self.max_xte = max(self.max_xte, xte)
        self.sum_xte += xte
        self.n_xte += 1

        # finished?
        ex, ey = self.pts[-1]
        d_fin = math.hypot(ex - x, ey - y)
        # EL CIERRE SE DECIDE POR DISTANCIA, NO POR INDICE.  nearest()
        # devuelve el punto MAS CERCANO, y sobre el final del lazo -que
        # coincide con el inicio- ese puede seguir siendo el 543 de 545: con
        # el indice como condicion, un robot a milimetros del final no cierra
        # la vuelta nunca.
        #
        # El indice se queda solo como CANDADO: impide cerrar al arrancar,
        # cuando el robot esta sobre el punto inicial y por tanto tambien
        # sobre el final.  Basta con exigir que se haya recorrido casi todo.
        cerca_del_final = i >= len(self.pts) - 10
        en_el_final = cerca_del_final and d_fin < self.close_tol
        # El cronometro se arma con el MISMO candado que el cierre.  Si se
        # armase con una condicion mas estricta y el indice no llegase nunca
        # ahi, el salvavidas no saltaria y la corrida colgaria hasta el tope
        # de --duration.
        if cerca_del_final and self._close_since is None:
            self._close_since = rospy.get_time()
        rendido = (self._close_since is not None
                   and rospy.get_time() - self._close_since > self.close_timeout)
        if rendido:
            rospy.logwarn('trajectory_follower: %.1f s rondando el final sin '
                          'entrar en close_tolerance=%.2f m; cierro con el '
                          'lazo abierto %.3f m',
                          self.close_timeout, self.close_tol, d_fin)
        if en_el_final or rendido:
            self.cierre_m = d_fin
            self.lap += 1
            if self.lap < self.laps:
                # nearest() busca SOLO hacia adelante desde self.idx, que es
                # justo lo que impide que el robot sea capturado por un tramo
                # anterior al pasar cerca del inicio.  Por eso la vuelta nueva
                # necesita reiniciar el indice a mano.
                self.idx = 0
                self._close_since = None
                rospy.loginfo('trajectory_follower: vuelta %d de %d completa, '
                              'lazo cerrado a %.3f m', self.lap, self.laps,
                              d_fin)
                return
            self.done = True
            self.cmd.publish(Twist())
            rospy.loginfo('trajectory_follower: path complete (%d vuelta(s)); '
                          'lazo cerrado a %.3f m del punto inicial',
                          self.laps, d_fin)
            # Cerrar la corrida en vez de esperar el tope de --duration.  El
            # nodo esta marcado required="true", asi que esto apaga el launch
            # entero y el logger escribe metrics.csv en su shutdown hook.
            rospy.signal_shutdown('path complete')
            return

        j = self.target(i, x, y, self.lookahead_now())
        tx, ty = self.pts[j]
        # target in the robot frame
        dx, dy = tx - x, ty - y
        lx = math.cos(th) * dx + math.sin(th) * dy
        ly = -math.sin(th) * dx + math.cos(th) * dy
        L2 = lx * lx + ly * ly
        # DISTANCIA MINIMA AL OBJETIVO.  atan2 sobre un vector de milimetros
        # no da un rumbo, da ruido de posicion: a 4 mm del objetivo el rumbo
        # salta entre -164 y +172 grados de una muestra a otra, y cada vez que
        # ese valor aleatorio pasa de align_threshold la regla de girar en el
        # sitio frena en seco y manda +-1.0 rad/s.  Y la curvatura de pure
        # pursuit, 2*ly/L2, explota igual con L2 diminuto.
        # Por debajo de min_aim_dist no se apunta: ni curvatura ni giro en el
        # sitio.  El termino de error lateral sigue actuando, que es el que
        # tiene sentido cuando ya se esta encima del punto.
        apuntando = L2 > self.min_aim_dist * self.min_aim_dist
        curv = 2.0 * ly / L2 if apuntando else 0.0

        # Cross-track feedback.  Positive xte_signed means the vehicle is left
        # of the path, so it has to steer right, so the curvature - positive
        # being a left turn - has to come down.  Clamped so a transient cannot
        # take the whole turn; pure pursuit above supplies the damping.
        curv_e = -self.k_xte * xte_signed
        curv_e = max(-self.xte_curv_max, min(self.xte_curv_max, curv_e))
        curv += curv_e

        head_err_signed = wrap(math.atan2(dy, dx) - th)
        v = self.v_max / (1.0 + self.k_curve * abs(curv))
        # El umbral se mide contra el tramo de la ruta, no contra el objetivo:
        # ver fuera_de_rumbo().  El sentido del arco sigue saliendo del objetivo.
        corrigiendo = apuntando and fuera_de_rumbo(th, self.pts, i, self.align_thresh)
        if corrigiendo and self.align_min_radius > 0.0:
            # demasiado desviado de rumbo para conducir hacia el objetivo:
            # se corrige con un arco de radio acotado, sin dejar de avanzar.
            v = min(self.align_v, self.v_max)
            w = math.copysign(v / self.align_min_radius, head_err_signed)
        elif corrigiendo:
            # comportamiento anterior: frenar y girar sobre el sitio
            v = 0.0
            w = math.copysign(self.w_max, head_err_signed)
        else:
            w = curv * max(v, self.w_v_floor)

        tw = Twist()
        tw.linear.x = max(-self.v_max, min(self.v_max, v))
        tw.angular.z = max(-self.w_max, min(self.w_max, w))
        # what the guard compares the ground speed against
        self.cmd_v = tw.linear.x
        self.last_v = tw.linear.x       # feeds lookahead_now() next cycle
        self.cmd.publish(tw)

    def report(self):
        self.cmd.publish(Twist())
        if self.n_xte:
            # lap, not lap + 1: self.lap counts COMPLETED laps.
            done = min(self.lap + (0 if self.done else 1), self.laps)
            rospy.loginfo('trajectory_follower: reached point %d of %d, '
                          'vuelta %d de %d; cross-track error mean %.3f m, '
                          'max %.3f m; %d stall(s)',
                          self.idx, len(self.pts), done, self.laps,
                          self.sum_xte / self.n_xte, self.max_xte,
                          self.stall_count)
            if self.cierre_m == self.cierre_m:      # no es nan
                rospy.loginfo('trajectory_follower: lazo cerrado a %.3f m '
                              '(close_tolerance %.2f m)',
                              self.cierre_m, self.close_tol)
            else:
                # La corrida no llego a cerrar ninguna vuelta -el vigia la
                # aborto, o se acabo el tiempo-.  Decirlo, en vez de imprimir
                # el nan que quedaba de la inicializacion.
                rospy.loginfo('trajectory_follower: el lazo NO llego a '
                              'cerrarse (close_tolerance %.2f m)',
                              self.close_tol)

    def run(self):
        rate = rospy.Rate(rospy.get_param('~rate', 20.0))
        while not rospy.is_shutdown():
            self.step()
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('trajectory_follower')
    try:
        Follower().run()
    except rospy.ROSInterruptException:
        pass
