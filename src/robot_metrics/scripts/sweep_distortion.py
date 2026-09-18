#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distorsion de barrido de un LiDAR giratorio, metida por formulas.

Entra  /velodyne_points_raw   la nube instantanea de 360 deg del VLP-16 normal
Sale   /velodyne_points       la misma nube con la distorsion de barrido dentro

Aguas abajo no cambia nada: rtabmap y move_base siguen leyendo
/velodyne_points, mismo frame, misma tasa, misma densidad de puntos.

POR QUE ESTO Y NO LA CUÑA GIRANDO
=================================
La cuña sobre un rotor depende de que Gazebo cumpla el update_rate del gpu_ray,
y no lo cumple: la cobertura real por vuelta sale 350-356 deg en vez de 360, y
cuanto mas cargada va la simulacion, menos cubre.  Eso convierte la distorsion
en una propiedad emergente del planificador, distinta por plataforma, que es
justo el confusor que no se puede controlar: medido, el creep de rumbo sigue al
deficit de cobertura a razon de ~0.020 deg/m por grado.

Deformando la nube entera la cobertura es 360.000 deg SIEMPRE, no hay
ensamblador, ni encoder, ni cola, ni esperas, ni descartes, ni costura que
precese, y la magnitud de la distorsion pasa a ser un parametro en vez de un
accidente.  Ademas sale mas barato: una nube a 10 Hz en vez de una cuña a
400 Hz.

EL MODELO
=========
Referencia: LIO-SAM, `imageProjection.cpp::deskewPoint()`, que CORRIGE con

    transBt = transStartInverse * transFinal
    p_ref   = transBt * p_meas

o sea  p_ref = T_start^-1 * T_i * p_meas,  con T_start la pose mundo del sensor
en el primer punto y T_i la pose en el instante del punto.  Generar la
distorsion es invertirlo:

    p_meas = T_i^-1 * T_start * p_ref                                      (1)

El instante de cada punto sale de su azimut, que es el modelo de columnas de
`undistortEgoMotion` de MATLAB:

    u   = ((dir * (phi - phi_cut)) mod 2pi) / 2pi        u en [0, 1)        (2)
    t_i = u * T_barrido

y las poses se interpolan entre extremos con velocidad lineal y angular
constante, que es el modelo de LOAM (Zhang & Singh 2014).  Se aplica por
SECTORES, no punto a punto: es la formulacion de paquetes del ICCV 2021 ("the
sweep is divided into N packets ... each packet is transformed to the
coordinate frame at which it was captured") y ademas es fiel, porque un VLP-16
real dispara en bloques de azimut y no de forma continua.

LOS DOS PARAMETROS QUE ANTES NO SE PODIAN TOCAR
===============================================
  ~cut_angle_deg  el azimut de la costura.  Fijo y absoluto, como el `cut_angle`
                  del driver de Velodyne, que corta por cruce de azimut y no por
                  tiempo.  Con la cuña la costura precesaba ~5.4 deg por vuelta.
  ~rotor_dir      el SENTIDO de giro, +1 o -1.  Invertirlo es el experimento de
                  quiralidad que discrimina si el sesgo lo mete el acoplamiento
                  rotor/avance: si el sesgo no cambia de signo, esa familia de
                  mecanismos queda descartada entera.  Antes exigia re-simular.

EL CANAL DE TIEMPO
==================
Sale un campo `time` float32 con el DESFASE respecto al stamp de cabecera, que
es lo que publica velodyne_pointcloud sobre un VLP-16 real y lo que esperan
LIO-SAM y rtabmap (`Odom/Deskewing`: "if input lidar has time channel, it will
be deskewed with a constant motion model").  rtabmap rechaza la nube si
encuentra MAS de un campo reservado, asi que si la entrada ya trae uno se
reutiliza en vez de añadir un segundo.

LO QUE ESTE MODELO NO REPRODUCE
===============================
Reusa geometria capturada en un instante, asi que no reproduce cambios de
oclusion ni objetos moviles DURANTE el barrido.  El mundo de la mina es
estatico y el sensor no ve el cuerpo del propio robot, asi que para esta
campaña es exacto; en un mundo dinamico no lo seria, y eso hay que decirlo en
el paper.
"""
import math
import threading
from collections import deque

import numpy as np

NPT = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
PF_FLOAT32 = 7
RESERVADOS = ('t', 'time', 'stamps', 'timestamp')


# --------------------------------------------------------------------- mates
# Estas cuatro no tocan ROS a proposito: son las que prueba el banco offline.
def quat_mat(q):
    """Matriz de rotacion 3x3 de un cuaternion (x, y, z, w)."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)]])


def slerp(q0, q1, u):
    """Interpolacion esferica entre dos cuaterniones (x, y, z, w)."""
    a = np.asarray(q0, dtype=float)
    b = np.asarray(q1, dtype=float)
    d = float(np.dot(a, b))
    if d < 0.0:                          # el camino corto
        b, d = -b, -d
    if d > 0.9995:                       # casi paralelos: lineal y normaliza
        r = a + u * (b - a)
        return r / np.linalg.norm(r)
    th = math.acos(max(-1.0, min(1.0, d)))
    s = math.sin(th)
    return (math.sin((1.0 - u) * th) / s) * a + (math.sin(u * th) / s) * b


def fase(x, y, cut, sentido):
    """(2): la fraccion del barrido en la que se midio cada punto, en [0, 1)."""
    return np.mod(sentido * (np.arctan2(y, x) - cut),
                  2.0 * math.pi) / (2.0 * math.pi)


def deforma(xyz, u, sectores, pose_de_u):
    """(1) por sectores: mete la distorsion de barrido en una nube instantanea.

    xyz        (N, 3) puntos en el frame del sensor en el instante de inicio
    u          (N,)   fraccion del barrido de cada punto, de fase()
    pose_de_u  callable(us) -> (R, x) pose MUNDO del sensor en la fraccion us,
               o None si no se sabe (ese sector se deja sin deformar)

    Devuelve (N, 3) los puntos como los habria medido un sensor que barre.
    """
    T0 = pose_de_u(0.0)
    if T0 is None:
        return xyz.copy()
    R0, x0 = T0
    k = np.minimum((u * sectores).astype(np.int64), sectores - 1)
    out = xyz.copy()
    for s in np.unique(k):
        sel = k == s
        Ti = pose_de_u((s + 0.5) / sectores)
        if Ti is None:
            continue
        Ri, xi = Ti
        # T_i^-1 * T_0 aplicado a p  ==  Ri^T * (R0 @ p + x0 - xi)
        out[sel] = (R0.dot(xyz[sel].T).T + (x0 - xi)).dot(Ri)
    return out


def deskew(xyz, u, sectores, pose_de_u):
    """La inversa de deforma(): la correccion de LIO-SAM, p_ref = T0^-1 Ti p.

    Existe para que el banco pueda cerrar el viaje de ida y vuelta.  El
    front-end de verdad (rtabmap con Odom/Deskewing) hace lo mismo, pero con su
    propio modelo de movimiento en vez de con la verdad-terreno.
    """
    T0 = pose_de_u(0.0)
    if T0 is None:
        return xyz.copy()
    R0, x0 = T0
    k = np.minimum((u * sectores).astype(np.int64), sectores - 1)
    out = xyz.copy()
    for s in np.unique(k):
        sel = k == s
        Ti = pose_de_u((s + 0.5) / sectores)
        if Ti is None:
            continue
        Ri, xi = Ti
        # T_0^-1 * T_i aplicado a p  ==  R0^T * (Ri @ p + xi - x0)
        out[sel] = (Ri.dot(xyz[sel].T).T + (xi - x0)).dot(R0)
    return out


# ---------------------------------------------------------------------- nodo
class Deformador(object):
    def __init__(self):
        import rospy
        import tf2_ros
        from gazebo_msgs.msg import ModelStates
        from sensor_msgs.msg import PointCloud2
        self.rospy, self.PointCloud2 = rospy, PointCloud2

        self.rev = 1.0 / float(rospy.get_param('~rotor_hz', 10.0))
        self.cut = math.radians(float(rospy.get_param('~cut_angle_deg', 0.0)))
        self.dir = 1.0 if float(rospy.get_param('~rotor_dir', 1.0)) >= 0 else -1.0
        self.sectores = int(rospy.get_param('~sectors', 360))
        self.model = rospy.get_param('~model_name', '')
        self.base = rospy.get_param('~base_frame', self.model + '/base_link')
        self.t_campo = rospy.get_param('~time_field', 'time')
        self.frame = rospy.get_param('~frame_id', '')

        self.lock = threading.Lock()
        self.poses = deque()             # (t, xyz, quat) del base, en mundo
        self.mount = None                # T_base<-sensor, constante
        self.dtype = self.dtype_out = self.campos = None
        self.n_in = self.n_out = self.n_sin_pose = self.n_sin_montaje = 0
        self.n_tarde = 0
        # Nubes esperando a que el historial de poses llegue al final de su
        # barrido.  La pose de t0+rev esta en el FUTURO cuando la nube llega,
        # asi que procesarla en el callback es imposible por construccion.
        self.pendientes = []
        # Cuanto se tolera esperar antes de soltarla sin deformar.  Cinco
        # barridos: si el historial no ha llegado en 0.5 s no va a llegar.
        self.espera_max = 5.0 * self.rev

        self.tfbuf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfbuf)

        self.pub = rospy.Publisher(
            rospy.get_param('~output', '/velodyne_points'), PointCloud2,
            queue_size=2)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.cb_gt,
                         queue_size=2000, tcp_nodelay=True)
        rospy.Subscriber(rospy.get_param('~input', '/velodyne_points_raw'),
                         PointCloud2, self.cb_nube, queue_size=8,
                         buff_size=2 ** 26, tcp_nodelay=True)
        # El drenaje va en un timer y no en el callback de ModelStates, que
        # entra a ~1000 Hz: deformar 28000 puntos ahi dentro lo atascaria.
        rospy.Timer(rospy.Duration(0.01), self.drena)
        rospy.on_shutdown(self.resumen)

    # ------------------------------------------------------------------ in
    def resumen(self):
        """Lo ultimo que se lee del nodo, y lo unico que hay que mirar.

        Una nube sin deformar no es un apaño: es OTRO experimento.  Sale la
        nube instantanea con un canal de tiempo pegado, o sea --lidar normal
        disfrazado, con ficheros de tamano normal y metricas buenisimas.  El
        2026-09-09 dos campañas enteras salieron asi y parecian un exito.
        """
        deformadas = self.n_out - self.n_sin_pose
        linea = ('sweep_distortion: entradas=%d deformadas=%d sin_deformar=%d '
                 'tarde=%d sin_montaje=%d' %
                 (self.n_in, deformadas, self.n_sin_pose, self.n_tarde,
                  self.n_sin_montaje))
        if self.n_sin_pose or self.n_sin_montaje or not deformadas:
            self.rospy.logerr('%s  <-- LA CORRIDA NO VALE COMO BARRIDO', linea)
        else:
            self.rospy.loginfo(linea)

    def cb_gt(self, msg):
        """ModelStates no trae stamp de cabecera: se sella al recibir, igual
        que metrics_logger.py.  El error queda acotado por un paso de fisica
        (0.5 ms), tres ordenes por debajo del barrido de 100 ms."""
        try:
            i = msg.name.index(self.model)
        except ValueError:
            self.rospy.logwarn_throttle(
                10.0, 'el modelo "%s" no esta en /gazebo/model_states; '
                'nombres=%s', self.model, list(msg.name))
            return
        p = msg.pose[i]
        t = self.rospy.Time.now().to_sec()
        with self.lock:
            self.poses.append((
                t,
                np.array([p.position.x, p.position.y, p.position.z]),
                np.array([p.orientation.x, p.orientation.y,
                          p.orientation.z, p.orientation.w])))
            # medio segundo de historia: cinco barridos, de sobra para
            # interpolar y poco para que la cola crezca sin limite
            while self.poses and t - self.poses[0][0] > 0.5:
                self.poses.popleft()

    # -------------------------------------------------------------- helpers
    def prepara(self, msg):
        """Igual que el generador anterior: reutiliza el canal de tiempo si
        ya viene, y si no lo añade al final alineado a 4 bytes."""
        from sensor_msgs.msg import PointField
        nombres, formatos, offsets = [], [], []
        for f in msg.fields:
            nombres.append(f.name)
            formatos.append(NPT[f.datatype])
            offsets.append(f.offset)
        self.dtype = np.dtype({'names': nombres, 'formats': formatos,
                               'offsets': offsets, 'itemsize': msg.point_step})
        ya = [n for n in RESERVADOS if n in nombres]
        if ya:
            self.t_campo = ya[0]
            self.dtype_out, self.campos = self.dtype, msg.fields
            self.point_step = msg.point_step
            self.rospy.loginfo(
                'la nube ya trae canal de tiempo "%s"; se reutiliza',
                self.t_campo)
            return
        self.point_step = msg.point_step + 4
        self.dtype_out = np.dtype({
            'names': nombres + [self.t_campo],
            'formats': formatos + ['f4'],
            'offsets': offsets + [msg.point_step],
            'itemsize': self.point_step})
        self.campos = list(msg.fields) + [
            PointField(name=self.t_campo, offset=msg.point_step,
                       datatype=PF_FLOAT32, count=1)]

    def pose_base(self, t):
        """Pose mundo del base en t, interpolada del historial."""
        with self.lock:
            P = list(self.poses)
        if len(P) < 2 or t < P[0][0] or t > P[-1][0]:
            return None
        lo, hi = 0, len(P) - 1
        while hi - lo > 1:
            m = (lo + hi) // 2
            if P[m][0] <= t:
                lo = m
            else:
                hi = m
        t0, x0, q0 = P[lo]
        t1, x1, q1 = P[hi]
        u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return x0 + u * (x1 - x0), slerp(q0, q1, u)

    def pose_sensor(self, t):
        """T_mundo<-sensor(t) = T_mundo<-base(t) * T_base<-sensor."""
        pb = self.pose_base(t)
        if pb is None or self.mount is None:
            return None
        x, q = pb
        R = quat_mat(q)
        Rm, xm = self.mount
        return R.dot(Rm), R.dot(xm) + x

    def lee_montaje(self, msg):
        """T_base<-sensor, una sola vez, de TF."""
        import rospy
        try:
            tr = self.tfbuf.lookup_transform(
                self.base, msg.header.frame_id, rospy.Time(0),
                rospy.Duration(2.0))
        except Exception as e:                             # noqa: BLE001
            # SIN MONTAJE NO SE PUBLICA NADA, y rtabmap se queda sin nubes: la
            # corrida sale vacia con ficheros de tamano normal, que es la forma
            # mas cara de fallar.  Por eso es logerr y no logwarn, y por eso
            # dice el sintoma y no solo la causa.
            self.n_sin_montaje += 1
            rospy.logerr_throttle(
                5.0, 'NO SE PUBLICA NINGUNA NUBE (%d ya descartadas): no hay '
                'TF %s <- %s (%s).  ~base_frame tiene que ser el ROOT LINK del '
                'modelo, que no siempre se llama base_link: en el tracked es '
                '`body`.  rtabmap no va a recibir nada mientras esto siga.',
                self.n_sin_montaje, self.base, msg.header.frame_id, e)
            return False
        v, r = tr.transform.translation, tr.transform.rotation
        self.mount = (quat_mat([r.x, r.y, r.z, r.w]),
                      np.array([v.x, v.y, v.z]))
        rospy.loginfo('montaje %s <- %s: xyz=(%.4f, %.4f, %.4f)',
                      self.base, msg.header.frame_id, v.x, v.y, v.z)
        return True

    # ----------------------------------------------------------------- work
    def cb_nube(self, msg):
        """Solo encola: la pose que hace falta todavia no ha ocurrido."""
        self.n_in += 1
        if self.mount is None and not self.lee_montaje(msg):
            return
        if self.dtype is None:
            self.prepara(msg)
        self.pendientes.append(msg)
        if len(self.pendientes) > 50:
            del self.pendientes[0]

    def drena(self, _evento=None):
        """Procesa las nubes cuyo barrido ya cubre el historial de poses."""
        while self.pendientes:
            msg = self.pendientes[0]
            t0 = msg.header.stamp.to_sec()
            with self.lock:
                ultimo = self.poses[-1][0] if self.poses else None
            if ultimo is None:
                return
            if ultimo < t0 + self.rev:
                # Todavia no ha pasado el barrido entero.  Se espera, salvo que
                # lleve tanto retraso que ya no vaya a llegar.
                if ultimo - t0 < self.espera_max:
                    return
                self.n_tarde += 1
            del self.pendientes[0]
            self.procesa(msg, t0)

    def procesa(self, msg, t0):
        a = np.frombuffer(msg.data, dtype=self.dtype)
        x = a['x'].astype(np.float64)
        y = a['y'].astype(np.float64)
        z = a['z'].astype(np.float64)
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        u = fase(x, y, self.cut, self.dir)

        def pose_de_u(us):
            return self.pose_sensor(t0 + us * self.rev)

        if pose_de_u(0.0) is None or pose_de_u(1.0) is None:
            # El historial no cubre el barrido entero.  Pasa en el arranque y
            # tras un hipo del planificador: se publica SIN deformar y se
            # cuenta, en vez de tirar la nube en silencio -- que es la fuga
            # que costo una campaña entera en el ensamblador viejo.
            self.n_sin_pose += 1
            # SIN DEFORMAR NO ES UN APAÑO, ES OTRO EXPERIMENTO: la nube que sale
            # es la instantanea con un canal de tiempo pegado, o sea --lidar
            # normal disfrazado.  Los ficheros salen de tamano normal y las
            # metricas salen buenisimas.  Por eso es logerr.
            self.rospy.logerr_throttle(
                5.0, 'NUBE SIN DEFORMAR (%d de %d): el historial de poses no '
                'cubre su barrido.  Lo que se esta midiendo es un LiDAR '
                'INSTANTANEO, no uno barrido: la corrida NO vale.',
                self.n_sin_pose, self.n_in)
            out = np.stack([x, y, z], axis=1)
        else:
            out = deforma(np.stack([x, y, z], axis=1), u, self.sectores,
                          pose_de_u)

        rec = np.zeros(int(np.count_nonzero(ok)), dtype=self.dtype_out)
        for nombre in self.dtype.names:
            rec[nombre] = a[nombre][ok]
        rec['x'], rec['y'], rec['z'] = out[ok, 0], out[ok, 1], out[ok, 2]
        rec[self.t_campo] = (u[ok] * self.rev).astype(np.float32)
        # rtabmap y LIO-SAM esperan el canal ordenado para no barrer la nube
        # buscando sus extremos.
        rec = rec[np.argsort(rec[self.t_campo], kind='stable')]

        vacios = self.sectores - len(np.unique(
            np.minimum((u[ok] * self.sectores).astype(np.int64),
                       self.sectores - 1)))
        out_msg = self.PointCloud2()
        out_msg.header.stamp = msg.header.stamp
        out_msg.header.frame_id = self.frame or msg.header.frame_id
        out_msg.height = 1
        out_msg.width = len(rec)
        out_msg.fields = self.campos
        out_msg.is_bigendian = False
        out_msg.point_step = self.point_step
        out_msg.row_step = self.point_step * len(rec)
        out_msg.is_dense = True
        out_msg.data = rec.tobytes()
        self.pub.publish(out_msg)
        self.n_out += 1
        self.rospy.loginfo_throttle(
            5.0, 'nube: %d puntos, %d sectores (%d vacios), costura en '
            '%.1f deg, sentido %+d, desfases %.4f..%.4f s | entradas=%d '
            'DEFORMADAS=%d sin_deformar=%d tarde=%d cola=%d',
            len(rec), self.sectores, vacios,
            math.degrees(self.cut), int(self.dir),
            float(rec[self.t_campo][0]), float(rec[self.t_campo][-1]),
            self.n_in, self.n_out - self.n_sin_pose, self.n_sin_pose,
            self.n_tarde, len(self.pendientes))


if __name__ == '__main__':
    import rospy
    rospy.init_node('sweep_distortion')
    Deformador()
    rospy.spin()
