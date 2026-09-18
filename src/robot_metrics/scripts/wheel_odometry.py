#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Odometria de ruedas para las tres plataformas, con un solo modelo.

Entra  <ns>/joint_states     velocidades de rueda y angulos de direccion
       /imu/data             cabeceo y balanceo, y la tasa de guiñada
Sale   ~odom                 nav_msgs/Odometry
       TF  wheel_odom -> <base_frame>

PARA QUE EXISTE
===============
Para romper el lazo del de-skew.  icp_odometry con deskewing:=true y SIN
guess_frame_id corrige la nube N con la velocidad que el mismo estimo de la
nube N-1; como aqui el barrido dura 100 ms, que es el periodo entre nubes, la
correccion es del mismo tamano que el desplazamiento a medir y el lazo va a
ganancia ~1.  Medido el 2026-09-09: la velocidad estimada alterna de signo y
diverge hasta 11.5 m/s contra 0.44 reales, y el ATE empeora 10x.

Con guess_frame_id apuntando aqui, icp_odometry deskewa por TF contra una
referencia que no sale del ICP, y el lazo desaparece.  Requiere ademas
deskewing_slerp:=true, o hace una consulta de TF POR PUNTO: ~20000 por nube.

EL ARBOL DE TF
==============
rtabmap con guess_frame_id NO publica odom -> base.  Publica la CORRECCION,
odom -> wheel_odom, y deja que este nodo cierre wheel_odom -> base:

    odom  --(icp_odometry)-->  wheel_odom  --(este nodo)-->  base_frame

Por eso guess_frame_id no puede valer lo mismo que odom_frame_id; rtabmap lo
comprueba y desactiva el guess si coinciden.

UN MODELO PARA LAS TRES
=======================
El Husky y el tracked son skid-steer y el Rocker-Bogie es Ackermann con cuatro
ruedas directrices, asi que no comparten formula cerrada.  Si comparten la
restriccion de rodadura: una rueda en (x, y) del cuerpo, con angulo de
direccion d, sobre un solido que se mueve a (vx, 0) con guiñada w, rueda a

    s = cos(d)*vx + w*(x*sin(d) - y*cos(d))

Una ecuacion por rueda, dos incognitas, minimos cuadrados sobre todas las
motrices.  Con d = 0 degenera en s = vx - w*y, el diferencial de siempre, asi
que el skid-steer sale del mismo codigo sin un caso aparte.

DE DONDE SALE CADA COSA, Y POR QUE
==================================
LONGITUDINAL: de las ruedas.  Medido contra el ground truth de Gazebo el
2026-09-09, escala 0.989 / 1.007 / 0.989 en husky / tracked / rocker.  Es
buena porque el modelo de mando usa estos mismos numeros, asi que leer y
mandar son inversos exactos.

GUIÑADA: del GIROSCOPO, no de las ruedas.  Las ruedas la sobreestiman -escala
0.855 / 0.481 / 0.633 en la misma medida- y no es un fallo de geometria sino
la hipotesis vy = 0, que es justo la que un skid-steer viola al girar: la via
efectiva sale 0.668 m contra 0.571 geometrica en el husky, y 1.186 m en el
tracked, donde rasca toda la oruga.  El Rocker-Bogie no desliza por via sino
por sus dos ruedas centrales, que no son directrices y arrastran en curva.
Se podria calibrar una via efectiva por plataforma, pero seria ajustar un
parametro contra el ground truth que el experimento intenta no mirar.

Y SE USA LA TASA DEL GIROSCOPO, NO EL `orientation` DEL IMU.  El plugin de
Gazebo deriva ese quaternion de la pose real del modelo, asi que sacar de ahi
la guiñada seria meter ground truth por la puerta de atras.  La tasa es un
sensor con su ruido declarado, su integracion deriva, y eso da igual aqui: el
de-skew solo usa transformadas RELATIVAS sobre 100 ms, y rtabmap consume el
guess tambien como incremento (previousPose.inverse() * guessCurrentPose).

Roll y pitch salen del `orientation` de /imu/data, que desde el 2026-09-09 lo
FUSIONA imu_ahrs.py a partir del giroscopo y el acelerometro ruidosos, y ya no
es la pose exacta de Gazebo.  Se necesitan porque el de-skew los usa: el LiDAR
va 0.9 m sobre el suelo, asi que un grado de cabeceo durante el barrido le
mueve el origen mas que buena parte del avance.

POR QUE PUBLICA AL RITMO DEL IMU
================================
Porque joint_states NO va igual en las tres: 50 Hz en husky y rocker, 10 Hz en
el tracked (publish_rate en su lcmine_two_track_world.launch).  A 10 Hz el TF
saldria un punto por nube, que para una referencia que tiene que resolver el
INTERIOR de un barrido de 100 ms no sirve.  El IMU va a 100 Hz en las tres
-esta en sensor_standardization.yaml- asi que integrar y publicar en su
callback iguala la cadencia y de paso la mejora.
"""
import math
import threading

import numpy as np
import rospy
import tf.transformations as tft
from geometry_msgs.msg import Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from tf2_ros import TransformBroadcaster


class OdometriaDeRuedas(object):

    def __init__(self):
        rospy.init_node('wheel_odometry', anonymous=False)

        self.radio = float(rospy.get_param('~wheel_radius'))
        # +1 si una velocidad de articulacion positiva empuja al robot hacia
        # el +x de nav; -1 si lo empuja hacia atras.  El Rocker-Bogie tiene
        # base_link mirando a -x y su twist_to_wheels ya invierte el mando una
        # vez, asi que al LEER las articulaciones hay que deshacerlo.
        self.signo = float(rospy.get_param('~joint_sign', 1.0))
        self.base = rospy.get_param('~base_frame')
        self.marco = rospy.get_param('~odom_frame', 'wheel_odom')
        self.pub_tf = bool(rospy.get_param('~publish_tf', True))
        # 'gyro' o 'wheels'.  Ver el bloque de arriba: por defecto giroscopo.
        self.fuente_yaw = rospy.get_param('~yaw_source', 'gyro')
        if self.fuente_yaw not in ('gyro', 'wheels'):
            rospy.logfatal('~yaw_source debe ser gyro o wheels, no "%s"',
                           self.fuente_yaw)
            raise rospy.ROSInitException('yaw_source invalido')
        # Por encima de esto un salto de reloj no se integra: no se inventa un
        # desplazamiento a partir de un hueco de planificacion.
        self.dt_max = float(rospy.get_param('~max_dt', 0.5))

        ruedas = rospy.get_param('~wheels')
        if not ruedas:
            rospy.logfatal('~wheels vacio: sin geometria no hay odometria')
            raise rospy.ROSInitException('~wheels vacio')
        self.ruedas = [{'joint': w['joint'],
                        'x': float(w.get('x', 0.0)),
                        'y': float(w['y']),
                        'steer': w.get('steer')} for w in ruedas]

        self.x = self.y = self.z = self.yaw = 0.0
        self.roll = self.pitch = 0.0
        self.vx = 0.0                 # de las ruedas
        self.w_ruedas = 0.0           # de las ruedas
        self.t_prev = None
        self.lock = threading.Lock()
        self.n_pub = 0
        self.n_joint = 0
        self.n_incompletos = 0

        self.pub = rospy.Publisher('~odom', Odometry, queue_size=100)
        self.bc = TransformBroadcaster()
        rospy.Subscriber(rospy.get_param('~joint_states_topic',
                                         'joint_states'),
                         JointState, self.cb_joints, queue_size=50)
        rospy.Subscriber(rospy.get_param('~imu_topic', '/imu/data'),
                         Imu, self.cb_imu, queue_size=100)

        rospy.loginfo('wheel_odometry: %d ruedas (%d directrices), r=%.5f m, '
                      'signo=%+.0f, guiñada de %s, %s -> %s',
                      len(self.ruedas),
                      sum(1 for w in self.ruedas if w['steer']),
                      self.radio, self.signo, self.fuente_yaw,
                      self.marco, self.base)
        rospy.on_shutdown(self.resumen)

    def resumen(self):
        rospy.loginfo('wheel_odometry: %d publicadas, %d joint_states, '
                      '%d descartados por articulaciones ausentes',
                      self.n_pub, self.n_joint, self.n_incompletos)

    # ------------------------------------------------------------- entradas
    def cb_joints(self, msg):
        pos, vel = {}, {}
        for i, n in enumerate(msg.name):
            if i < len(msg.position):
                pos[n] = msg.position[i]
            if i < len(msg.velocity):
                vel[n] = msg.velocity[i]

        A, b = [], []
        for w in self.ruedas:
            if w['joint'] not in vel:
                continue
            d = 0.0
            if w['steer'] is not None:
                if w['steer'] not in pos:
                    continue
                d = pos[w['steer']]
            s = self.signo * vel[w['joint']] * self.radio
            c, sn = math.cos(d), math.sin(d)
            A.append([c, w['x'] * sn - w['y'] * c])
            b.append(s)

        if len(A) < 2:
            self.n_incompletos += 1
            rospy.logwarn_throttle(
                10.0, 'solo %d ruedas utilizables en joint_states; hacen '
                'falta 2 para resolver (vx, w)', len(A))
            return

        sol, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=None)
        with self.lock:
            self.vx = float(sol[0])
            self.w_ruedas = float(sol[1])
        self.n_joint += 1

    def cb_imu(self, msg):
        q = msg.orientation
        roll, pitch, _ = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        wz = msg.angular_velocity.z
        t = msg.header.stamp.to_sec()
        if t <= 0.0:
            t = rospy.Time.now().to_sec()

        with self.lock:
            self.roll, self.pitch = roll, pitch
            vx = self.vx
            w = wz if self.fuente_yaw == 'gyro' else self.w_ruedas
            if self.t_prev is not None:
                dt = t - self.t_prev
                if 0.0 < dt < self.dt_max:
                    # Punto medio en guiñada: integrar con el rumbo del
                    # principio sesga cada curva hacia fuera.
                    dyaw = w * dt
                    med = self.yaw + 0.5 * dyaw
                    avance = vx * dt
                    # El cabeceo reparte el avance entre horizontal y
                    # vertical; sin esto la odometria sube la rampa en
                    # horizontal y le sobra recorrido plano.
                    horiz = avance * math.cos(pitch)
                    self.x += horiz * math.cos(med)
                    self.y += horiz * math.sin(med)
                    self.z += -avance * math.sin(pitch)
                    self.yaw += dyaw
            self.t_prev = t
            X, Y, Z, YAW = self.x, self.y, self.z, self.yaw

        self.publica(t, X, Y, Z, roll, pitch, YAW, vx, w)

    # -------------------------------------------------------------- salidas
    def publica(self, t, X, Y, Z, roll, pitch, yaw, vx, w):
        q = tft.quaternion_from_euler(roll, pitch, yaw)
        stamp = rospy.Time.from_sec(t)

        if self.pub_tf:
            tr = TransformStamped()
            tr.header.stamp = stamp
            tr.header.frame_id = self.marco
            tr.child_frame_id = self.base
            tr.transform.translation.x = X
            tr.transform.translation.y = Y
            tr.transform.translation.z = Z
            tr.transform.rotation = Quaternion(*q)
            self.bc.sendTransform(tr)

        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = self.marco
        o.child_frame_id = self.base
        o.pose.pose.position.x = X
        o.pose.pose.position.y = Y
        o.pose.pose.position.z = Z
        o.pose.pose.orientation = Quaternion(*q)
        o.twist.twist.linear.x = vx
        o.twist.twist.angular.z = w
        self.pub.publish(o)
        self.n_pub += 1


if __name__ == '__main__':
    try:
        OdometriaDeRuedas()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
