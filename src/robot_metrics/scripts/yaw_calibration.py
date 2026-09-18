#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Banco de calibracion del giro: cuanto gira de verdad cada plataforma.

POR QUE EXISTE
==============
El seguidor manda una velocidad angular y da por hecho que la plataforma la
entrega.  Ninguna de las tres lo hace.  Medido el 2026-08-31 sobre la vuelta
del differential, que cierra y por tanto se puede contrastar contra 2*pi de
giro neto: se mandaron 20.40 rad y se consiguieron 6.40.  El Husky realiza el
31 % de lo que se le pide.

Un skid-steer tiene que arrastrar las ruedas de lado para rotar, y el modelo
cinematico que usan los nodos twist_to_* -v_L,R = (v -+ w*B/2)/r- no lo
contempla.  La correccion es un ancho de via EFECTIVO, B_ef = B / alfa, pero
para eso hace falta alfa, y alfa no se puede sacar de las corridas de ruta:

  * ahi el mando no se estabiliza nunca -el seguidor corrige a 20 Hz-, asi
    que no hay regimen permanente que medir;
  * dos estimadores razonables sobre los mismos datos dan cosas distintas
    (cociente de integrales 0.31, pendiente por minimos cuadrados 0.16), lo
    que significa que la respuesta NO es una ganancia estatica: hay retardo,
    entre 0.15 y 0.80 s segun el tramo;
  * el R2 de la regresion se queda entre 0.10 y 0.50, o sea que el mando
    instantaneo explica poco de lo que el robot hace en ese instante.

Este banco mide lo unico que sirve para una correccion estatica: la GANANCIA
EN CONTINUA.  Manda escalones de velocidad angular constante, aguanta cada
uno hasta que asienta, descarta el transitorio y mide el giro conseguido en
el tramo estable.

QUE MIDE
========
Una rejilla de w x v, porque en un skid-steer el arrastre depende de las dos:
girar parado no cuesta lo mismo que girar en marcha.  Cada escalon se manda
en los dos signos seguidos, de modo que el robot vuelve sobre si mismo y el
banco no se va caminando por el mapa.

Sale un CSV por muestra y un resumen por escalon.  El analisis lo hace
analiza_calibracion.py, no este nodo: aqui solo se recoge.

LO QUE NO HACE
==============
No comprueba que el suelo sea llano.  Registra el cabeceo y el balanceo en
cada escalon para que el analisis pueda descartar los que se tomaron en
pendiente, pero no elige el sitio.
"""
from __future__ import print_function

import csv
import math
import os
import time

import rospy
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

try:
    from gazebo_msgs.msg import ModelStates, ModelState
    from gazebo_msgs.srv import SetModelState
except ImportError:
    ModelStates = ModelState = SetModelState = None


class YawCalibration(object):
    def __init__(self):
        rospy.init_node('yaw_calibration')

        self.model = rospy.get_param('~model_name')
        topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel')
        self.out = rospy.get_param('~output_csv')
        # Sostenimiento por escalon.  El retardo medido llega a 0.80 s, y
        # cinco constantes de tiempo son ~4 s; 6 s deja margen y aun asi
        # mantiene la rejilla entera por debajo de 5 minutos de simulacion.
        self.hold = float(rospy.get_param('~hold_s', 6.0))
        # Lo que se tira por delante de cada escalon: el transitorio.
        self.skip = float(rospy.get_param('~skip_s', 2.5))
        # Reposo entre escalones, para que el siguiente arranque de cero.
        self.rest = float(rospy.get_param('~rest_s', 2.0))

        # REPOSICION ENTRE ESCALONES.
        #
        # La primera version dejaba al robot conducir libremente y se iba
        # caminando por el mapa: a 0.5 m/s son 3 m por escalon, y con 24
        # escalones en marcha acababa a 30 m del sitio, encajado contra el
        # terreno de la mina.  Medido el 2026-08-31: el tracked paso sus doce
        # escalones de v=0.50 con 54 grados de inclinacion y sin girar nada.
        # Esos datos no eran del vehiculo, eran de la pared que lo sujetaba.
        #
        # Ahora cada escalon empieza en el MISMO sitio, y ese sitio esta en el
        # grass_plane: una losa de 150x150 m centrada en (-70, 0) con la cara
        # superior en z = -0.55, o sea llana de verdad y despejada -la mina
        # ocupa x de 18 a 57-.  Asi la inclinacion deja de contaminar la
        # medida y todos los escalones ven el mismo suelo.
        self.reset_pose = bool(rospy.get_param('~reset_pose', True))
        self.rx = float(rospy.get_param('~reset_x', -70.0))
        self.ry = float(rospy.get_param('~reset_y', 0.0))
        # ALTURA DE REPOSICION.
        #
        # -0.20 dejaba al robot 20 cm por encima de su altura de reposo
        # (-0.407 en el differential), o sea un golpe en cada uno de los 36
        # escalones.  Con el contacto blando de antes (kp = 1e5) daba igual;
        # con kp = 1e7 es un impacto de verdad, y el rocker_bogie -el unico
        # con suspension articulada- puede quedarse oscilando.
        #
        # Ahora esto es solo la altura del PRIMER intento: tras asentar, el
        # nodo mide donde reposa de verdad y usa esa altura mas drop_margin
        # para el resto.  Asi se auto-ajusta a cada plataforma sin fijar un
        # numero por robot, que era la otra opcion y envejece mal.
        self.rz = float(rospy.get_param('~reset_z', -0.20))
        self.drop_margin = float(rospy.get_param('~drop_margin', 0.005))
        self._rz_medida = None
        self.settle = float(rospy.get_param('~settle_s', 2.5))
        self.set_state = None

        ws = rospy.get_param('~yaw_rates', '0.1,0.2,0.3,0.5,0.7,1.0')
        vs = rospy.get_param('~lin_speeds', '0.0,0.25,0.5')
        self.ws = [float(x) for x in str(ws).split(',') if x.strip()]
        self.vs = [float(x) for x in str(vs).split(',') if x.strip()]

        self.pub = rospy.Publisher(topic, Twist, queue_size=10)
        self.pose = None
        self.z_actual = None
        self.rpy = (0.0, 0.0, 0.0)
        if ModelStates is not None:
            rospy.Subscriber('/gazebo/model_states', ModelStates,
                             self.cb, queue_size=50, tcp_nodelay=True)

        self.rows = []
        self._guardado = False
        # Un banco que muere a medias tiene datos utiles: los escalones que
        # si completo.  Sin esto se perdian enteros.
        rospy.on_shutdown(self.guarda)
        rospy.loginfo('yaw_calibration: modelo=%s topico=%s', self.model, topic)
        rospy.loginfo('  w = %s rad/s', self.ws)
        rospy.loginfo('  v = %s m/s', self.vs)
        n = len(self.ws) * len(self.vs) * 2
        rospy.loginfo('  %d escalones x %.1f s + %.1f s de reposo = %.0f s de '
                      'simulacion', n, self.hold, self.rest,
                      n * (self.hold + self.rest))

    def cb(self, msg):
        try:
            i = msg.name.index(self.model)
        except ValueError:
            return
        p = msg.pose[i]
        q = p.orientation
        self.rpy = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.z_actual = p.position.z
        self.pose = (p.position.x, p.position.y, self.rpy[2])

    # ------------------------------------------------------------------
    def manda(self, v, w):
        tw = Twist()
        tw.linear.x = v
        tw.angular.z = w
        self.pub.publish(tw)

    def repon(self):
        """Devuelve el robot al punto de partida, parado y a nivel."""
        if not self.reset_pose or self.set_state is None:
            return
        self.manda(0.0, 0.0)
        time.sleep(0.2)
        st = ModelState()
        st.model_name = self.model
        st.pose.position.x = self.rx
        st.pose.position.y = self.ry
        st.pose.position.z = (self._rz_medida if self._rz_medida is not None
                              else self.rz)
        st.pose.orientation.w = 1.0          # sin rotacion
        st.reference_frame = 'world'
        try:
            self.set_state(st)
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(30.0, 'no pude reponer la pose: %s', e)
            return
        # Asentar: la suspension del rocker y las orugas necesitan unos
        # decimos de segundo para dejar de rebotar tras el salto.
        t0 = time.time()
        while not rospy.is_shutdown() and time.time() - t0 < self.settle:
            self.manda(0.0, 0.0)
            time.sleep(0.05)
        # La primera vez, aprender la altura de reposo real de ESTA
        # plataforma y usarla a partir de ahora: caida de drop_margin en vez
        # de los 20 cm del arranque.
        if self._rz_medida is None and self.z_actual is not None:
            self._rz_medida = self.z_actual + self.drop_margin
            rospy.loginfo('yaw_calibration: reposa en z = %.4f; a partir de '
                          'ahora repongo en %.4f (caida de %.0f mm)',
                          self.z_actual, self._rz_medida,
                          self.drop_margin * 1000.0)

    def espera_pose(self):
        """Espera la primera pose, contando en reloj de PARED.

        Con rospy.get_time() esto colgaba: si /clock no avanza -el
        rocker_bogie se quedo asi el 2026-08-31, con el reloj clavado en
        0.000000- el tiempo de simulacion nunca crece, el timeout no salta y
        rospy.Rate.sleep() bloquea para siempre.  El banco se comio los
        1200 s del tope sin escribir una fila.  El reloj de pared corre
        pase lo que pase.
        """
        t0 = time.time()
        ultimo_aviso = t0
        while not rospy.is_shutdown() and self.pose is None:
            ahora = time.time()
            if ahora - t0 > 60.0:
                rospy.logerr('yaw_calibration: 60 s sin pose de '
                             '/gazebo/model_states para el modelo "%s"; '
                             'sim_time=%.3f', self.model, rospy.get_time())
                return False
            if ahora - ultimo_aviso > 10.0:
                ultimo_aviso = ahora
                rospy.logwarn('yaw_calibration: esperando pose de "%s" '
                              '(%.0f s, sim_time=%.3f)',
                              self.model, ahora - t0, rospy.get_time())
            time.sleep(0.1)
        if self.pose is None:
            return False
        # El reloj tiene que avanzar, o los escalones no terminan nunca.
        t0, s0 = time.time(), rospy.get_time()
        while not rospy.is_shutdown() and rospy.get_time() == s0:
            if time.time() - t0 > 30.0:
                rospy.logerr('yaw_calibration: el reloj de simulacion no '
                             'avanza (sim_time clavado en %.3f); abandono',
                             s0)
                return False
            time.sleep(0.1)
        return True

    def escalon(self, idx, v, w):
        """Un escalon.  Devuelve las filas que produjo."""
        rate = rospy.Rate(50.0)
        t0 = rospy.get_time()
        filas = []
        while not rospy.is_shutdown():
            t = rospy.get_time() - t0
            if t >= self.hold:
                break
            self.manda(v, w)
            if self.pose is not None:
                filas.append([idx, '%.4f' % v, '%.4f' % w, '%.4f' % t,
                              '%.6f' % self.pose[0], '%.6f' % self.pose[1],
                              '%.6f' % self.pose[2],
                              '%.6f' % self.rpy[0], '%.6f' % self.rpy[1]])
            rate.sleep()
        # reposo
        t0 = rospy.get_time()
        while not rospy.is_shutdown() and rospy.get_time() - t0 < self.rest:
            self.manda(0.0, 0.0)
            rate.sleep()
        return filas

    def run(self):
        if not self.espera_pose():
            return
        # Un poco de quietud antes de empezar, para que el spawn asiente.
        t0 = rospy.get_time()
        while not rospy.is_shutdown() and rospy.get_time() - t0 < 3.0:
            self.manda(0.0, 0.0)
            rospy.sleep(0.05)

        if self.reset_pose and SetModelState is not None:
            try:
                rospy.wait_for_service('/gazebo/set_model_state', timeout=20.0)
                self.set_state = rospy.ServiceProxy(
                    '/gazebo/set_model_state', SetModelState)
                rospy.loginfo('yaw_calibration: repongo la pose en '
                              '(%.1f, %.1f, %.2f) antes de cada escalon',
                              self.rx, self.ry, self.rz)
            except rospy.ROSException:
                rospy.logwarn('yaw_calibration: sin /gazebo/set_model_state; '
                              'los escalones en marcha se iran del sitio')
                self.set_state = None

        idx = 0
        for v in self.vs:
            for w in self.ws:
                # Los dos signos seguidos: el robot deshace lo que acaba de
                # hacer y el banco no se va caminando por el mapa.
                for s in (1.0, -1.0):
                    if rospy.is_shutdown():
                        break
                    idx += 1
                    self.repon()
                    incl = math.degrees(math.hypot(self.rpy[0], self.rpy[1]))
                    rospy.loginfo('escalon %d: v=%.2f w=%+.2f  '
                                  '(inclinacion en reposo %.1f deg)',
                                  idx, v, s * w, incl)
                    self.rows.extend(self.escalon(idx, v, s * w))
        self.manda(0.0, 0.0)
        self.guarda()

    def guarda(self):
        if self._guardado or not self.rows:
            return
        self._guardado = True
        d = os.path.dirname(self.out)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(self.out, 'w') as fh:
            wr = csv.writer(fh)
            wr.writerow(['step', 'cmd_v[m/s]', 'cmd_w[rad/s]', 't_in_step[s]',
                         'gt_x[m]', 'gt_y[m]', 'gt_yaw[rad]',
                         'gt_roll[rad]', 'gt_pitch[rad]'])
            wr.writerows(self.rows)
        rospy.loginfo('yaw_calibration: %d muestras -> %s',
                      len(self.rows), self.out)


if __name__ == '__main__':
    try:
        YawCalibration().run()
    except rospy.ROSInterruptException:
        pass
