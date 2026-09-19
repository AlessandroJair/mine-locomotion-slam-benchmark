#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Produce /imu/data a partir de /imu/data_raw: FUSIONA la orientacion y
rellena las covarianzas.

Entra  ~input   (/imu/data_raw)   del plugin de Gazebo
Sale   ~output  (/imu/data)       lo que leen rtabmap, metrics_logger y
                                  wheel_odometry

POR QUE SE FUSIONA, SI EL PLUGIN YA DA UNA ORIENTACION
======================================================
Porque esa orientacion no es una medida: es la POSE EXACTA del link, copiada
de Gazebo.  Y no es "verdad de terreno con ruido encima" - es verdad de
terreno a secas.  La spec de SDF no tiene elemento de ruido para la
orientacion: el bloque <imu> solo declara <noise> bajo <angular_velocity> y
<linear_acceleration>, y el <gaussianNoise> del propio plugin esta en 0.0
porque usa el rand() de C, al que `gzserver --seed` no llega.  O sea que la
semilla, que si hace reproducible todo lo demas, ahi no tiene nada que sembrar.

Consumirla era meter la respuesta correcta en una campana cuyo objeto es medir
cuanto se equivoca la estimacion.  rtabmap la usaba con Imu/Enable y
wait_imu_to_init, y wheel_odometry.py sacaba de ella su roll y su pitch.

QUE SE HACE EN SU LUGAR
=======================
Un filtro complementario sobre las DOS senales que si son medidas con ruido
sembrado: el giroscopo y el acelerometro.

  ROLL y PITCH   el giroscopo los propaga y el acelerometro los ancla.  Es
                 observable porque la gravedad da una vertical absoluta.
  GUIÑADA        SOLO integracion del giroscopo, y por tanto DERIVA.  Es lo
                 correcto: sin magnetometro, un IMU no observa el rumbo
                 absoluto, y este proyecto no declara ninguno.

La deriva de guiñada no rompe a sus dos consumidores porque ninguno usa la
orientacion como actitud absoluta:
  - rtabmap toma la diferencia entre transformadas consecutivas
    (imuLastTransform_.inverse() * imuCurrentTransform, Odometry.cpp), y con
    guess_frame_id puesto ni siquiera entra en esa rama.
  - wheel_odometry.py integra el ritmo, no lee el quaternion.

LAS TASAS DE EULER NO SON LAS DEL CUERPO
========================================
Se propaga con la cinematica correcta y no con w*dt, porque el Rocker-Bogie
llega a 27 grados de cabeceo y el tracked a 39, donde la aproximacion de
angulo pequeno ya se nota:

    roll'  = wx + sin(roll)tan(pitch) wy + cos(roll)tan(pitch) wz
    pitch' =      cos(roll)           wy - sin(roll)           wz
    yaw'   = (   sin(roll)            wy + cos(roll)           wz)/cos(pitch)

EL ACELEROMETRO SE IGNORA CUANDO NO MIDE GRAVEDAD
=================================================
Solo corrige mientras |a| se parece a g.  En este terreno hace falta: el pico
de |a_z - g| medido es de 494 m/s2 en el husky.  Sin la compuerta, cada
impacto contra un escalon inclinaria la estimacion como si el robot se hubiera
volcado.

EL SESGO DEL GIROSCOPO SE ESTIMA PARADO
=======================================
Los bloques <noise> declaran bias_mean 0.00075 y bias_stddev 0.005 rad/s, que
en una corrida de 300 s son hasta 1.5 rad de guiñada si no se quita.  Se
promedian las primeras muestras mientras el robot esta quieto - que es lo que
hace un IMU real al arrancar, y lo que permite el warmup de la campana - y se
publica el ritmo ya compensado.  La estimacion sale del propio sensor, no de
Gazebo.
"""
import math

import rospy
import tf.transformations as tft
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Imu


def mediana(v):
    """Mediana sin numpy.  Se usa en vez de la media porque el castañeteo de
    contacto son picos: desplazan la media y no la mediana."""
    s = sorted(v)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class Ahrs(object):

    def __init__(self):
        rospy.init_node('imu_ahrs')

        # stddev, no varianza: se declaran igual que en sim_config.yaml
        w_sd = float(rospy.get_param('~angular_velocity_stddev', 0.009))
        a_sd = float(rospy.get_param('~linear_acceleration_stddev', 0.021))
        o_sd = float(rospy.get_param('~orientation_stddev', 0.02))
        self.w_var, self.a_var, self.o_var = w_sd ** 2, a_sd ** 2, o_sd ** 2

        # Constante de tiempo del filtro.  Grande = mas giroscopo, mas inmune
        # a la aceleracion propia, y mas sesgo residual (~bias*tau).  Con los
        # 0.005 rad/s declarados, 1 s deja ~0.3 grados.
        self.tau = float(rospy.get_param('~tau', 1.0))
        self.g = float(rospy.get_param('~gravity', 9.81))
        # Cuanto puede alejarse |a| de g y seguir valiendo como vertical.
        self.tol_g = float(rospy.get_param('~accel_gate', 0.15))
        self.n_cal = int(rospy.get_param('~bias_samples', 200))
        self.cal_max_s = float(rospy.get_param('~bias_timeout_s', 30.0))
        # Umbrales del test de REPOSO, que no son los de la compuerta de
        # gravedad.  El de aceleracion es flojo a proposito: el tracked
        # castañetea contra el suelo estando parado -18.3 % de sus muestras
        # caen fuera de +-15 % de g- y con 0.5 m/s2 no juntaba muestras y se
        # quedaba sin calibrar.  El del giroscopo es el que manda, porque el
        # giroscopo es lo que se esta midiendo.
        self.cal_w_max = float(rospy.get_param('~bias_gyro_max', 0.05))
        self.cal_a_tol = float(rospy.get_param('~bias_accel_tol', 2.0))
        self.compensa = bool(rospy.get_param('~publish_bias_corrected', True))

        self.roll = self.pitch = self.yaw = 0.0
        self.t_prev = None
        self.bias = [0.0, 0.0, 0.0]
        self.cal = []
        self.calibrando = True
        self.t0 = None
        self.n_sin_gravedad = 0
        self.n_total = 0

        self.pub = rospy.Publisher('~output', Imu, queue_size=20)
        rospy.Subscriber('~input', Imu, self.cb, queue_size=20)
        rospy.loginfo('imu_ahrs: filtro complementario tau=%.2f s, '
                      'compuerta |a|/g +-%.0f%%, sesgo con %d muestras quieto',
                      self.tau, 100 * self.tol_g, self.n_cal)
        rospy.on_shutdown(self.resumen)

    def resumen(self):
        rospy.loginfo('imu_ahrs: %d muestras, %d sin gravedad utilizable '
                      '(%.1f%%), sesgo=[%+.5f %+.5f %+.5f] rad/s',
                      self.n_total, self.n_sin_gravedad,
                      100.0 * self.n_sin_gravedad / max(1, self.n_total),
                      *self.bias)

    # ------------------------------------------------------------ calibrado
    def calibra(self, w, a, t):
        """Promedia el giroscopo mientras el robot esta quieto."""
        if self.t0 is None:
            self.t0 = t
        quieto = (max(abs(x) for x in w) < self.cal_w_max and
                  abs(math.sqrt(sum(x * x for x in a)) - self.g)
                  < self.cal_a_tol)
        if quieto:
            self.cal.append((w, a))
        if len(self.cal) >= self.n_cal:
            self.bias = [mediana([c[0][i] for c in self.cal]) for i in range(3)]
            am = [mediana([c[1][i] for c in self.cal]) for i in range(3)]
            self.roll, self.pitch = self.de_acelerometro(am)
            self.yaw = 0.0
            self.calibrando = False
            rospy.loginfo('imu_ahrs: sesgo del giroscopo [%+.5f %+.5f %+.5f] '
                          'rad/s con %d muestras; actitud inicial '
                          'roll=%+.2f pitch=%+.2f deg',
                          self.bias[0], self.bias[1], self.bias[2], len(self.cal),
                          math.degrees(self.roll), math.degrees(self.pitch))
        elif t - self.t0 > self.cal_max_s:
            self.calibrando = False
            rospy.logwarn('imu_ahrs: solo %d muestras quieto en %.0f s; se '
                          'sigue con sesgo cero y la guiñada derivara mas',
                          len(self.cal), self.cal_max_s)
            if self.cal:
                am = [mediana([c[1][i] for c in self.cal]) for i in range(3)]
                self.roll, self.pitch = self.de_acelerometro(am)

    def de_acelerometro(self, a):
        """roll y pitch de la direccion de la gravedad.  a = fuerza especifica,
        que en reposo vale (0, 0, +g)."""
        ax, ay, az = a
        return (math.atan2(ay, az),
                math.atan2(-ax, math.sqrt(ay * ay + az * az)))

    # ------------------------------------------------------------- callback
    def cb(self, msg):
        self.n_total += 1
        w = [msg.angular_velocity.x, msg.angular_velocity.y,
             msg.angular_velocity.z]
        a = [msg.linear_acceleration.x, msg.linear_acceleration.y,
             msg.linear_acceleration.z]
        t = msg.header.stamp.to_sec()
        if t <= 0.0:
            t = rospy.Time.now().to_sec()

        if self.calibrando:
            self.calibra(w, a, t)
            self.t_prev = t
            self.publica(msg, w, a)
            return

        wc = [w[i] - self.bias[i] for i in range(3)]
        dt = t - self.t_prev if self.t_prev is not None else 0.0
        self.t_prev = t

        if 0.0 < dt < 0.5:
            cr, sr = math.cos(self.roll), math.sin(self.roll)
            cp = math.cos(self.pitch)
            tp = math.tan(self.pitch)
            if abs(cp) < 1e-3:            # cerca de +-90 deg no hay solucion
                cp = 1e-3 if cp >= 0 else -1e-3
            dr = wc[0] + sr * tp * wc[1] + cr * tp * wc[2]
            dp = cr * wc[1] - sr * wc[2]
            dy = (sr * wc[1] + cr * wc[2]) / cp

            roll = self.roll + dr * dt
            pitch = self.pitch + dp * dt
            self.yaw += dy * dt

            norma = math.sqrt(sum(x * x for x in a))
            if abs(norma - self.g) <= self.tol_g * self.g:
                ra, pa = self.de_acelerometro(a)
                al = self.tau / (self.tau + dt)
                # Diferencia envuelta: sin esto, un paso por +-pi da un salto.
                roll += (1.0 - al) * math.atan2(math.sin(ra - roll),
                                                math.cos(ra - roll))
                pitch += (1.0 - al) * math.atan2(math.sin(pa - pitch),
                                                 math.cos(pa - pitch))
            else:
                self.n_sin_gravedad += 1
            self.roll, self.pitch = roll, pitch

        self.publica(msg, wc if self.compensa else w, a)

    # -------------------------------------------------------------- salida
    def publica(self, orig, w, a):
        m = Imu()
        m.header = orig.header
        m.orientation = Quaternion(*tft.quaternion_from_euler(
            self.roll, self.pitch, self.yaw))
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = w
        m.linear_acceleration.x = a[0]
        m.linear_acceleration.y = a[1]
        m.linear_acceleration.z = a[2]
        m.orientation_covariance = [self.o_var, 0, 0,
                                    0, self.o_var, 0,
                                    0, 0, self.o_var]
        m.angular_velocity_covariance = [self.w_var, 0, 0,
                                         0, self.w_var, 0,
                                         0, 0, self.w_var]
        m.linear_acceleration_covariance = [self.a_var, 0, 0,
                                            0, self.a_var, 0,
                                            0, 0, self.a_var]
        self.pub.publish(m)


if __name__ == '__main__':
    try:
        Ahrs()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
