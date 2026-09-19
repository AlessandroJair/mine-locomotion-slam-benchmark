#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fija el factor de tiempo real de Gazebo EN CALIENTE, sin tocar el mundo.

    set_rtf.py 0.25

Por que en caliente y no editando lcmine.world: check_sim_parity.py comprueba
que los tres lcmine.world son BYTE A BYTE IDENTICOS (md5).  Editar uno rompe esa
comprobacion para todas las plataformas y para todas las campanas.  Aqui se
llama al servicio y el archivo se queda como esta.

QUE ES EL TECHO DE RTF
======================
Gazebo avanza max_step_size segundos de simulacion por paso, y le pide al reloj
de pared real_time_update_rate pasos por segundo.  El producto es el RTF que
INTENTA mantener:

    RTF_techo = max_step_size * real_time_update_rate = 0.0005 * 1000 = 0.50

Es un techo, no un suelo: si la fisica no da para tanto, el RTF cae por debajo.
Por eso el husky mide 0.482 (pegado al techo, no le limita la maquina) y el
tracked 0.246 (a la mitad del techo, limitado por su fisica).

max_update_rate se calcula a partir del time_step QUE GAZEBO REPORTA, no de un
0.0005 escrito aqui: si algun dia cambia el paso, esto sigue dando el RTF
pedido en vez de uno silenciosamente distinto.

POR QUE IMPORTA PARA LAS MEDIDAS
================================
Con use_sim_time, bajar el RTF le da al SLAM MAS tiempo de pared por segundo de
simulacion, asi que procesa MAS nubes de las 10/s que produce el sensor.  Medido
el 2026-09-03: husky RTF 0.482 -> 51 % de las nubes; tracked RTF 0.246 -> 96 %.
O sea que esto NO es un ajuste de rendimiento: es una variable experimental.
"""
import sys

import rospy
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties


def main():
    if len(sys.argv) != 2:
        sys.stderr.write('uso: set_rtf.py <rtf objetivo>\n')
        return 2
    objetivo = float(sys.argv[1])
    if not 0.0 < objetivo <= 10.0:
        sys.stderr.write('rtf fuera de rango: %r\n' % objetivo)
        return 2

    rospy.init_node('set_rtf', anonymous=True)
    rospy.loginfo('esperando a los servicios de fisica de Gazebo...')
    rospy.wait_for_service('/gazebo/get_physics_properties', timeout=120.0)
    rospy.wait_for_service('/gazebo/set_physics_properties', timeout=120.0)

    lee = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
    pon = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)

    p = lee()
    techo_antes = p.time_step * p.max_update_rate
    nuevo = objetivo / p.time_step

    rospy.loginfo('antes : time_step=%.6f s  max_update_rate=%.1f Hz  -> techo RTF %.3f',
                  p.time_step, p.max_update_rate, techo_antes)
    rospy.loginfo('pido  : max_update_rate=%.1f Hz  -> techo RTF %.3f', nuevo, objetivo)

    # Se reenvia TODO lo que Gazebo tenia.  El servicio reemplaza el bloque
    # entero, asi que omitir el ode_config dejaria el solver en los valores por
    # defecto compilados (quick, 50 iteraciones) en vez de las 100 que declara
    # el mundo, y eso SI cambiaria la fisica.
    r = pon(time_step=p.time_step,
            max_update_rate=nuevo,
            gravity=p.gravity,
            ode_config=p.ode_config)
    if not r.success:
        rospy.logerr('el servicio dijo que no: %s', r.status_message)
        return 1

    q = lee()
    rospy.loginfo('despues: time_step=%.6f s  max_update_rate=%.1f Hz  -> techo RTF %.3f',
                  q.time_step, q.max_update_rate, q.time_step * q.max_update_rate)
    rospy.loginfo('solver: %d iteraciones, w=%.3f (el mundo declara 100)',
                  q.ode_config.sor_pgs_iters, q.ode_config.sor_pgs_w)
    if abs(q.time_step * q.max_update_rate - objetivo) > 1e-6:
        rospy.logerr('NO se aplico: el techo sigue en %.3f', q.time_step * q.max_update_rate)
        return 1
    rospy.loginfo('techo de RTF fijado en %.3f', objetivo)
    return 0


if __name__ == '__main__':
    sys.exit(main())
