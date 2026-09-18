#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Banco de las columnas de suspension.  Sin ROS, sin Gazebo.

Lo que comprueba, en orden de importancia:

1. LA LISTA DE ARTICULACIONES ES LA MISMA EN LOS TRES SITIOS.  El plugin
   suspension_joint_state_publisher de ensamblajeurdf.gazebo decide QUE se
   publica; waypoint_navigation.launch y run_campaign.sh deciden que se
   REGISTRA.  Si se separan, la columna sobrante sale entera en NaN y la que
   falta no sale: silencioso en los dos sentidos.  Esta es la unica de las
   comprobaciones que caza un fallo que no se ve al mirar el CSV.
2. LAS COLUMNAS VAN AL FINAL Y NO MUEVEN A LAS DE ANTES.  metrics.csv se lee
   por nombre, pero gt_highrate y las herramientas que cuentan columnas no.
3. EL TOPIC ES <robotNamespace>/joint_states.  libgazebo_ros_joint_state_
   publisher IGNORA <topicName> -- gazebo_plugins 2.9.3 lo tiene hardcodeado.
   Un <topicName> en el .gazebo no falla: se ignora en silencio y el que lo
   lea apuntara a un topic que no existe.  Costo una campana entera con las
   columnas en NaN (2026-09-10).
4. SIN ARTICULACIONES NO CAMBIA NADA.  Las otras dos plataformas no publican
   suspension, y su CSV tiene que salir byte a byte como antes.

    python3 test_suspension_columns.py
"""
import io
import os
import re
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(RAIZ, 'src/robot_metrics/scripts'))

from metrics_io import (METRIC_COLUMNS, contact_columns,      # noqa: E402
                        csv_header, suspension_columns)

RUEDAS = 'Rueda_6_1,Rueda_4_1,Rueda_2_1,Rueda_1_1,Rueda_3_1,Rueda_5_1'.split(',')


def falla(msg):
    print('  FALLA: %s' % msg)
    return 1


def lista_del_plugin():
    p = os.path.join(RAIZ, 'src/rocker_bogie/urdf/ensamblajeurdf.gazebo')
    s = io.open(p, encoding='utf-8').read()
    m = re.search(r'suspension_joint_state_publisher.*?<jointName>(.*?)'
                  r'</jointName>', s, re.S)
    if not m:
        return None
    return [x.strip() for x in m.group(1).split(',') if x.strip()]


def lista_de(path):
    s = io.open(os.path.join(RAIZ, path), encoding='utf-8').read()
    m = re.search(r'rocker_pivot_left[^"\s]*', s)
    return [x.strip() for x in m.group(0).split(',')] if m else None


def main():
    err = 0

    plugin = lista_del_plugin()
    if not plugin:
        return falla('no encuentro el jointName del plugin en el .gazebo')
    print('1. plugin de Gazebo publica: %s' % ','.join(plugin))
    for path in ('src/rocker_bogie/launch/waypoint_navigation.launch',
                 'src/robot_metrics/scripts/run_campaign.sh'):
        otra = lista_de(path)
        if otra is None:
            err |= falla('%s no declara ninguna articulacion' % path)
        elif sorted(otra) != sorted(plugin):
            err |= falla('%s pide %s' % (path, ','.join(otra)))
        else:
            print('   coincide con %s' % path)

    base = list(METRIC_COLUMNS) + contact_columns(RUEDAS)
    cols = base + suspension_columns(plugin)
    h, hb = csv_header(cols), csv_header(base)

    print('2. %d columnas, %d de suspension al final' % (len(cols), len(plugin)))
    if h[:len(hb)] != hb:
        err |= falla('las columnas de suspension han movido a las de antes')
    esperado = ['suspension_%s[rad]' % j for j in plugin]
    if h[len(hb):] != esperado:
        err |= falla('nombres inesperados: %s' % h[len(hb):])
    if len(cols) != len(METRIC_COLUMNS) + 2 * len(RUEDAS) + len(plugin):
        err |= falla('la cuenta de columnas no sale')

    gaz = io.open(os.path.join(
        RAIZ, 'src/rocker_bogie/urdf/ensamblajeurdf.gazebo'),
        encoding='utf-8').read()
    bloque = gaz[gaz.index('suspension_joint_state_publisher'):]
    bloque = bloque[:bloque.index('</plugin>')]
    # Sin los comentarios: la nota que explica por que NO se pone <topicName>
    # contiene la palabra, y sin esto el banco se acusa a si mismo.
    bloque = re.sub(r'<!--.*?-->', '', bloque, flags=re.S)
    if '<topicName>' in bloque:
        err |= falla('el plugin declara <topicName>, que NO se lee: el topic '
                     'sale en <robotNamespace>/joint_states pase lo que pase')
    else:
        print('3. el .gazebo no declara <topicName> (bien: es inerte)')

    for path, marca in (
            ('src/robot_metrics/launch/metrics_logger.launch',
             'suspension_topic'),
            ('src/robot_metrics/scripts/metrics_logger.py',
             "'~suspension_topic'")):
        t = io.open(os.path.join(RAIZ, path), encoding='utf-8').read()
        i = t.index(marca)
        tramo = t[i:i + 240]
        if '/joint_states' not in tramo or 'suspension_joint_states' in tramo:
            err |= falla('%s no apunta a <ns>/joint_states' % path)
        else:
            print('   %s apunta a <ns>/joint_states' % path)

    print('4. sin articulaciones: %d columnas' % len(hb))
    if csv_header(base + suspension_columns([])) != hb:
        err |= falla('una plataforma sin suspension ve su CSV cambiado')

    print('\n%s' % ('TODO OK' if not err else 'HAY FALLOS'))
    return err


if __name__ == '__main__':
    sys.exit(main())
