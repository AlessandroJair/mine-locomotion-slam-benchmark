#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aparta el topic del LiDAR de un modelo, para que el nodo de distorsion de
barrido pueda ponerse en medio.

    retopic_lidar.py <modelo.sdf>                    differential y tracked
    retopic_lidar.py --xacro <robot.xacro> [--xacro-arg n:=v ...]
                                                     rocker_bogie

    gazebo  ->  /velodyne_points_raw  ->  sweep_distortion  ->  /velodyne_points

Escribe el modelo modificado por stdout, para que un launch pueda meterlo en un
parametro con <param command="..."/> y spawnearlo con -param ... -sdf (o -urdf).
No se guarda ninguna copia en el repositorio: hay una sola fuente -el model.sdf
o el xacro de cada robot- y la variante barrida se deriva de ella en cada
arranque, asi que no pueden separarse.

Dos formatos porque los tres robots no se spawnean igual: differential y
tracked salen de su model.sdf con -database, y el rocker_bogie de su
robot_description con -urdf.

LO UNICO QUE SE TOCA ES EL TOPIC
================================
Ni una linea de geometria, ni el ruido sembrado, ni el update_rate, ni los 16
anillos.  Verificado a mano sobre el tracked: el modelo que sale es BIT A BIT
igual al original salvo la cadena del topic.  Esa es la diferencia con el
generador anterior, que borraba el sensor y montaba en su lugar un rotor con
una cuña de 21 grados: aquello dependia de que Gazebo cumpliera el update_rate
del gpu_ray -y no lo cumple-, asi que la cobertura de azimut salia de 350 a 356
grados en vez de 360 y bajaba con la carga de simulacion.  La distorsion
acababa siendo una propiedad del planificador, distinta por plataforma.  Ahora
la mete `sweep_distortion.py` con formulas, sobre la nube entera.

--xacro-arg EXISTE POR ALGO
===========================
Los args del xacro tienen que ser LOS MISMOS con los que se genera
robot_description en el launch.  Los defaults del xacro no coinciden con los
del launch, y si se dejan al azar el modelo barrido y el normal dejan de ser
el mismo robot.
"""
import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET

RAW_TOPIC = '/velodyne_points_raw'


def busca_lidar_sdf(raiz):
    """Devuelve el primer sensor gpu_ray del SDF."""
    for modelo in raiz.iter('model'):
        for link in modelo.findall('link'):
            for sensor in link.findall('sensor'):
                if sensor.get('type') in ('gpu_ray', 'ray'):
                    return sensor
    return None


def busca_lidar_urdf(raiz):
    """Devuelve el primer sensor gpu_ray de los bloques <gazebo> de un URDF."""
    for gz in raiz.findall('gazebo'):
        for sensor in gz.findall('sensor'):
            if sensor.get('type') in ('gpu_ray', 'ray'):
                return sensor
    return None


def retopica(raiz, es_urdf):
    sensor = busca_lidar_urdf(raiz) if es_urdf else busca_lidar_sdf(raiz)
    if sensor is None:
        return 'no hay ningun sensor gpu_ray en el modelo'
    tocados = 0
    for plug in sensor.findall('plugin'):
        t = plug.find('topicName')
        if t is not None:
            t.text = RAW_TOPIC
            tocados += 1
    if not tocados:
        return ('el sensor no declara topicName en ningun plugin; sin eso el '
                'nodo de distorsion no puede ponerse en medio')
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('modelo', nargs='?',
                    help='model.sdf a convertir (differential, tracked)')
    ap.add_argument('--xacro', help='xacro del robot a convertir (rocker_bogie)')
    ap.add_argument('--xacro-arg', nargs='*', default=[], metavar='n:=v',
                    help='args que se le pasan al xacro.  TIENEN que ser los '
                         'mismos con los que se genera robot_description; los '
                         'defaults del xacro no coinciden con los del launch')
    a = ap.parse_args()

    if a.xacro:
        try:
            texto_xml = subprocess.check_output(
                ['xacro', a.xacro] + list(a.xacro_arg))
        except (OSError, subprocess.CalledProcessError) as exc:
            sys.stderr.write('xacro fallo sobre %s: %s\n' % (a.xacro, exc))
            return 1
        raiz = ET.fromstring(texto_xml)
        error = retopica(raiz, True)
    elif a.modelo:
        raiz = ET.parse(a.modelo).getroot()
        error = retopica(raiz, False)
    else:
        ap.error('hace falta un model.sdf o --xacro')
        return 1

    if error:
        sys.stderr.write('%s\n' % error)
        return 1
    sys.stdout.write(ET.tostring(raiz, encoding='unicode'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
