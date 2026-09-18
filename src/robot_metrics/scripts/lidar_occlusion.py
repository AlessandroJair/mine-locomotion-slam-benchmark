#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A que altura deja cada plataforma de taparse su propio anillo bajo.

POR QUE EXISTE.  lidar_height_above_contact_m = 0.9000 (sim_config.yaml) se
fijo el 2026-08-26 con un argumento de ocultacion: a 0.8248 m el rocker se
tapaba el anillo de -15 grados "6.8 deg hacia la camara, 3.9 en las esquinas
del chasis y 1.7 sobre los tubos del rocker", y a 0.9000 lo despejaba por
4.2 grados.  Aquella medida se tomo sobre el LiDAR TROCEADO, que montaba la
cuña en un link propio (velodyne_rotor_link) y por eso veia el chasis.  Ese
arm se retiro el 2026-09-09.  Este script vuelve a hacer la cuenta sobre el
modelo que Gazebo carga HOY.

LA REGLA QUE DECIDE QUE SE VE.  Un gpu_ray no ve el visual de su PROPIO link.
Y la conversion URDF->SDF FUSIONA los links unidos por junta fija, asi que en
el rocker -la unica plataforma que spawnea un URDF- velodyne_base_link,
camera_link e imu_link acaban dentro de base_link y el sensor no ve ni el
chasis ni la camara ni los mastiles.  El husky y el tracked spawnean SDF
nativo, donde no hay fusion, asi que su sensor si ve su chasis entero.  Por
eso el script carga el SDF convertido y no el URDF.

LA CUENTA.  Un rayo a -theta grados que sale de una altura h sobre la huella
esta, a distancia horizontal r del sensor, a la altura h - r*tan(theta).  Un
punto del vehiculo a (r, z) lo tapa si z >= h - r*tan(theta), o sea si

    h <= (z - z_huella) + r*tan(theta)

asi que la altura MINIMA a la que ese punto deja de tapar es el lado derecho,
y la altura minima del sensor es el maximo sobre todos los puntos del cuerpo.
No hace falta buscar: se despeja.

    python3 lidar_occlusion.py [--ring 15] [--margin 4.2] [--rocker-drop X]
"""
from __future__ import print_function

import argparse
import math
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_parser as mp

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(RAIZ, 'src')

# el archivo que Gazebo spawnea de verdad, por plataforma
SPAWN = {
    'differential': ('sdf', 'differential/model/differential/model.sdf'),
    'tracked': ('sdf', 'tracked/gazebo_continuous_track_example/model/'
                       'two_track_robot/model.sdf'),
    'rocker_bogie': ('xacro', 'rocker_bogie/urdf/ensamblajeurdf.xacro'),
}
MESH_ROOTS = [os.path.expanduser('~/.gazebo/models'), SRC]

# LOS OTROS SENSORES NO CUENTAN PARA EL SUELO, y es la trampa de esta cuenta.
# La camara va atornillada al mismo mastil que el LiDAR, 0.1393 m por debajo y
# 0.371 m por delante, asi que tapa el anillo bajo a un angulo FIJO que no
# depende de la altura: si las dos bajan juntas, su margen es el mismo a 0.90
# que a 0.70.  Meterla en el maximo da un "suelo" que se persigue a si mismo.
# Se informa aparte.
CON_EL_SENSOR = {'camera_link', 'camera_optical_frame', 'imu_link'}

# Desde el 2026-09-17 la camara del tracked y la del husky ya no son un link:
# van dentro del chasis (la junta fija cedia, ver CAMARA FUSIONADA en sus
# model.sdf), asi que su caja entra en la nube con el nombre del CHASIS y
# CON_EL_SENSOR deja de atraparla.  Se la reconoce por posicion: todo lo que
# caiga a menos de CAM_R del origen de la camara es la camara.  0.05 m cubre
# la media diagonal de su caja de 0.03 x 0.08 x 0.03, que son 0.0456.
CAM_R = 0.05


def origen_camara(model):
    """Punto desde el que mira la camara, o None si el modelo no la trae."""
    for link in model.links.values():
        for s in link.sensors:
            if s.get('type') == 'camera':
                return link.pose.dot(s['pose'])[:3, 3]
    return None


# ---------------------------------------------------------------- geometria
def _stl_bbox(path):
    """(min, max) de un STL, binario o ascii."""
    with open(path, 'rb') as f:
        head = f.read(84)
        if len(head) < 84:
            return None
        n = struct.unpack('<I', head[80:84])[0]
        body = f.read()
    if len(body) >= 50 * n and n > 0:
        P = np.empty((n * 3, 3))
        for i in range(n):
            o = i * 50 + 12
            P[3 * i:3 * i + 3] = np.frombuffer(
                body[o:o + 36], dtype='<f4').reshape(3, 3)
        return P.min(axis=0), P.max(axis=0)
    txt = open(path, 'rb').read().decode('ascii', 'ignore')
    P = np.array([[float(x) for x in l.split()[1:4]]
                  for l in txt.splitlines() if l.strip().startswith('vertex')])
    return (P.min(axis=0), P.max(axis=0)) if len(P) else None


def _find_mesh(uri):
    rel = uri.replace('model://', '').replace('package://', '')
    for root in MESH_ROOTS:
        for cand in (os.path.join(root, rel),
                     os.path.join(root, *rel.split('/')[1:])):
            if os.path.isfile(cand):
                return cand
    for root in MESH_ROOTS:
        for dirpath, _, files in os.walk(root):
            if os.path.basename(rel) in files:
                return os.path.join(dirpath, os.path.basename(rel))
    return None


def puntos(geom, n=24):
    """Puntos extremos de una primitiva, en su propio marco.  El maximo de la
    elevacion sobre un cuerpo convexo cae siempre en un vertice o en un borde,
    asi que con esquinas y aros basta."""
    t = geom['type']
    if t == 'box':
        sx, sy, sz = np.asarray(geom['size'], float) / 2.0
        return np.array([[x, y, z] for x in (-sx, sx)
                         for y in (-sy, sy) for z in (-sz, sz)])
    if t == 'cylinder':
        r, L = float(geom['radius']), float(geom['length']) / 2.0
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        rim = np.stack([r * np.cos(a), r * np.sin(a), np.zeros(n)], axis=1)
        return np.vstack([rim + [0, 0, L], rim - [0, 0, L]])
    if t == 'sphere':
        r = float(geom['radius'])
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        out = [[0, 0, r], [0, 0, -r]]
        for el in (-0.5, 0.0, 0.5):
            out += [[r * math.cos(el) * math.cos(t_), r * math.cos(el) * math.sin(t_),
                     r * math.sin(el)] for t_ in a]
        return np.array(out)
    if t == 'mesh':
        f = _find_mesh(geom.get('uri', ''))
        if not f:
            return None
        bb = _stl_bbox(f)
        if bb is None:
            return None
        lo, hi = bb
        s = np.asarray(geom.get('scale', (1, 1, 1)), float)
        lo, hi = lo * s, hi * s
        return np.array([[x, y, z] for x in (lo[0], hi[0])
                         for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return None


# ---------------------------------------------------------------- carga
def carga(robot, rocker_drop=None):
    kind, rel = SPAWN[robot]
    path = os.path.join(SRC, rel)
    if kind == 'sdf':
        return mp.load_sdf(path), path
    # el rocker spawnea URDF: hay que pasarlo por la conversion, que es donde
    # se fusionan los links de junta fija y el sensor cambia de dueño
    args = ['xacro', '--inorder', path, 'differential:=false',
            'differential_linkage:=true', 'bogie_upper:=0.6', 'bogie_drop:=0.0']
    if rocker_drop is not None:
        args.append('rocker_drop:=%s' % rocker_drop)
    urdf = subprocess.check_output(args)
    tmp = tempfile.NamedTemporaryFile(suffix='.urdf', delete=False)
    tmp.write(urdf)
    tmp.close()
    sdf = subprocess.check_output(['gz', 'sdf', '-p', tmp.name],
                                  stderr=subprocess.PIPE)
    out = tempfile.NamedTemporaryFile(suffix='.sdf', delete=False)
    out.write(sdf)
    out.close()
    os.unlink(tmp.name)
    return mp.load_sdf(out.name), out.name


def origen_rayos(model):
    """(link del sensor, punto de origen de los rayos en el marco del modelo)."""
    for name, link in model.links.items():
        for s in link.sensors:
            if s.get('type') == 'gpu_ray':
                T = link.pose.dot(s['pose'])
                return name, T[:3, 3]
    raise SystemExit('sin gpu_ray')


def nube(model, excluir):
    """[(link, punto)] de todo lo que el sensor SI ve."""
    out = []
    sin_malla = set()
    for name, link in model.links.items():
        if name == excluir:
            continue
        geos = link.visuals or link.collisions
        for T, g in geos:
            P = puntos(g)
            if P is None:
                sin_malla.add((name, g.get('uri', g['type'])))
                continue
            W = link.pose.dot(T)
            for p in P:
                q = W.dot(np.append(p, 1.0))[:3]
                out.append((name, q))
    return out, sin_malla


# ---------------------------------------------------------------- informe
def analiza(robot, ring, margen, rocker_drop=None, h_actual=0.6700):
    model, path = carga(robot, rocker_drop)
    sensor_link, O = origen_rayos(model)
    z_huella = O[2] - h_actual          # la paridad certifica los 0.9000
    pts, sin_malla = nube(model, sensor_link)
    tan = math.tan(math.radians(ring + margen))
    tan0 = math.tan(math.radians(ring))

    peor_h, peor = -1e9, None
    peor_el, peor_el_link = -1e9, None
    cam_el, cam_link = -1e9, None
    O_cam = origen_camara(model)
    for name, q in pts:
        por_pos = O_cam is not None and np.linalg.norm(q - O_cam) <= CAM_R
        if name in CON_EL_SENSOR or por_pos:
            r = math.hypot(q[0] - O[0], q[1] - O[1])
            el = math.degrees(math.atan2(q[2] - O[2], r)) if r > 1e-9 else 90.0
            if el > cam_el:
                # el nombre es el del link que la LLEVA cuando va fusionada
                cam_el = el
                cam_link = 'camara en ' + name if por_pos else name
            continue
        r = math.hypot(q[0] - O[0], q[1] - O[1])
        z = q[2] - z_huella
        h = z + r * tan
        if h > peor_h:
            peor_h, peor = h, (name, r, z)
        el = math.degrees(math.atan2(q[2] - O[2], r)) if r > 1e-9 else \
            (90.0 if q[2] > O[2] else -90.0)
        if el > peor_el:
            peor_el, peor_el_link = el, (name, r, z)
    return dict(robot=robot, path=path, sensor_link=sensor_link,
                n_links=len(model.links), n_pts=len(pts), sin_malla=sin_malla,
                z_huella=z_huella, h_min=peor_h, culpable=peor,
                elev=peor_el, elev_link=peor_el_link,
                margen_actual=-ring - peor_el, tan0=tan0,
                cam_el=cam_el, cam_link=cam_link,
                cam_margen=-ring - cam_el if cam_link else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ring', type=float, default=15.0,
                    help='elevacion del anillo mas bajo del VLP-16, en grados')
    ap.add_argument('--margin', type=float, default=4.2,
                    help='margen exigido, en grados (4.2 es el que se declaro '
                         'al subir a 0.9000)')
    ap.add_argument('--rocker-drop', default=None)
    ap.add_argument('--robots', default='differential tracked rocker_bogie')
    a = ap.parse_args()

    print('anillo mas bajo -%.1f deg, margen exigido %.1f deg\n' % (a.ring, a.margin))
    print('CUERPO (lo unico que sube respecto al sensor cuando el sensor baja)')
    print('%-14s %-24s %6s %9s %9s %9s' %
          ('plataforma', 'lo que mas tapa', 'r [m]', 'z [m]', 'elev', 'suelo'))
    print('-' * 80)
    h_ref, peor, filas = 0.6700, 0.0, []
    for rb in a.robots.split():
        d = analiza(rb, a.ring, a.margin,
                    a.rocker_drop if rb == 'rocker_bogie' else None)
        peor = max(peor, d['h_min'])
        filas.append(d)
        n, r, z = d['culpable']
        # la elevacion DEL CULPABLE, no la maxima del cuerpo: son puntos
        # distintos en cuanto el vehiculo tiene algo alto y cerca
        el = math.degrees(math.atan2(z - h_ref, r)) if r > 1e-9 else -90.0
        print('%-14s %-24s %6.3f %9.4f %+8.2f  %8.4f' %
              (d['robot'], n[:24], r, z, el, d['h_min']))
        for miss in sorted(d['sin_malla']):
            print('%-14s   [aviso] malla sin resolver: %s' % ('', miss))
    print('-' * 80)
    print('suelo comun del LiDAR: %.4f m  (lo pone %s)'
          % (peor, max(filas, key=lambda f: f['h_min'])['robot']))
    print()
    print('SENSORES QUE BAJAN CON EL LIDAR (angulo fijo, no dependen de la altura)')
    for d in filas:
        if d['cam_link']:
            print('%-14s %-24s elev %+6.2f deg, margen %5.2f deg'
                  % (d['robot'], d['cam_link'], d['cam_el'], d['cam_margen']))
        else:
            print('%-14s (fusionados en base_link: el sensor no los ve)' % d['robot'])


if __name__ == '__main__':
    main()
