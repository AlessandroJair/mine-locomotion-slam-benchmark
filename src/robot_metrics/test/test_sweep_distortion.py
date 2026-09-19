#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Banco offline de sweep_distortion.py.  Sin ROS, sin Gazebo.

Lo que comprueba, en orden de importancia:

1. CONTRA LA VERDAD, no contra si mismo.  Se construye un mundo estatico, una
   trayectoria analitica del sensor, y se calcula lo que un sensor que barre
   MEDIRIA de verdad:  p_meas = R(t_i)^T (P_mundo - x(t_i)).  deforma() tiene
   que dar eso.  Es la prueba que caza un signo invertido, que es el error que
   este modelo invita a cometer.
2. VIAJE DE IDA Y VUELTA: deskew(deforma(p)) == p.  Es la correccion de
   LIO-SAM, asi que si esto cierra, la nube que sale es deskewable por
   cualquiera que siga esa convencion (rtabmap incluido).
3. COBERTURA: fase() reparte los puntos por los 360 grados sin hueco.  Es la
   propiedad que la cuña girando NO tenia (350-356 deg) y la razon de cambiar.
4. QUIRALIDAD: invertir ~rotor_dir invierte la deformacion.  Es el experimento
   que estaba pendiente y que aqui pasa a ser un parametro.
5. MAGNITUD: a 0.45 m/s y 10 deg/s la distorsion tiene que salir de centimetros,
   no de metros ni de micras.

    python3 test_sweep_distortion.py
"""
import math
import sys

import numpy as np

SCRIPTS = '/home/alessandro/journal_comparison/src/robot_metrics/scripts'
sys.path.insert(0, SCRIPTS)

from sweep_distortion import deforma, deskew, fase, quat_mat, slerp  # noqa

REV = 0.1                 # 10 Hz
SECTORES = 360
V = np.array([0.45, 0.03, 0.0])          # m/s, avance del tracked
OMEGA = math.radians(10.0)               # rad/s de guiñada
N = 20000


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# Pose de partida NO trivial a proposito: un origen y una orientacion cualquiera
# cazan errores de frame que con la identidad pasan desapercibidos.
X0 = np.array([7.0, -3.0, 0.9])
R0 = rotz(0.7).dot(rotx(0.05))


def pose_real(us):
    """Pose MUNDO del sensor en la fraccion us del barrido, analitica."""
    t = us * REV
    return rotz(OMEGA * t).dot(R0), X0 + V * t


def falla(msg):
    print('  FALLO: ' + msg)
    return 1


def main():
    rng = np.random.default_rng(0)
    # Mundo estatico alrededor del sensor, entre 2 y 20 m
    d = rng.normal(size=(N, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    d[:, 2] *= 0.25
    P = X0 + d * rng.uniform(2.0, 20.0, size=(N, 1))

    # La nube instantanea que da Gazebo: el mundo visto desde S(t0)
    Rs0, xs0 = pose_real(0.0)
    ref = (P - xs0).dot(Rs0)                    # = Rs0^T (P - xs0)

    u = fase(ref[:, 0], ref[:, 1], 0.0, +1.0)
    err = 0

    # --- 1. contra la verdad -------------------------------------------------
    got = deforma(ref, u, SECTORES, pose_real)
    k = np.minimum((u * SECTORES).astype(np.int64), SECTORES - 1)
    us_c = (k + 0.5) / SECTORES                 # el sector que usa deforma()
    quiere = np.empty_like(ref)
    for s in np.unique(k):
        sel = k == s
        Ri, xi = pose_real((s + 0.5) / SECTORES)
        quiere[sel] = (P[sel] - xi).dot(Ri)     # Ri^T (P - xi): lo que mide
    dmax = float(np.abs(got - quiere).max())
    print('1. contra la verdad      : max |deforma - medida real| = %.3e m' % dmax)
    if dmax > 1e-9:
        err |= falla('deforma() no reproduce lo que el sensor mediria')

    # y cuanto cuesta cuantizar en sectores, contra el continuo
    cont = np.empty_like(ref)
    for i in range(0, N, 97):                   # una muestra, no hace falta mas
        Ri, xi = pose_real(u[i])
        cont[i] = (P[i] - xi).dot(Ri)
    sub = slice(0, N, 97)
    dq = float(np.abs(got[sub] - cont[sub]).max())
    print('   cuantizacion en %d sectores: %.2f mm' % (SECTORES, dq * 1000.0))
    if dq > 0.005:
        err |= falla('la cuantizacion por sectores pasa de 5 mm')

    # --- 2. ida y vuelta -----------------------------------------------------
    vuelta = deskew(got, u, SECTORES, pose_real)
    dv = float(np.abs(vuelta - ref).max())
    print('2. ida y vuelta          : max |deskew(deforma(p)) - p| = %.3e m' % dv)
    if dv > 1e-9:
        err |= falla('el viaje de ida y vuelta no cierra; revisa el signo')

    # --- 3. cobertura --------------------------------------------------------
    ocupados = len(np.unique(k))
    print('3. cobertura             : %d/%d sectores con puntos, u en '
          '[%.4f, %.4f]' % (ocupados, SECTORES, u.min(), u.max()))
    if ocupados < SECTORES:
        err |= falla('quedan %d sectores vacios' % (SECTORES - ocupados))

    # --- 4. quiralidad -------------------------------------------------------
    u_inv = fase(ref[:, 0], ref[:, 1], 0.0, -1.0)
    got_inv = deforma(ref, u_inv, SECTORES, pose_real)
    dif = float(np.abs(got - got_inv).max())
    print('4. quiralidad            : invertir el sentido mueve los puntos '
          'hasta %.1f mm' % (dif * 1000.0))
    if dif < 1e-4:
        err |= falla('invertir ~rotor_dir no cambia nada; el parametro no hace '
                     'lo que dice')
    # u y u_inv tienen que ser complementarias salvo el punto de corte
    comp = np.abs((u + u_inv) - 1.0)
    comp = np.minimum(comp, np.abs(comp - 1.0))
    if float(comp.max()) > 1e-9:
        err |= falla('fase() no es simetrica al invertir el sentido')

    # --- 5. magnitud ---------------------------------------------------------
    desp = np.linalg.norm(got - ref, axis=1)
    print('5. magnitud              : desplazamiento medio %.1f mm, '
          'p95 %.1f mm, max %.1f mm'
          % (desp.mean() * 1000.0, np.percentile(desp, 95) * 1000.0,
             desp.max() * 1000.0))
    # a 0.45 m/s el sensor avanza 45 mm en un barrido; con 10 deg/s de guiñada
    # un punto a 20 m se mueve otros ~35 mm.  Centimetros, no metros.
    if not (0.001 < desp.mean() < 0.5):
        err |= falla('la distorsion no esta en el orden de centimetros')

    # --- 6. el canal de tiempo ----------------------------------------------
    t = u * REV
    print('6. canal de tiempo       : desfases %.4f..%.4f s de un barrido de '
          '%.3f s' % (t.min(), t.max(), REV))
    if not (t.min() >= 0.0 and t.max() < REV + 1e-12):
        err |= falla('los desfases se salen de [0, T)')

    # --- 7. slerp / quat_mat, por si acaso ----------------------------------
    q = np.array([0.0, 0.0, math.sin(0.35), math.cos(0.35)])
    dm = float(np.abs(quat_mat(q) - rotz(0.7)).max())
    dh = float(np.abs(quat_mat(slerp([0, 0, 0, 1], q, 0.5))
                      - rotz(0.35)).max())
    print('7. cuaterniones          : quat_mat %.2e, slerp a mitad %.2e'
          % (dm, dh))
    if max(dm, dh) > 1e-12:
        err |= falla('quat_mat/slerp no cuadran con la rotacion esperada')

    print('\n%s' % ('TODO OK' if not err else 'HAY FALLOS'))
    return err


if __name__ == '__main__':
    sys.exit(main())
