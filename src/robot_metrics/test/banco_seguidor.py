#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Banco frio del seguidor: afina la accion integral sin arrancar Gazebo.

PARTE 1. Identifica la planta de guiñada de cada plataforma a partir de las
corridas: w_real ~ primer orden de ganancia g y constante tau sobre w_mandada.
La ganancia sale del regimen permanente (tramos de mando constante) y tau del
ajuste del transitorio completo.

PARTE 2. Cierra el lazo contra la ruta de referencia REAL con la ley de
trajectory_follower.py -pure pursuit + termino de error lateral + integral- y
barre k_xte_i. Lo que se mira es el error lateral que queda y si aparece
oscilacion.

No sustituye a una corrida: no hay contactos, ni suspension, ni derrape
longitudinal. Sirve para descartar ganancias malas y para elegir uno o dos
candidatos que probar de verdad.
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import yaml

RUTA = ('/home/alessandro/journal_comparison/src/robot_metrics/config/'
        'reference_path_mine.yaml')
B = '/home/alessandro/metrics_output_rtf025_3robots/fixed_trajectory/sliced_lidar'
CORRIDAS = [('differential', f'{B}/differential/run01'),
            ('tracked', f'{B}/tracked/run01'),
            ('rocker_bogie', f'{B}/rocker_bogie/run01')]

# Los del seguidor, tal cual estan hoy
V_MAX = 0.5
W_MAX = 1.0
K_CURVE = 0.6
K_XTE = 2.5
XTE_CURV_MAX = 0.5
LA_MIN, LA_K, LA_MAX = 0.6, 1.6, 1.6
W_V_FLOOR = 0.10
MIN_AIM = 0.25
ALIGN_TH = 0.8
DT = 0.02                     # el seguidor corre a 50 Hz
WARMUP_S = 5.0


# ------------------------------------------------------------ identificacion
def strip_unit(c):
    return c.rsplit('[', 1)[0].strip() if c.endswith(']') and '[' in c else c.strip()


def identifica(run_dir):
    df = pd.read_csv(os.path.join(run_dir, 'metrics.csv')).rename(
        columns=lambda c: strip_unit(c))
    t = df['timestamp'].values
    t = t - t[0]
    m = t >= WARMUP_S
    t, cmd, w = t[m], df['cmd_vyaw'].values[m], df['angular_vel_z'].values[m]

    # ganancia en permanente: tramos de mando casi constante >= 1 s
    razones, pesos = [], []
    i, n = 0, len(cmd)
    while i < n:
        j = i + 1
        while j < n and abs(cmd[j] - cmd[i]) <= 0.05:
            j += 1
        if t[j - 1] - t[i] >= 1.0 and abs(np.mean(cmd[i:j])) > 0.10:
            k = i + int(np.searchsorted(t[i:j] - t[i], 0.5))
            if j - k >= 3:
                razones.append(np.mean(w[k:j]) / np.mean(cmd[i:j]))
                pesos.append(t[j - 1] - t[k])
        i = j
    g = float(np.average(razones, weights=pesos))

    # tau: primer orden que mejor explica el transitorio, con esa g fija
    dt = float(np.median(np.diff(t)))
    mejor = (1e9, None)
    for tau in np.arange(0.05, 2.01, 0.05):
        a = math.exp(-dt / tau)
        y = np.empty_like(w)
        y[0] = w[0]
        for k in range(1, len(w)):
            y[k] = a * y[k - 1] + (1 - a) * g * cmd[k]
        e = float(np.mean((y - w) ** 2))
        if e < mejor[0]:
            mejor = (e, float(tau))
    return g, mejor[1], dt


# ------------------------------------------------------------------- la ruta
def carga_ruta():
    y = yaml.safe_load(open(RUTA))

    def saca(o):
        if isinstance(o, dict):
            for k in ('path', 'points', 'waypoints', 'poses'):
                if k in o:
                    return saca(o[k])
        return o
    pts = saca(y)
    return np.array([[p['x'], p['y']] if isinstance(p, dict) else [p[0], p[1]]
                     for p in pts])


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------- lazo cerrado
def simula(P, g, tau, k_i, vueltas=1, i_max=0.4):
    """Devuelve (xte medio, p95, max, oscilacion) siguiendo P."""
    n = len(P)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    largo = float(seg.sum())
    x, y, th = P[0, 0], P[0, 1], math.atan2(P[1, 1] - P[0, 1], P[1, 0] - P[0, 0])
    w_real = 0.0
    integral = 0.0
    v_prev = 0.0
    a = math.exp(-DT / tau)
    xtes, curvs = [], []
    recorrido = 0.0
    pasos = 0
    lim = int(vueltas * largo / (0.3 * DT))       # tope de seguridad
    while recorrido < vueltas * largo and pasos < lim:
        pasos += 1
        d = np.hypot(P[:, 0] - x, P[:, 1] - y)
        i = int(np.argmin(d))
        # signo del error lateral respecto al tramo
        j0 = min(i, n - 2)
        tx, ty = P[j0 + 1] - P[j0]
        nx, ny = -ty, tx
        norm = math.hypot(nx, ny) or 1.0
        xte_s = ((x - P[j0, 0]) * nx + (y - P[j0, 1]) * ny) / norm
        xtes.append(abs(xte_s))

        L = min(LA_MAX, LA_MIN + LA_K * v_prev)
        # punto objetivo: el primero a mas de L por delante
        k = i
        acc = 0.0
        while k < n - 1 and acc < L:
            acc += float(np.linalg.norm(P[k + 1] - P[k]))
            k += 1
        txp, typ = P[k]
        dx, dy = txp - x, typ - y
        lx = math.cos(th) * dx + math.sin(th) * dy
        ly = -math.sin(th) * dx + math.cos(th) * dy
        L2 = lx * lx + ly * ly
        apuntando = L2 > MIN_AIM * MIN_AIM
        curv = 2.0 * ly / L2 if apuntando else 0.0

        # termino lateral: proporcional + integral, con anti-windup
        term = -(K_XTE * xte_s + k_i * integral)
        sat = abs(term) > XTE_CURV_MAX
        term = max(-XTE_CURV_MAX, min(XTE_CURV_MAX, term))
        if not (sat and term * (-xte_s) > 0) and apuntando:
            integral += xte_s * DT
            integral = max(-i_max, min(i_max, integral))
        curv += term
        curvs.append(curv)

        head_err = abs(wrap(math.atan2(dy, dx) - th))
        v = V_MAX / (1.0 + K_CURVE * abs(curv))
        girando = apuntando and head_err > ALIGN_TH
        if girando:
            v = 0.0
        w_cmd = curv * max(v, W_V_FLOOR)
        if girando:
            w_cmd = math.copysign(W_MAX, wrap(math.atan2(dy, dx) - th))
        w_cmd = max(-W_MAX, min(W_MAX, w_cmd))
        v = max(0.0, min(V_MAX, v))

        w_real = a * w_real + (1 - a) * g * w_cmd
        th = wrap(th + w_real * DT)
        x += v * math.cos(th) * DT
        y += v * math.sin(th) * DT
        recorrido += v * DT
        v_prev = v

    xtes = np.array(xtes)
    c = np.array(curvs)
    # oscilacion: cuantas veces por metro cambia de signo la curvatura mandada
    cruces = int(np.sum(np.diff(np.sign(c)) != 0))
    return (float(xtes.mean()), float(np.percentile(xtes, 95)),
            float(xtes.max()), cruces / max(recorrido, 1e-6))


def main():
    P = carga_ruta()
    print('ruta: %d puntos, %.1f m\n' % (len(P), np.linalg.norm(
        np.diff(P, axis=0), axis=1).sum()))

    plantas = {}
    print('%-14s %8s %8s' % ('planta', 'g', 'tau [s]'))
    for nombre, d in CORRIDAS:
        g, tau, _ = identifica(d)
        plantas[nombre] = (g, tau)
        print('%-14s %8.3f %8.2f' % (nombre, g, tau))
    print()

    ks = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
    print('%-14s %6s %9s %9s %9s %11s' %
          ('planta', 'k_i', 'xte med', 'xte p95', 'xte max', 'cruces/m'))
    for nombre, (g, tau) in plantas.items():
        for k_i in ks:
            med, p95, mx, osc = simula(P, g, tau, k_i)
            marca = '   <- hoy' if k_i == 0.0 else ''
            print('%-14s %6.1f %9.3f %9.3f %9.3f %11.2f%s' %
                  (nombre, k_i, med, p95, mx, osc, marca))
        print()


if __name__ == '__main__':
    main()
