#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Saca alfa -y el ancho de via efectivo- de los CSV del banco de giro.

Por cada escalon descarta el transitorio y ajusta una recta al yaw del tramo
estable: la pendiente ES la velocidad angular conseguida, sin derivar nada
punto a punto y por tanto sin amplificar ruido.

alfa = w_conseguida / w_mandada, y la correccion es B_efectivo = B / alfa.
"""
from __future__ import print_function

import argparse
import csv
import math
import os

import numpy as np

# Ancho de via declarado en cada nodo twist_to_*.  El rocker_bogie no tiene
# uno: va con Ackermann y dos vias distintas, asi que su correccion no es un
# ancho efectivo -se informa alfa igual, que es lo que se queria medir-.
B_DECLARADO = {
    'differential': 0.5709,
    'tracked': 0.5709,
    'rocker_bogie': None,
}
# Por encima de esto el escalon se tomo en pendiente y no se promedia.
PENDIENTE_MAX_DEG = 8.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def desenrolla(y):
    """Yaw continuo: quita los saltos de +-2*pi para poder ajustar una recta."""
    out = [y[0]]
    for k in range(1, len(y)):
        out.append(out[-1] + wrap(y[k] - y[k - 1]))
    return np.array(out)


def lee(p):
    with open(p) as fh:
        rd = csv.reader(fh)
        next(rd)
        filas = []
        for f in rd:
            try:
                filas.append([float(x) for x in f])
            except ValueError:
                continue
    return np.array(filas) if filas else None


def analiza(path, skip):
    D = lee(path)
    if D is None or len(D) < 50:
        return None, []
    step, v, w, t, x, y, yaw, roll, pitch = (D[:, i] for i in range(9))
    filas = []
    for s in sorted(set(step.tolist())):
        m = step == s
        if m.sum() < 20:
            continue
        tt, yy = t[m], desenrolla(yaw[m])
        est = tt >= skip                      # fuera el transitorio
        if est.sum() < 20:
            continue
        A = np.vstack([tt[est], np.ones(est.sum())]).T
        pend = float(np.linalg.lstsq(A, yy[est], rcond=None)[0][0])
        res = yy[est] - (A @ np.linalg.lstsq(A, yy[est], rcond=None)[0])
        r2 = 1.0 - float((res ** 2).sum()
                         / max(((yy[est] - yy[est].mean()) ** 2).sum(), 1e-12))
        incl = math.degrees(math.hypot(float(np.mean(roll[m])),
                                       float(np.mean(pitch[m]))))
        filas.append({
            'step': int(s), 'v': float(v[m][0]), 'w_cmd': float(w[m][0]),
            'w_real': pend, 'r2': r2, 'incl': incl,
            'alfa': pend / float(w[m][0]) if w[m][0] else float('nan'),
        })
    return filas, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.expanduser(
        '~/metrics_output/yaw_calibration'))
    ap.add_argument('--skip', type=float, default=2.5,
                    help='segundos de transitorio que se tiran, por escalon')
    a = ap.parse_args()

    resumen = {}
    for robot in ('differential', 'tracked', 'rocker_bogie'):
        p = os.path.join(a.dir, robot + '.csv')
        if not os.path.isfile(p):
            print('%-14s (sin CSV)' % robot)
            continue
        filas, _ = analiza(p, a.skip)
        if not filas:
            print('%-14s (sin escalones utiles)' % robot)
            continue
        print()
        print('=' * 72)
        print('%s   (%d escalones)' % (robot.upper(), len(filas)))
        print('=' * 72)
        print('  v      w_mand   w_real    alfa    R2   inclin')
        for f in filas:
            marca = '  <- pendiente' if f['incl'] > PENDIENTE_MAX_DEG else ''
            print('  %4.2f  %+6.2f  %+7.3f  %6.3f  %5.3f  %5.1f deg%s'
                  % (f['v'], f['w_cmd'], f['w_real'], f['alfa'], f['r2'],
                     f['incl'], marca))

        buenas = [f for f in filas
                  if f['incl'] <= PENDIENTE_MAX_DEG and f['r2'] > 0.9
                  and f['alfa'] == f['alfa']]
        if not buenas:
            print('  (ningun escalon limpio)')
            continue
        # alfa por velocidad lineal: en un skid-steer no tiene por que ser
        # el mismo parado que en marcha.
        print()
        print('  alfa por velocidad lineal:')
        for vv in sorted(set(f['v'] for f in buenas)):
            sub = [f['alfa'] for f in buenas if f['v'] == vv]
            print('    v = %.2f m/s : alfa = %.3f  (n=%d, sd %.3f)'
                  % (vv, np.mean(sub), len(sub), np.std(sub)))
        # LA CURVA alfa(w).  Es el entregable: el banco del 2026-08-31
        # mostro que alfa NO es una constante -el Husky no rota por debajo de
        # 0.5 rad/s y el tracked se satura, cayendo de 0.55 a 0.17-, asi que
        # un escalar no sirve como correccion.  Esta tabla si.
        print()
        print('  CURVA alfa(w)  -- promediando los dos signos:')
        print('    |w| mandada   alfa    n   w_real media')
        for ww in sorted(set(abs(f['w_cmd']) for f in buenas)):
            sub = [f for f in buenas if abs(f['w_cmd']) == ww]
            aa = [f['alfa'] for f in sub]
            rr = [abs(f['w_real']) for f in sub]
            print('      %5.2f      %6.3f  %3d   %6.3f rad/s'
                  % (ww, np.mean(aa), len(sub), np.mean(rr)))
        # Asimetria izquierda/derecha: en el tracked salio de casi 2x.
        izq = [f['alfa'] for f in buenas if f['w_cmd'] > 0]
        der = [f['alfa'] for f in buenas if f['w_cmd'] < 0]
        if izq and der:
            print('    asimetria: alfa(+) %.3f  vs  alfa(-) %.3f  (%.2fx)'
                  % (np.mean(izq), np.mean(der),
                     max(np.mean(izq), np.mean(der))
                     / max(min(np.mean(izq), np.mean(der)), 1e-6)))

        al = [f['alfa'] for f in buenas]
        m, sd = float(np.mean(al)), float(np.std(al))
        resumen[robot] = (m, sd, len(al))
        print()
        print('  ALFA GLOBAL = %.3f   (n=%d, sd %.3f)' % (m, len(al), sd))
        B = B_DECLARADO.get(robot)
        if B:
            print('  B declarado %.4f m  ->  B_EFECTIVO = %.4f m  (x %.2f)'
                  % (B, B / m, 1.0 / m))
        else:
            print('  Ackermann: la correccion no es un ancho de via; alfa se '
                  'informa para comparar')

    if resumen:
        print()
        print('=' * 72)
        print('RESUMEN')
        for k, (m, sd, n) in resumen.items():
            B = B_DECLARADO.get(k)
            extra = ('B_ef %.4f m' % (B / m)) if B else 'Ackermann'
            print('  %-14s alfa %.3f +- %.3f  (n=%d)   %s' % (k, m, sd, n, extra))


if __name__ == '__main__':
    main()
