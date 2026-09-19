#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Barrido fino del lookahead, vigilando que no cambie la VELOCIDAD.

Acortar el lookahead reduce el corte de curva, pero pide mas curvatura, y el
seguidor frena con la curvatura (v = v_max/(1+0.6|curv|)).  Si de paso cambia
la velocidad -o peor, la cambia distinto en cada plataforma- se cambia un
confundido por otro: la velocidad fija las nubes/m.
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/alessandro')
import banco_seguidor as bs                                     # noqa: E402


def simula(P, g, tau, la_min, la_k, la_max):
    """Como bs.simula pero devolviendo tambien velocidad media y giro/m."""
    bs.LA_MIN, bs.LA_K, bs.LA_MAX = la_min, la_k, la_max
    n = len(P)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    largo = float(seg.sum())
    x, y = P[0, 0], P[0, 1]
    th = math.atan2(P[1, 1] - P[0, 1], P[1, 0] - P[0, 0])
    w_real, v_prev = 0.0, 0.0
    a = math.exp(-bs.DT / tau)
    xtes, curvs, vs, giro = [], [], [], 0.0
    recorrido, pasos = 0.0, 0
    lim = int(largo / (0.3 * bs.DT))
    while recorrido < largo and pasos < lim:
        pasos += 1
        d = np.hypot(P[:, 0] - x, P[:, 1] - y)
        i = int(np.argmin(d))
        j0 = min(i, n - 2)
        tx, ty = P[j0 + 1] - P[j0]
        nx, ny = -ty, tx
        norm = math.hypot(nx, ny) or 1.0
        xte_s = ((x - P[j0, 0]) * nx + (y - P[j0, 1]) * ny) / norm
        xtes.append(abs(xte_s))

        L = min(la_max, la_min + la_k * v_prev)
        k, acc = i, 0.0
        while k < n - 1 and acc < L:
            acc += float(np.linalg.norm(P[k + 1] - P[k]))
            k += 1
        dx, dy = P[k, 0] - x, P[k, 1] - y
        lx = math.cos(th) * dx + math.sin(th) * dy
        ly = -math.sin(th) * dx + math.cos(th) * dy
        L2 = lx * lx + ly * ly
        apuntando = L2 > bs.MIN_AIM * bs.MIN_AIM
        curv = 2.0 * ly / L2 if apuntando else 0.0
        term = -bs.K_XTE * xte_s
        term = max(-bs.XTE_CURV_MAX, min(bs.XTE_CURV_MAX, term))
        curv += term
        curvs.append(curv)

        head_err = abs(bs.wrap(math.atan2(dy, dx) - th))
        v = bs.V_MAX / (1.0 + bs.K_CURVE * abs(curv))
        girando = apuntando and head_err > bs.ALIGN_TH
        if girando:
            v = 0.0
        w_cmd = curv * max(v, bs.W_V_FLOOR)
        if girando:
            w_cmd = math.copysign(bs.W_MAX, bs.wrap(math.atan2(dy, dx) - th))
        w_cmd = max(-bs.W_MAX, min(bs.W_MAX, w_cmd))
        v = max(0.0, min(bs.V_MAX, v))

        w_real = a * w_real + (1 - a) * g * w_cmd
        th = bs.wrap(th + w_real * bs.DT)
        x += v * math.cos(th) * bs.DT
        y += v * math.sin(th) * bs.DT
        recorrido += v * bs.DT
        giro += abs(w_real) * bs.DT
        vs.append(v)
        v_prev = v

    xtes, c, vs = np.array(xtes), np.array(curvs), np.array(vs)
    mov = vs > 0.15
    cruces = int(np.sum(np.diff(np.sign(c)) != 0)) / max(recorrido, 1e-6)
    return dict(med=float(xtes.mean()), p95=float(np.percentile(xtes, 95)),
                mx=float(xtes.max()), osc=cruces,
                v=float(vs[mov].mean()) if mov.any() else 0.0,
                giro_m=math.degrees(giro) / max(recorrido, 1e-6))


def main():
    P = bs.carga_ruta()
    plantas = {}
    for nombre, d in bs.CORRIDAS:
        g, tau, _ = bs.identifica(d)
        plantas[nombre] = (g, tau)

    rejilla = [(0.6, 1.6, 1.6), (0.5, 1.3, 1.4), (0.4, 1.0, 1.2),
               (0.35, 0.9, 1.1), (0.3, 0.8, 1.0), (0.25, 0.6, 0.9),
               (0.2, 0.5, 0.8)]
    print('%-14s %20s %8s %8s %8s %8s %8s %9s' %
          ('planta', 'L = min + k*v (tope)', 'xte med', 'p95', 'max',
           'v [m/s]', 'giro/m', 'cruces/m'))
    for nombre, (g, tau) in plantas.items():
        for lmin, lk, lmax in rejilla:
            r = simula(P, g, tau, lmin, lk, lmax)
            et = '%.2f + %.1f v (%.1f)' % (lmin, lk, lmax)
            print('%-14s %20s %8.3f %8.3f %8.3f %8.3f %8.2f %9.2f%s' %
                  (nombre, et, r['med'], r['p95'], r['mx'], r['v'],
                   r['giro_m'], r['osc'],
                   '  <- hoy' if (lmin, lk, lmax) == (0.6, 1.6, 1.6) else ''))
        print()


if __name__ == '__main__':
    main()
