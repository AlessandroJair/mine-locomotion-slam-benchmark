#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chassis agitation against SLAM HEADING error - the relationship the
translational analysis could not see.

WHY THIS FIGURE EXISTS
======================
The stability analysis correlated attitude agitation against translational ATE
and came out null: nothing replicated between two runs of the same platform,
and the signs flipped.  That is a real result for position error, and it is not
the whole story, because in a corridor the two are not equally constrained.
The walls fix a LiDAR ICP laterally; almost nothing fixes it in heading.  Yaw
is the degree of freedom that chassis roll should hurt, by tilting the plane
the sensor sweeps - and a heading error is what makes a trajectory diverge
globally while its local relative motion stays accurate.

That is the shape of the campaign's numbers.  Across the three platforms the
ATE ordering does not follow the RPE ordering:

    vibration RMS  2.27 < 3.00 < 3.10    rocker, differential, tracked
    ATE rotational 7.51 < 10.1 < 15.1    same order
    ATE trans      2.03 < 2.84 < 5.07    same order
    RPE trans      0.059  0.096  0.054   NOT the same order

CUIDADO CON ESA PRIMERA FILA.  Esos numeros de vibracion son del IMU con
`accel_z - 9.81`, que es lo que este fichero calculaba hasta 2026-09-04, y ese
orden NO es el de la vibracion real: en esta ruta el residuo de gravedad que
deja sin quitar tiene un RMS de ~1.9 m/s^2 y coloca al rocker por debajo del
husky.  Con gt_a*, en la campana de rtf025_3robots, el orden es

    vibration RMS  1.153 < 2.110 < 2.423   husky, rocker, tracked

El panel (c) ya usa gt_a*.  La fila de arriba se deja porque documenta la
comparacion tal como se leyo entonces; no se cite.

slam_metrics has always computed ate_rot_series, and nothing consumed it.

WHAT THE THREE PANELS SAY, AND WHAT THEY DO NOT
===============================================
(a) The trend, binned.  Raw 1 s windows are a noise cloud - the effect is a few
    tenths of a percent of the variance - so the windows are pooled into
    agitation deciles per platform and the panel shows the mean heading-error
    growth in each, with the standard error.  The faint points behind are the
    raw windows, so the reader can see how much scatter the bins are hiding.

(b) Why the claim is "consistent", not "significant".  Every run's correlation
    with its bootstrap CI.  Against heading error all six are positive; against
    translational error the sign is mixed.  Most individual CIs span zero: the
    evidence here is the agreement of the sign, not any one run.  And the six
    runs are NOT independent - they are two runs of each of three platforms -
    so the honest count is three, which is p ~ 0.25 under a sign test, not the
    p ~ 0.03 that six independent runs would give.  Suggestive.  Not proof.

(c) Where the strong signal actually is: between platforms, not within runs.
    Six points, monotonic.  Also the panel that shows why (a) and (b) cannot
    carry the claim on their own - with three platforms, vibration is
    confounded with every other thing that distinguishes one chassis from
    another.

Usage:
    ./plot_agitation_heading.py --results ~/metrics_output/fixed_trajectory \\
        --output_dir ~/metrics_output/fixed_trajectory/paper_tables
"""
import argparse
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import slam_metrics as sm                                # noqa: E402
from metrics_io import (load_metrics, load_gt_highrate,  # noqa: E402
                        ventanas_alta_frecuencia, vibration_rms)
from aggregate_runs import trim_warmup                   # noqa: E402

# Same palette and typography as analyze_metrics.py, so the figure sits beside
# the others without looking like it came from somewhere else.
COLORS = {'rocker_bogie': '#2ca02c',
          'differential': '#1f77b4',
          'tracked': '#d62728'}
LABEL = {'rocker_bogie': 'Rocker-Bogie',
         'differential': 'Husky (differential)',
         'tracked': 'Tracked'}
ORDER = ['rocker_bogie', 'differential', 'tracked']
MARKERS = {'run01': 'o', 'run02': '^'}

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 9,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
})

# 100 ms: un barrido del LiDAR.  Ver load_gt_highrate() en
# metrics_io.py para por que era 1 s y por que ya no hace falta.
WINDOW_S = 0.1


def _etiqueta_ventana():
    """'100 ms' o '1 s', segun WINDOW_S.  Para que los ejes no mientan."""
    return ('%.0f ms' % (WINDOW_S * 1000.0) if WINDOW_S < 1.0
            else '%g s' % WINDOW_S)


def windows(df, res):
    """Per-window agitation, vibration, and growth of both error channels."""
    t = df['timestamp'].values
    t = t - t[0]
    n = min(len(t), len(res['ate_trans_series']), len(res['ate_rot_series']))
    t = t[:n]
    e_tr = np.asarray(res['ate_trans_series'][:n], dtype=float)
    e_ro = np.degrees(np.asarray(res['ate_rot_series'][:n], dtype=float))
    # Verdad-terreno, no IMU: el ruido de orientacion del sensor entra en
    # la std de la ventana en cuadratura y no es movimiento del chasis.
    pitch = (df['gt_pitch'] if 'gt_pitch' in df.columns else df['pitch']).values[:n]
    roll = (df['gt_roll'] if 'gt_roll' in df.columns else df['roll']).values[:n]

    # Agitacion desde gt_highrate cuando la corrida lo tiene: verdad-terreno
    # a 1000 Hz, que es lo que hace viable la ventana de 100 ms.  A los 50 Hz
    # de metrics.csv una ventana asi tiene 5 muestras y la std no significa
    # nada.  Las corridas viejas caen al calculo de siempre.
    hr = load_gt_highrate(df.attrs.get('source', ''))
    hr_win = (ventanas_alta_frecuencia(hr, df['timestamp'].values[0], WINDOW_S)
              if hr is not None else {})

    edges = np.arange(0.0, t[-1] + WINDOW_S, WINDOW_S)
    idx = np.digitize(t, edges) - 1
    agit, g_ro, g_tr = [], [], []
    for k in range(len(edges) - 1):
        m = idx == k
        if m.sum() < (3 if hr_win else 5):
            continue
        clave = round(float(edges[k]), 6)
        if hr_win:
            if clave not in hr_win:
                continue
            agit.append(math.degrees(hr_win[clave][0]))
        else:
            agit.append(math.degrees(math.hypot(float(np.std(pitch[m])),
                                                float(np.std(roll[m])))))
        a, b = e_tr[m], e_ro[m]
        g_tr.append(float(a[-1] - a[0]))
        g_ro.append(float(b[-1] - b[0]))
    return np.array(agit), np.array(g_ro), np.array(g_tr)


def collect(results_dir):
    out = []
    for robot in ORDER:
        for run in ('run01', 'run02'):
            p = os.path.join(results_dir, robot, run, 'metrics.csv')
            if not os.path.exists(p):
                continue
            df = trim_warmup(load_metrics(p))
            res = sm.evaluate(df, rpe_delta_m=1.0)
            if res is None:
                continue
            agit, g_ro, g_tr = windows(df, res)
            out.append({
                'robot': robot, 'run': run,
                'agit': agit, 'g_ro': g_ro, 'g_tr': g_tr,
                'ate_rot': res.get('ate_origin_rot_rmse_deg',
                                   math.degrees(res.get('ate_origin_rot_rmse', float('nan')))),
                # La misma vibracion que las tablas: verdad-terreno.  Este
                # panel leia el IMU con `accel_z - 9.81`, que en esta ruta es
                # sobre todo la proyeccion de g del cabeceo, y ponia al rocker
                # POR DEBAJO del husky mientras la tabla lo ponia por encima.
                'vib': vibration_rms(df, '%s %s' % (robot, run)),
            })
            print('  %-13s %s  %4d ventanas' % (robot, run, len(agit)))
    return out


def panel_trend(ax, data):
    """Binned mean heading-error growth against agitation."""
    bin_x, bin_y, bin_e, all_agit = [], [], [], []
    for robot in ORDER:
        runs = [d for d in data if d['robot'] == robot]
        if not runs:
            continue
        agit = np.concatenate([d['agit'] for d in runs])
        g_ro = np.concatenate([d['g_ro'] for d in runs])
        all_agit.append(agit)
        c = COLORS[robot]
        ax.plot(agit, g_ro, '.', color=c, alpha=0.10, markersize=2,
                rasterized=True)
        # deciles of agitation, so every bin holds the same number of windows
        qs = np.quantile(agit, np.linspace(0, 1, 11))
        qs = np.unique(qs)
        xs, ys, es = [], [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (agit >= lo) & (agit < hi)
            if m.sum() < 5:
                continue
            xs.append(float(np.mean(agit[m])))
            ys.append(float(np.mean(g_ro[m])))
            es.append(float(np.std(g_ro[m]) / math.sqrt(m.sum())))
        ax.errorbar(xs, ys, yerr=es, color=c, marker='o', markersize=3.5,
                    linewidth=1.3, capsize=2, label=LABEL[robot])
        bin_x.extend(xs)
        bin_y.extend(ys)
        bin_e.extend(es)
    ax.axhline(0.0, color='0.4', linewidth=0.7, linestyle=':')

    # Scale to the BINNED means.  The raw windows span about +-17 deg while the
    # bins live inside +-0.5, so autoscaling on the cloud flattened the trend
    # onto the axis line and the panel showed nothing at all.
    allb = np.array(bin_y)
    alle = np.array(bin_e)
    if len(allb):
        lo = float(np.min(allb - alle))
        hi = float(np.max(allb + alle))
        pad = max(0.05, 0.25 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
    if len(bin_x):
        ax.set_xlim(0.0, float(np.percentile(np.concatenate(all_agit), 98)))
    # El rotulo sale de WINDOW_S, no escrito a mano: el 2026-09-01 la
    # ventana bajo a 100 ms y la figura siguio diciendo "1 s" mientras
    # dibujaba datos de 100 ms.  Una figura que se lee mal es peor que
    # una que falta.
    ax.set_xlabel('Attitude agitation per ' + _etiqueta_ventana() +
                  ' window,' + chr(10) +
                  r'$\sqrt{\sigma_\theta^2+\sigma_\phi^2}$ (deg)')
    ax.set_ylabel('Heading-error growth\nin the window (deg)')
    ax.set_title('(a) binned trend', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')


def panel_forest(ax, data):
    """Every run's r, with its bootstrap CI, for both response channels."""
    rows = []
    for d in data:
        r_ro, _ = sm.pearson(d['agit'], d['g_ro'])
        ci_ro = sm.bootstrap_ci(d['agit'], d['g_ro'])
        r_tr, _ = sm.pearson(d['agit'], d['g_tr'])
        ci_tr = sm.bootstrap_ci(d['agit'], d['g_tr'])
        rows.append((d, r_ro, ci_ro, r_tr, ci_tr))

    y = np.arange(len(rows))[::-1]
    for k, (d, r_ro, ci_ro, r_tr, ci_tr) in enumerate(rows):
        c = COLORS[d['robot']]
        yy = y[k]
        ax.plot([ci_ro[0], ci_ro[1]], [yy + 0.16, yy + 0.16], color=c,
                linewidth=1.4, solid_capstyle='butt')
        ax.plot([r_ro], [yy + 0.16], 'o', color=c, markersize=5)
        ax.plot([ci_tr[0], ci_tr[1]], [yy - 0.16, yy - 0.16], color=c,
                linewidth=1.0, alpha=0.45, solid_capstyle='butt')
        ax.plot([r_tr], [yy - 0.16], 's', color=c, markersize=3.5, alpha=0.45)
    ax.axvline(0.0, color='0.3', linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(['%s %s' % (LABEL[d['robot']].split(' ')[0], d['run'])
                        for d, _, _, _, _ in rows], fontsize=7)
    ax.set_xlabel('Pearson $r$ with 95% bootstrap CI')
    ax.set_title('(b) per run: heading (filled) vs position (faded)',
                 fontsize=9)


def panel_between(ax, data):
    """The strong signal: between platforms."""
    for d in data:
        ax.plot(d['vib'], d['ate_rot'], MARKERS[d['run']],
                color=COLORS[d['robot']], markersize=7,
                label='%s %s' % (LABEL[d['robot']], d['run']))
    for robot in ORDER:
        runs = [d for d in data if d['robot'] == robot]
        if len(runs) > 1:
            ax.plot([r['vib'] for r in runs], [r['ate_rot'] for r in runs],
                    '-', color=COLORS[robot], alpha=0.35, linewidth=1.0)
    ax.set_xlabel(r'Vibration RMS, $|a_{\rm gt}|$ (m/s$^2$)')
    ax.set_ylabel('ATE rotational RMSE (deg)')
    ax.set_title('(c) between platforms', fontsize=9)
    ax.legend(fontsize=6.5, loc='upper left')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True)
    ap.add_argument('--output_dir', required=True)
    a = ap.parse_args()

    print('leyendo corridas...')
    data = collect(a.results)
    if not data:
        sys.exit('ABORTA: no encontre corridas en %s' % a.results)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))
    panel_trend(axes[0], data)
    panel_forest(axes[1], data)
    panel_between(axes[2], data)
    fig.tight_layout()

    if not os.path.isdir(a.output_dir):
        os.makedirs(a.output_dir)
    for ext in ('eps', 'png'):
        fig.savefig(os.path.join(a.output_dir, 'agitation_vs_heading.%s' % ext),
                    dpi=200)
    plt.close(fig)

    # The numbers behind the figure, so the caption can be written from them.
    print('')
    print('%-13s %6s %8s %8s' % ('robot', 'run', 'r(head)', 'r(pos)'))
    signs = []
    for d in data:
        r_ro, _ = sm.pearson(d['agit'], d['g_ro'])
        r_tr, _ = sm.pearson(d['agit'], d['g_tr'])
        signs.append(r_ro)
        print('%-13s %6s %8.3f %8.3f' % (d['robot'], d['run'], r_ro, r_tr))
    s = np.array(signs)
    print('')
    print('r(heading): media %.3f, rango [%.3f, %.3f], %d de %d positivas'
          % (s.mean(), s.min(), s.max(), (s > 0).sum(), len(s)))
    print('figura escrita en %s/agitation_vs_heading.{eps,png}' % a.output_dir)


if __name__ == '__main__':
    main()
