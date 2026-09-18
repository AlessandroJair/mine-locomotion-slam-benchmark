#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same metre of mine, different platform: does agitation cost heading accuracy?

WHAT THIS TEST IS FOR
=====================
Every correlation in this study so far pools windows along the route, and the
route is not homogeneous - the step, the corners and the long straights differ
in how much they shake a chassis AND in how well a LiDAR ICP is constrained
there.  A correlation computed that way partly measures where on the route the
window sits.  That confound was visible directly: on one-lap runs the
"significant" results were all against accumulated error, and they collapsed to
zero once a second lap visited the same ground at a different error level.

The fixed-trajectory design makes a cleaner test possible, and nothing has used
it yet.  All three platforms drove the SAME recorded path, so their windows can
be matched by position instead of pooled.  Then the question becomes paired:
at this metre of mine, did the platform that shook more here accumulate more
heading error here?

METHOD
======
Each window (100 ms con gt_highrate, 1 s sin el) is assigned the
arc length of the nearest point on
reference_path_mine.yaml - not elapsed time, and not distance travelled, which
drifts from path position as a platform weaves.  Windows are collected into
bins of --bin-m metres; with two laps each bin holds about two visits per run
and four per platform.

The estimate is a within-bin (fixed-effects) correlation: agitation and heading
error growth are both centred on their bin mean before being pooled, so every
comparison is between platforms at one place, and any effect of the place
itself is removed by construction.

The confidence interval resamples BINS, not windows.  Windows inside a bin are
not independent - they are the same few metres of tunnel - and bootstrapping
them individually would report an interval several times too narrow.

WHAT IT STILL CANNOT DO
=======================
It removes the route.  It does not remove the platform: with three chassis,
agitation is confounded with every other thing that differs between them -
kinematics, odometry model, wheel slip, sensor mounting.  A within-bin
correlation that survives says the association is not an artefact of where the
rough ground is.  It does not say agitation is the cause.

Usage:
    ./plot_position_matched.py --results ~/metrics_output/fixed_trajectory \\
        --output_dir ~/metrics_output/fixed_trajectory/paper_tables
"""
import argparse
import io
import math
import os
import sys

import numpy as np
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import slam_metrics as sm                                # noqa: E402
from metrics_io import (load_metrics, load_gt_highrate,  # noqa: E402
                        ventanas_alta_frecuencia)
from aggregate_runs import trim_warmup                   # noqa: E402

COLORS = {'rocker_bogie': '#2ca02c',
          'differential': '#1f77b4',
          'tracked': '#d62728'}
LABEL = {'rocker_bogie': 'Rocker-Bogie',
         'differential': 'Husky (differential)',
         'tracked': 'Tracked'}
ORDER = ['rocker_bogie', 'differential', 'tracked']

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


def load_path(p):
    d = yaml.safe_load(io.open(p, encoding='utf-8'))
    pts = np.array([[q['x'], q['y']] for q in d['points']])
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return pts, s


def window_table(df, res, pts, s_of):
    """Por ventana (ver WINDOW_S): posicion en la ruta, agitacion y
    crecimiento del error de rumbo."""
    t = df['timestamp'].values
    t = t - t[0]
    n = min(len(t), len(res['ate_rot_series']))
    t = t[:n]
    e_ro = np.degrees(np.asarray(res['ate_rot_series'][:n], dtype=float))
    # Verdad-terreno, no IMU: ver plot_agitation_heading.windows().
    pitch = (df['gt_pitch'] if 'gt_pitch' in df.columns else df['pitch']).values[:n]
    roll = (df['gt_roll'] if 'gt_roll' in df.columns else df['roll']).values[:n]
    gx = df['gt_x'].values[:n]
    gy = df['gt_y'].values[:n]

    # Agitacion desde gt_highrate cuando la corrida lo tiene: verdad-terreno
    # a 1000 Hz, que es lo que hace viable la ventana de 100 ms.  A los 50 Hz
    # de metrics.csv una ventana asi tiene 5 muestras y la std no significa
    # nada.  Las corridas viejas caen al calculo de siempre.
    hr = load_gt_highrate(df.attrs.get('source', ''))
    hr_win = (ventanas_alta_frecuencia(hr, df['timestamp'].values[0], WINDOW_S)
              if hr is not None else {})

    edges = np.arange(0.0, t[-1] + WINDOW_S, WINDOW_S)
    idx = np.digitize(t, edges) - 1
    pos, agit, grow = [], [], []
    for k in range(len(edges) - 1):
        m = idx == k
        if m.sum() < (3 if hr_win else 5):
            continue
        cx, cy = float(np.mean(gx[m])), float(np.mean(gy[m]))
        j = int(np.argmin((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2))
        pos.append(float(s_of[j]))
        clave = round(float(edges[k]), 6)
        if hr_win:
            if clave not in hr_win:
                continue
            agit.append(math.degrees(hr_win[clave][0]))
        else:
            agit.append(math.degrees(math.hypot(float(np.std(pitch[m])),
                                                float(np.std(roll[m])))))
        b = e_ro[m]
        grow.append(float(b[-1] - b[0]))
    return np.array(pos), np.array(agit), np.array(grow)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--path', default='/home/alessandro/journal_comparison/'
                                      'src/robot_metrics/config/reference_path_mine.yaml')
    ap.add_argument('--bin-m', type=float, default=2.0)
    a = ap.parse_args()

    pts, s_of = load_path(a.path)
    print('camino: %d puntos, %.1f m' % (len(pts), s_of[-1]))

    rows = []
    for robot in ORDER:
        for run in ('run01', 'run02'):
            p = os.path.join(a.results, robot, run, 'metrics.csv')
            if not os.path.exists(p):
                continue
            df = trim_warmup(load_metrics(p))
            res = sm.evaluate(df, rpe_delta_m=1.0)
            if res is None:
                continue
            pos, agit, grow = window_table(df, res, pts, s_of)
            rows.append((robot, run, pos, agit, grow))
            print('  %-13s %s  %4d ventanas' % (robot, run, len(pos)))

    nbin = int(math.ceil(s_of[-1] / a.bin_m))
    edges = np.arange(nbin + 1) * a.bin_m

    # (robot, bin) -> mean agitation, mean heading-error growth
    cell = {}
    for robot, run, pos, agit, grow in rows:
        b = np.clip(np.digitize(pos, edges) - 1, 0, nbin - 1)
        for k in range(nbin):
            m = b == k
            if m.sum() < 2:
                continue
            key = (robot, k)
            cell.setdefault(key, [[], []])
            cell[key][0].append(float(np.mean(agit[m])))
            cell[key][1].append(float(np.mean(grow[m])))

    # Keep only bins where all three platforms are present: the test is paired.
    usable = [k for k in range(nbin)
              if all((r, k) in cell for r in ORDER)]
    print('')
    print('bins de %.1f m: %d en total, %d con las tres plataformas'
          % (a.bin_m, nbin, len(usable)))

    A = np.array([[np.mean(cell[(r, k)][0]) for r in ORDER] for k in usable])
    G = np.array([[np.mean(cell[(r, k)][1]) for r in ORDER] for k in usable])

    # Within-bin centring: every comparison is between platforms at one place.
    Ac = A - A.mean(axis=1, keepdims=True)
    Gc = G - G.mean(axis=1, keepdims=True)

    r_within, _ = sm.pearson(Ac.ravel(), Gc.ravel())
    r_pooled, _ = sm.pearson(A.ravel(), G.ravel())

    # Bootstrap over BINS, not windows.
    rng = np.random.RandomState(0)
    boot = []
    for _ in range(2000):
        take = rng.randint(0, len(usable), len(usable))
        rr, _ = sm.pearson(Ac[take].ravel(), Gc[take].ravel())
        if not math.isnan(rr):
            boot.append(rr)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # How often is the most-agitated platform in a bin also the worst?
    conc = sum(1 for i in range(len(usable))
               if int(np.argmax(A[i])) == int(np.argmax(G[i])))
    print('')
    print('correlacion agitacion vs crecimiento del error de rumbo')
    print('  pooled (sin controlar la posicion) r = %6.3f' % r_pooled)
    print('  WITHIN-BIN (posicion controlada)   r = %6.3f   IC95%% [%.3f, %.3f]'
          % (r_within, lo, hi))
    print('  %s' % ('EXCLUYE 0' if not (lo <= 0 <= hi) else 'incluye 0'))
    print('')
    print('  la plataforma mas agitada del bin es tambien la de mas error:')
    print('    %d de %d bins (%.1f%%; al azar seria 33.3%%)'
          % (conc, len(usable), 100.0 * conc / len(usable)))

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))

    ax = axes[0]
    for j, robot in enumerate(ORDER):
        ax.plot(Ac[:, j], Gc[:, j], 'o', color=COLORS[robot], markersize=3.5,
                alpha=0.6, label=LABEL[robot])
    if len(Ac.ravel()) > 2:
        z = np.polyfit(Ac.ravel(), Gc.ravel(), 1)
        xs = np.linspace(Ac.min(), Ac.max(), 10)
        ax.plot(xs, np.polyval(z, xs), '-', color='0.2', linewidth=1.4)
    ax.axhline(0, color='0.5', linewidth=0.7, linestyle=':')
    ax.axvline(0, color='0.5', linewidth=0.7, linestyle=':')
    ax.set_xlabel('Agitation, deviation from\nthe bin mean (deg)')
    ax.set_ylabel('Heading-error growth,\ndeviation from the bin mean (deg)')
    ax.set_title('(a) position-matched, $r=%.3f$ [%.3f, %.3f]'
                 % (r_within, lo, hi), fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    x = np.array([edges[k] for k in usable])
    for j, robot in enumerate(ORDER):
        ax.plot(x, A[:, j], '-', color=COLORS[robot], linewidth=1.1,
                label=LABEL[robot])
    ax.set_xlabel('Position along the route (m)')
    ax.set_ylabel('Agitation in the bin (deg)')
    ax.set_title('(b) where each platform is shaken', fontsize=9)
    ax.legend(fontsize=7)

    fig.tight_layout()
    if not os.path.isdir(a.output_dir):
        os.makedirs(a.output_dir)
    for ext in ('eps', 'png'):
        fig.savefig(os.path.join(a.output_dir, 'position_matched.%s' % ext),
                    dpi=200)
    plt.close(fig)
    print('')
    print('figura: %s/position_matched.{eps,png}' % a.output_dir)


if __name__ == '__main__':
    main()
