#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATE instantaneo medio contra agitacion y contra vibracion.

QUE PREGUNTA RESPONDE
=====================
Si el chasis va mas sacudido, la estimacion del SLAM es peor EN ESE INSTANTE?

Es distinto de lo que mira plot_agitation_heading.py, que correlaciona la
agitacion con el CRECIMIENTO del error dentro de la ventana - o sea, cuanto
error NUEVO se genero ahi.  Esa es la pregunta causal y la buena, pero es
ruidosa: el crecimiento por ventana son decimas de milimetro sobre un error
de decimetros.

Esta figura mira el NIVEL: cuanto vale el error mientras el chasis esta
sacudido.  Es mas facil de leer y responde a la pregunta que se hace primero
al ver los datos, pero OJO con interpretarla: un error alto y una agitacion
alta pueden coincidir sencillamente porque las dos cosas pasan en el mismo
sitio -la rampa del oeste, el escalon- sin que una cause la otra.  Por eso
las dos figuras van juntas y esta no sustituye a aquella.

EL PREDICTOR
============
  agitacion  hypot(std(cabeceo), std(balanceo)) dentro de la ventana: cuanto
             se MOVIO la actitud del chasis.  Desde el 2026-09-01 sale de
             gt_highrate.csv -verdad terreno a 1000 Hz- y no del IMU.

Por ventana de 100 ms, que es un barrido del LiDAR.  Es el mismo eje x que
usa agitation_vs_heading.py, para que las dos figuras se lean en paralelo.
La vibracion da aqui practicamente lo mismo (r 0.072 contra 0.075), asi que
no se dibuja; la comparacion entre predictores vive en la tabla de
correlaciones.

USO
===
    ./plot_ate_vs_attitude.py --results ~/metrics_output/fixed_trajectory \\
        --output_dir <dir>
"""
from __future__ import print_function

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
from metrics_io import load_metrics, load_gt_highrate    # noqa: E402
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

N_BINS = 10


def recoge(results_dir):
    """[(robot, run, ventanas)] de cuantas corridas haya."""
    out = []
    for robot in ORDER:
        for run in sorted(os.listdir(os.path.join(results_dir, robot))
                          if os.path.isdir(os.path.join(results_dir, robot))
                          else []):
            p = os.path.join(results_dir, robot, run, 'metrics.csv')
            if not os.path.isfile(p):
                continue
            df = trim_warmup(load_metrics(p))
            res = sm.evaluate(df, rpe_delta_m=1.0)
            if res is None:
                continue
            hr = load_gt_highrate(p)
            corr = sm.stability_error_correlation(
                df, res['ate_trans_series'], hr=hr,
                ate_rot=res.get('ate_rot_series'))
            if corr is None or not corr['windows']:
                continue
            out.append((robot, run, corr))
            print('  %-13s %-8s %5d ventanas de %.0f ms'
                  % (robot, run, len(corr['windows']),
                     corr['window_s'] * 1000.0))
    return out


def binned(x, y, n=N_BINS):
    """Media de y en deciles de x.  Devuelve (centros, medias, errores)."""
    x, y = np.asarray(x), np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < n * 2:
        return None, None, None
    bordes = np.percentile(x, np.linspace(0, 100, n + 1))
    bordes[-1] += 1e-9
    cx, cy, ce = [], [], []
    for k in range(n):
        m = (x >= bordes[k]) & (x < bordes[k + 1])
        if m.sum() < 3:
            continue
        cx.append(float(np.mean(x[m])))
        cy.append(float(np.mean(y[m])))
        ce.append(float(np.std(y[m]) / math.sqrt(m.sum())))
    return np.array(cx), np.array(cy), np.array(ce)


def panel(ax, datos, clave_x, clave_y, xlabel, ylabel, escala_x=1.0,
          escala_y=1.0, pct=98.0):
    # El eje x se recorta al percentil `pct` de los datos agrupados.  Sin
    # esto, unas pocas ventanas extremas -la rampa del oeste, el escalon-
    # estiraban el eje hasta 2.5 grados mientras el 98 % de los puntos vive
    # por debajo de 0.5, y los deciles quedaban apretados en el primer
    # quinto del grafico.  Los puntos de fuera no se borran: se salen del
    # recorte, que es lo honesto, pero dejan de mandar sobre la escala.
    todo_x = []
    for robot, run, corr in datos:
        w = corr['windows']
        x = np.array([r[clave_x] for r in w]) * escala_x
        y = np.array([r[clave_y] for r in w]) * escala_y
        todo_x.append(x[np.isfinite(x)])
        c = COLORS.get(robot, '#666666')
        # la nube cruda, muy tenue: sin ella los deciles enganan sobre
        # cuanta dispersion esconden
        ax.plot(x, y, '.', color=c, alpha=0.06, markersize=1.5,
                rasterized=True)
        bx, by, be = binned(x, y)
        if bx is None:
            continue
        ax.errorbar(bx, by, yerr=be, color=c, marker='o', markersize=3.5,
                    linewidth=1.2, capsize=2,
                    label='%s %s' % (LABEL.get(robot, robot), run))
        # Pearson sobre las ventanas crudas, no sobre los deciles
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 10:
            r = float(np.corrcoef(x[ok], y[ok])[0, 1])
            ax.annotate('r = %+.3f' % r, xy=(0.97, 0.05),
                        xycoords='axes fraction', ha='right', fontsize=7,
                        color=c)
    if todo_x:
        junto = np.concatenate(todo_x)
        if len(junto) > 20:
            hi = float(np.percentile(junto, pct))
            if hi > 0:
                ax.set_xlim(0.0, hi * 1.05)
                fuera = int((junto > hi).sum())
                if fuera:
                    ax.annotate('%d ventanas fuera de escala' % fuera,
                                xy=(0.97, 0.93), xycoords='axes fraction',
                                ha='right', fontsize=6, color='#666666')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--output_dir', required=True)
    a = ap.parse_args()

    print('leyendo corridas de %s' % a.results)
    datos = recoge(a.results)
    if not datos:
        print('no hay corridas utilizables'); return 1

    w_ms = datos[0][2]['window_s'] * 1000.0
    vent = ('%.0f ms' % w_ms) if w_ms < 1000 else ('%g s' % (w_ms / 1000.0))

    # Un solo predictor: la agitacion de actitud.  Es la misma magnitud que
    # usa agitation_vs_heading.py en su eje x, asi que las dos figuras se
    # leen en paralelo sin cambiar de unidades mentalmente.  La vibracion se
    # descarto porque aqui da practicamente lo mismo -r 0.072 contra 0.075
    # sobre el nivel del ATE- y una segunda columna casi identica solo ocupa
    # sitio.  Sigue en table_stability_correlation.txt, que es donde toca
    # compararlas.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ag = 'Attitude agitation per %s window (deg)' % vent

    panel(axes[0], datos, 'attitude_agitation_rad', 'ate_mean_m',
          ag, 'Mean instantaneous' + chr(10) + 'ATE translation (m)',
          escala_x=180.0 / math.pi)
    panel(axes[1], datos, 'attitude_agitation_rad', 'ate_rot_mean_deg',
          ag, 'Mean instantaneous' + chr(10) + 'ATE rotation (deg)',
          escala_x=180.0 / math.pi)

    axes[0].set_title('(a) translation', fontsize=9)
    axes[1].set_title('(b) rotation', fontsize=9)

    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc='upper center', ncol=min(3, len(l)),
                   fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Mean instantaneous ATE against chassis attitude '
                 'agitation', fontsize=10, y=1.12)
    fig.tight_layout()

    if not os.path.isdir(a.output_dir):
        os.makedirs(a.output_dir)
    for ext in ('png', 'eps'):
        fig.savefig(os.path.join(a.output_dir, 'ate_vs_attitude.%s' % ext),
                    dpi=200 if ext == 'png' else None)
    print('figura escrita en %s/ate_vs_attitude.{png,eps}' % a.output_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
