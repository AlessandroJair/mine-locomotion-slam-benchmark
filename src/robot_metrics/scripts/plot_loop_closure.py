#!/usr/bin/env python3
"""Figures for the loop-closure investigation of 2026-08-23/24.

Four figures, each answering one question:

  loop_closure_sweep     Which parameter actually closes the loop?
  loop_closure_ate       Online estimate against optimised graph.
  loop_closure_traj      What the correction looks like on the route.
  loop_closure_gaps      Revisits against local links, per platform.

The sweep values are measurements, not a live computation: they come from
reprocessing ONE two-lap rocker_bogie run with rtabmap-reprocess, so every
configuration saw byte-identical sensor data.  They are recorded in the table
below and in the comment above RGBD/ProximityMaxGraphDepth in each platform's
rtabmap_3d_slam.launch.  Re-measure with:

    rtabmap-reprocess --uwarn --RGBD/ProximityMaxGraphDepth 0 in.db out.db
    rtabmap-export --poses --poses_format 10 out.db

Usage:
    ./plot_loop_closure.py --results ~/metrics_output/fixed_trajectory \\
        --output_dir ~/metrics_output/fixed_trajectory/paper_tables
"""

import argparse
import csv
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slam_metrics as sm                 # noqa: E402

# Validated categorical slots 1-3 (light surface).  Three is the cap that
# clears the all-pairs colour-blind separation floors; a fourth series would
# put yellow next to orange and fail them.
C_BLUE, C_ORANGE, C_AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#d8d7d2'

# depth, radius, proximity detections, links spanning >1/2 lap, ATE rmse, max
SWEEP = [
    (50, 20,  60,   0, 0.792, 1.443),
    (0, 10,  86, 172, 0.177, 0.302),
    (0, 15, 121, 180, 0.171, 0.325),
    (0, 20, 150, 180, 0.170, 0.340),
]

PLATFORMS = [('rocker_bogie', 'Rocker-bogie'),
             ('tracked', 'Tracked'),
             ('differential', 'Husky (differential)')]


def style(ax):
    ax.set_facecolor('white')
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def save(fig, out_dir, name):
    # No tight_layout here: it runs after the caller's subplots_adjust and
    # undoes it, which put every caption on top of the tick labels.  Each
    # figure lays itself out and reserves its own caption band.
    for ext in ('png', 'eps'):
        p = os.path.join(out_dir, '%s.%s' % (name, ext))
        fig.savefig(p, dpi=200 if ext == 'png' else None,
                    facecolor='white')
    plt.close(fig)
    print('  %s.png / .eps' % name)


def load_tum(path):
    stamps, xy, mats = [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            M = np.eye(4)
            M[:3, :3] = sm.quat_to_rot(np.array([float(v) for v in p[4:8]]))
            M[:3, 3] = [float(p[1]), float(p[2]), float(p[3])]
            stamps.append(float(p[0]))
            xy.append((float(p[1]), float(p[2])))
            mats.append(M)
    return np.array(stamps), np.array(xy), np.array(mats)


def match(es, eT, gs, gT, tol=0.05):
    idx = np.clip(np.searchsorted(gs, es), 1, len(gs) - 1)
    idx = np.where(np.abs(es - gs[idx - 1]) < np.abs(es - gs[idx]),
                   idx - 1, idx)
    keep = np.abs(es - gs[idx]) < tol
    return keep, idx[keep]


def ate_of(run_dir, which):
    """ATE of est_traj.tum ('online') or opt_traj.tum ('optimised')."""
    f = {'online': 'est_traj.tum', 'optimised': 'opt_traj.tum'}[which]
    p, g = os.path.join(run_dir, f), os.path.join(run_dir, 'gt_traj.tum')
    if not (os.path.isfile(p) and os.path.isfile(g)):
        return None
    es, _, eT = load_tum(p)
    gs, _, gT = load_tum(g)
    if len(es) < 5 or len(gs) < 5:
        return None
    keep, gidx = match(es, eT, gs, gT)
    if keep.sum() < 5:
        return None
    t, _ = sm.ate(sm.align_umeyama(eT[keep], gT[gidx]), gT[gidx])
    return float(np.sqrt(np.mean(t ** 2))), float(np.max(t)), int(keep.sum())


# ---------------------------------------------------------------- figure 1
def fig_sweep(out_dir):
    """Two panels, never two y-axes on one: the measures differ in scale."""
    labels = ['depth %d\nradius %d m' % (d, r) for d, r, *_ in SWEEP]
    x = np.arange(len(SWEEP))
    ate = [s[4] for s in SWEEP]
    cross = [s[3] for s in SWEEP]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    ax = axes[0]
    style(ax)
    bars = ax.bar(x, ate, width=0.55, color=C_BLUE, zorder=3)
    bars[0].set_color(C_ORANGE)
    for xi, v in zip(x, ate):
        ax.text(xi, v + 0.02, '%.3f' % v, ha='center', va='bottom',
                fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('ATE RMSE, optimised graph (m)', fontsize=9, color=INK_2)
    ax.set_ylim(0, max(ate) * 1.25)
    ax.set_title('Accumulated error', fontsize=10, color=INK, loc='left')

    ax = axes[1]
    style(ax)
    bars = ax.bar(x, cross, width=0.55, color=C_AQUA, zorder=3)
    bars[0].set_color(C_ORANGE)
    for xi, v in zip(x, cross):
        ax.text(xi, v + 3, '%d' % v, ha='center', va='bottom',
                fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('links spanning > half a lap', fontsize=9, color=INK_2)
    ax.set_ylim(0, max(cross) * 1.22)
    ax.set_title('Links that can correct drift', fontsize=10, color=INK,
                 loc='left')

    fig.suptitle('RGBD/ProximityMaxGraphDepth is what closes the loop; '
                 'the radius is not', fontsize=11, color=INK, x=0.01,
                 ha='left')
    fig.text(0.012, 0.045,
             'One two-lap rocker_bogie run reprocessed per configuration, so '
             'the sensor data is identical in every bar.',
             fontsize=7.5, color=INK_2, ha='left')
    fig.text(0.012, 0.012,
             'Orange = the values shipped before 2026-08-23.',
             fontsize=7.5, color=INK_2, ha='left')
    fig.tight_layout()
    fig.subplots_adjust(top=0.80, bottom=0.32)
    save(fig, out_dir, 'loop_closure_sweep')


# ---------------------------------------------------------------- figure 2
def fig_ate(results, out_dir):
    names, online, opt, dist = [], [], [], []
    for key, label in PLATFORMS:
        r = results.get(key)
        if not r:
            continue
        names.append(label)
        online.append(r['online'][0] if r['online'] else np.nan)
        opt.append(r['optimised'][0] if r['optimised'] else np.nan)
        dist.append(r.get('distance', float('nan')))
    full = np.nanmax(dist) if len(dist) else float('nan')
    if not names:
        print('  (no runs found, skipping loop_closure_ate)')
        return

    x = np.arange(len(names))
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    style(ax)
    ax.bar(x - w / 2, online, w, label='Online estimate (est_traj.tum)',
           color=C_BLUE, zorder=3)
    ax.bar(x + w / 2, opt, w, label='Optimised graph (opt_traj.tum)',
           color=C_ORANGE, zorder=3)
    for xi, v in zip(x - w / 2, online):
        if np.isfinite(v):
            ax.text(xi, v + 0.02, '%.3f' % v, ha='center', va='bottom',
                    fontsize=8, color=INK)
    for xi, v in zip(x + w / 2, opt):
        if np.isfinite(v):
            ax.text(xi, v + 0.02, '%.3f' % v, ha='center', va='bottom',
                    fontsize=8, color=INK)
        else:
            ax.text(xi, 0.02, 'graph not\npreserved', ha='center',
                    va='bottom', fontsize=7, color=INK_2)
    # Distance covered goes under each platform.  A run that stopped early
    # has a smaller ATE for the trivial reason that it drove less, so the
    # label has to travel with the bar.
    ticks = []
    for label, d in zip(names, dist):
        if not np.isfinite(d):
            ticks.append(label)
        elif np.isfinite(full) and d < 0.9 * full:
            ticks.append('%s\n%.0f m of %.0f m - ABORTED' % (label, d, full))
        else:
            ticks.append('%s\n%.0f m, route completed' % (label, d))
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, fontsize=8.5)
    ax.set_ylabel('ATE translational RMSE (m)', fontsize=9, color=INK_2)
    ax.set_title('Loop closure is invisible in the online trajectory',
                 fontsize=11, color=INK, loc='left')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    vals = [v for v in online + opt if np.isfinite(v)]
    ax.set_ylim(0, max(vals) * 1.32)
    fig.text(0.012, 0.055,
             'Same run, same data: a loop closure rewrites the graph behind '
             'the pose already published, so the logged',
             fontsize=7.5, color=INK_2, ha='left')
    fig.text(0.012, 0.020,
             'online trajectory never receives the correction. Compare '
             'platforms only over equal distance.',
             fontsize=7.5, color=INK_2, ha='left')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    save(fig, out_dir, 'loop_closure_ate')


# ---------------------------------------------------------------- figure 3
def fig_traj(results, out_dir):
    r = results.get('rocker_bogie')
    if not r or r['gt_xy'] is None or r['opt_xy'] is None:
        print('  (no rocker_bogie trajectories, skipping loop_closure_traj)')
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    style(ax)
    gt, est, opt = r['gt_xy'], r['est_xy'], r['opt_xy']
    ax.plot(gt[:, 0], gt[:, 1], color=GRID, linewidth=3.0,
            label='Ground truth', zorder=2)
    if est is not None:
        ax.plot(est[:, 0], est[:, 1], color=C_BLUE, linewidth=1.6,
                label='Online estimate', zorder=3)
    ax.plot(opt[:, 0], opt[:, 1], color=C_ORANGE, linewidth=1.6,
            label='Optimised graph', zorder=4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('x (m)', fontsize=9, color=INK_2)
    ax.set_ylabel('y (m)', fontsize=9, color=INK_2)
    ax.set_title('Rocker-bogie, two laps of the mine loop',
                 fontsize=11, color=INK, loc='left')
    ax.legend(fontsize=8, frameon=False, loc='best')
    fig.text(0.01, 0.005,
             'Trajectories are Umeyama-aligned to ground truth, as the ATE '
             'is.',
             fontsize=7.5, color=INK_2, ha='left')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13)
    save(fig, out_dir, 'loop_closure_traj')


# ---------------------------------------------------------------- figure 4
def fig_gaps(results, out_dir):
    have = [(k, lab) for k, lab in PLATFORMS
            if results.get(k) and results[k]['gaps']]
    if not have:
        print('  (no events, skipping loop_closure_gaps)')
        return
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    style(ax)
    colors = [C_BLUE, C_ORANGE, C_AQUA]
    for i, (key, label) in enumerate(have):
        g = np.array(results[key]['gaps'])
        y = np.full(len(g), len(have) - 1 - i, dtype=float)
        y += np.linspace(-0.16, 0.16, len(g))
        ax.scatter(g, y, s=14, color=colors[i % 3], zorder=3,
                   edgecolors='white', linewidths=0.5)
        rev = int((g < 3.0).sum())
        ax.text(23.6, len(have) - 1 - i,
                '%d of %d are revisits' % (rev, len(g)),
                fontsize=8, color=INK, va='center', ha='right')
    ax.axvline(3.0, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=2)
    ax.text(3.3, len(have) - 0.42, 'revisit threshold, 3 m',
            fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(have)))
    ax.set_yticklabels([lab for _, lab in reversed(have)], fontsize=9)
    ax.set_xlabel('ground-truth distance to the linked node (m)',
                  fontsize=9, color=INK_2)
    ax.set_xlim(-0.6, 24)
    ax.set_ylim(-0.6, len(have) - 0.3)
    ax.set_title('What each proximity link actually connected',
                 fontsize=11, color=INK, loc='left')
    fig.text(0.012, 0.055,
             'Left of the line: the same place seen again - the links that '
             'correct accumulated drift.',
             fontsize=7.5, color=INK_2, ha='left')
    fig.text(0.012, 0.020,
             'Right: nearby nodes of the same corridor, which only stiffen '
             'the graph locally.',
             fontsize=7.5, color=INK_2, ha='left')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.34)
    save(fig, out_dir, 'loop_closure_gaps')


def read_run(run_dir):
    out = {'online': ate_of(run_dir, 'online'),
           'optimised': ate_of(run_dir, 'optimised'),
           'gaps': [], 'gt_xy': None, 'est_xy': None, 'opt_xy': None,
           'distance': float('nan')}

    # How far it actually got.  An ATE over a quarter of the route is not
    # comparable with one over the whole route, and without this the
    # differential's aborted 61 m run plots as the most accurate platform.
    meta = os.path.join(run_dir, 'run_meta.yaml')
    if os.path.isfile(meta):
        with open(meta) as fh:
            for line in fh:
                if line.startswith('gt_distance_m:'):
                    try:
                        out['distance'] = float(line.split(':', 1)[1])
                    except ValueError:
                        pass

    ev = os.path.join(run_dir, 'events.csv')
    if os.path.isfile(ev):
        with open(ev) as fh:
            for row in csv.DictReader(fh):
                if row.get('event[-]') != 'proximity_detection':
                    continue
                m = re.search(r'gt_gap=([0-9.]+)', row.get('detail[-]', ''))
                if m:
                    out['gaps'].append(float(m.group(1)))

    g = os.path.join(run_dir, 'gt_traj.tum')
    o = os.path.join(run_dir, 'opt_traj.tum')
    e = os.path.join(run_dir, 'est_traj.tum')
    if os.path.isfile(g) and os.path.isfile(o):
        gs, _, gT = load_tum(g)
        os_, _, oT = load_tum(o)
        keep, gidx = match(os_, oT, gs, gT)
        if keep.sum() >= 5:
            G = gT[gidx]
            out['gt_xy'] = G[:, :2, 3]
            out['opt_xy'] = sm.align_umeyama(oT[keep], G)[:, :2, 3]
        if os.path.isfile(e):
            es, _, eT = load_tum(e)
            k2, g2 = match(es, eT, gs, gT)
            if k2.sum() >= 5:
                out['est_xy'] = sm.align_umeyama(eT[k2], gT[g2])[:, :2, 3]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--run', default='run01')
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    results = {}
    for key, label in PLATFORMS:
        d = os.path.join(a.results, key, a.run)
        if os.path.isdir(d):
            results[key] = read_run(d)
            r = results[key]
            print('%-14s online %-22s optimised %-22s %d proximity events'
                  % (key,
                     ('%.3f m' % r['online'][0]) if r['online'] else 'n/a',
                     ('%.3f m' % r['optimised'][0]) if r['optimised']
                     else 'graph not preserved',
                     len(r['gaps'])))

    print('\nfigures written to %s' % a.output_dir)
    fig_sweep(a.output_dir)
    fig_ate(results, a.output_dir)
    fig_traj(results, a.output_dir)
    fig_gaps(results, a.output_dir)


if __name__ == '__main__':
    main()
