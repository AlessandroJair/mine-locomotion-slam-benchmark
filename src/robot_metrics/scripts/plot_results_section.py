#!/usr/bin/env python3
"""Figures for the Results section that the campaign does not draw.

    dynamic_profiles_step     proper acceleration (body z and x) and pitch/roll
                              of every platform across the step, first pass of
                              run01, from gt_highrate (1 kHz), against distance
    attitude_distribution     boxplots of pitch, roll and their rates, all runs
    trajectories_gt_vs_slam   plan view per platform, run01: ground truth,
                              online odometry and optimised graph, aligned
                              exactly as the ATE that is quoted for them

It also prints the numbers the text quotes: the crossing peaks, the attitude
statistics and the window sweep drawn in rpe_corr_vs_window.

Everything goes through the same functions as aggregate_runs.py (evaluate_run,
_barrido_ventanas, the alignments of slam_metrics), so a number in the text and
a number in a table cannot come from two definitions.

Usage:
    ./plot_results_section.py --results ~/metrics_final_version/fixed_trajectory/swept_lidar
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate_runs as ag                               # noqa: E402
import slam_metrics as sm                                 # noqa: E402
from plot_wheel_contact import load_layout                # noqa: E402

import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.colors import to_rgb                      # noqa: E402

ORDER = ['differential', 'tracked', 'rocker_bogie']
STATS = ('pitch', 'roll', 'pitch_rate', 'roll_rate')


def short(robot):
    return ag.DISPLAY_NAME.get(robot, robot).split(' ')[0]


def first_crossing(hr, step, margin):
    """(samples of the first pass within `margin` m of the slab, (i_on, i_off)).

    Not plot_wheel_contact.find_step_window: that one returns first..last
    sample, and on a two-lap run that spans the whole lap in between.
    """
    cx, cy = step['pose_xyzrpy'][0], step['pose_xyzrpy'][1]
    # slab rotated 1.57 rad about z: size_m[1] lies along world x
    half_x, half_y = step['size_m'][1] / 2.0, step['size_m'][0] / 2.0
    x, y, t = hr['gt_x'].values, hr['gt_y'].values, hr['timestamp'].values

    def inside(m):
        return (np.abs(x - cx) <= half_x + m) & (np.abs(y - cy) <= half_y + m)

    near = np.where(inside(margin))[0]
    if near.size == 0:
        return None, None
    salida = np.where(np.diff(t[near]) > 5.0)[0]     # left and came back later
    if salida.size:
        near = near[:salida[0] + 1]
    on = near[inside(0.0)[near]]
    return near, ((on[0], on[-1]) if on.size else None)


def rate_deg_s(t, a):
    """Finite-difference rate, skipping the gaps gt_highrate has."""
    dt, da = np.diff(t), np.diff(np.unwrap(a))
    ok = (dt > 0) & (dt <= 0.0025)
    return np.degrees(da[ok] / dt[ok])


def profile(res, step, margin):
    hr = res['hr']
    near, span = first_crossing(hr, step, margin)
    if near is None or span is None:
        return None
    t = hr['timestamp'].values
    df = res['df']
    d = np.interp(t[near], df['timestamp'].values, df['gt_distance'].values)
    d0 = np.interp(t[span[0]], df['timestamp'].values, df['gt_distance'].values)
    d1 = np.interp(t[span[1]], df['timestamp'].values, df['gt_distance'].values)
    pitch = np.degrees(hr['gt_pitch'].values[near])
    roll = np.degrees(hr['gt_roll'].values[near])
    return {
        'x': d - d0, 'slab_len': d1 - d0,
        'az': hr['gt_az'].values[near], 'ax': hr['gt_ax'].values[near],
        'pitch': pitch, 'roll': roll,
        'pitch_rate': rate_deg_s(t[near], hr['gt_pitch'].values[near]),
        'roll_rate': rate_deg_s(t[near], hr['gt_roll'].values[near]),
        'duration_s': t[near][-1] - t[near][0],
    }


def plot_profiles(robots, prof, out_dir):
    rows = [('az', r'$a_z$ body (m/s$^2$)'), ('ax', r'$a_x$ body (m/s$^2$)'),
            ('pitch', r'Pitch $\theta$ (deg)'), ('roll', r'Roll $\phi$ (deg)')]
    fig, axes = plt.subplots(len(rows), 1, figsize=(ag.COLUMN_WIDTH_IN, 6.4),
                             sharex=True)
    largo = np.mean([prof[r]['slab_len'] for r in robots if r in prof])
    for ax, (key, label) in zip(axes, rows):
        ax.axvspan(0.0, largo, color='0.85', zorder=0)
        for i, robot in enumerate(robots):
            p = prof.get(robot)
            if p is None:
                continue
            ax.plot(p['x'], p[key], ag.LINE_STYLES[i % 3], color=ag.COLORS[i % 3],
                    linewidth=0.7, label=ag.DISPLAY_NAME.get(robot, robot))
        ax.set_ylabel(label)
    axes[0].legend(loc='upper left', fontsize=7, ncol=3)
    axes[-1].set_xlabel('Distance from the step edge along the path (m); '
                        'shaded: base over the 0.10 m slab')
    fig.tight_layout()
    ag.save(fig, out_dir, 'dynamic_profiles_step')


def plot_distribution(robots, stats, out_dir):
    panels = [('pitch', r'Pitch $\theta$ (deg)'), ('roll', r'Roll $\phi$ (deg)'),
              ('pitch_rate', r'$\dot{\theta}$ (deg/s)'),
              ('roll_rate', r'$\dot{\phi}$ (deg/s)')]
    fig, axes = plt.subplots(1, len(panels), figsize=(ag.COLUMN_WIDTH_IN, 2.7))
    for ax, (key, title) in zip(axes, panels):
        b = ax.boxplot([stats[r][key] for r in robots], whis=(1, 99),
                       showfliers=False, widths=0.55, patch_artist=True,
                       medianprops={'color': 'black', 'linewidth': 1.0})
        for patch, c in zip(b['boxes'], ag.COLORS):
            # solid tint, not alpha: the PostScript backend drops alpha
            patch.set_facecolor([1.0 - 0.65 * (1.0 - v) for v in to_rgb(c)])
        ax.set_xticks(range(1, len(robots) + 1))
        ax.set_xticklabels([short(r) for r in robots], rotation=35, ha='right',
                           fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
    fig.tight_layout()
    ag.save(fig, out_dir, 'attitude_distribution')


def plan(res):
    """GT, online odometry and optimised graph in the plane, shifted so the
    ground truth starts at the origin.  Same alignments as the quoted ATEs."""
    df = res['df']
    T_gt = sm.poses_from_rpy(df['gt_x'].values, df['gt_y'].values,
                             df['gt_z'].values, df['gt_roll'].values,
                             df['gt_pitch'].values, df['gt_yaw'].values)
    T_est = sm.poses_from_rpy(df['odom_x'].values, df['odom_y'].values,
                              df['odom_z'].values, df['odom_roll'].values,
                              df['odom_pitch'].values, df['odom_yaw'].values)
    T_gt = sm.correct_body_frame(T_gt, sm.heading_offset(T_gt))
    T_est = sm.correct_body_frame(T_est, sm.heading_offset(T_est))
    T_org = sm.align_origin(T_est, T_gt)
    p0 = T_gt[0, :2, 3]
    out = {'gt': T_gt[:, :2, 3] - p0, 'odom': T_org[:, :2, 3] - p0, 'opt': None}

    run_dir = res['run_dir']
    opt, gt = (os.path.join(run_dir, f) for f in ('opt_traj.tum', 'gt_traj.tum'))
    if os.path.isfile(opt) and os.path.isfile(gt):
        es, eT = ag._load_tum(opt)
        gs, gT = ag._load_tum(gt)
        idx = np.clip(np.searchsorted(gs, es), 1, len(gs) - 1)
        idx = np.where(np.abs(es - gs[idx - 1]) < np.abs(es - gs[idx]), idx - 1, idx)
        keep = np.abs(es - gs[idx]) < 0.05
        if keep.sum() >= 5:
            T = sm.align_umeyama(eT[keep], gT[idx[keep]])
            out['opt'] = T[:, :2, 3] - p0
    return out


def plot_trajectories(robots, first, out_dir):
    fig, axes = plt.subplots(1, len(robots), figsize=(ag.COLUMN_WIDTH_IN, 2.9),
                             sharex=True, sharey=True)
    for i, (ax, robot) in enumerate(zip(np.atleast_1d(axes), robots)):
        res = first[robot]
        p = plan(res)
        ax.plot(p['gt'][:, 0], p['gt'][:, 1], color='black', linewidth=1.1,
                label='Ground truth')
        ax.plot(p['odom'][:, 0], p['odom'][:, 1], '--', color=ag.COLORS[i % 3],
                linewidth=0.9, label='Online odometry (%.2f m)'
                % res['ate_origin_trans_rmse'])
        if p['opt'] is not None:
            ax.plot(p['opt'][:, 0], p['opt'][:, 1], ':', color='0.45',
                    linewidth=1.1, label='Optimised graph (%.2f m)'
                    % res['ate_opt_trans_rmse'])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(ag.DISPLAY_NAME.get(robot, robot), fontsize=8)
        ax.set_xlabel('$x$ (m)')
        ax.legend(fontsize=5.5, loc='center')
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel('$y$ (m)')
    fig.tight_layout()
    ag.save(fig, out_dir, 'trajectories_gt_vs_slam')


def print_numbers(robots, prof, stats, sweeps):
    print('\n=== STEP CROSSING (run01, first pass) ===')
    for r in robots:
        p = prof.get(r)
        if p is None:
            print('%-14s no crossing found' % short(r))
            continue
        print('%-14s |a_z|max %6.2f  |a_x|max %6.2f  pitch [%6.2f, %6.2f]  '
              'roll [%6.2f, %6.2f]  |th_dot|max %7.1f  |ph_dot|max %7.1f  '
              'slab %.2f m  window %.1f s'
              % (short(r), np.max(np.abs(p['az'])), np.max(np.abs(p['ax'])),
                 p['pitch'].min(), p['pitch'].max(), p['roll'].min(),
                 p['roll'].max(), np.max(np.abs(p['pitch_rate'])),
                 np.max(np.abs(p['roll_rate'])), p['slab_len'], p['duration_s']))

    print('\n=== ATTITUDE DISTRIBUTION (all runs, gt_highrate) ===')
    print('%-14s %-10s %8s %8s %8s %8s %8s %8s'
          % ('robot', 'signal', 'median', 'p25', 'p75', 'p1', 'p99', 'std'))
    for r in robots:
        for k in STATS:
            v = stats[r][k]
            q = np.percentile(v, [50, 25, 75, 1, 99])
            print('%-14s %-10s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f'
                  % ((short(r), k) + tuple(q) + (np.std(v),)))

    print('\n=== WINDOW SWEEP: mean over runs of r and r|velocity ===')
    pares = [('vibration_rms_m_s2', 'rpe_rot_mean_deg_m', 'vib->rot'),
             ('vibration_rms_m_s2', 'rpe_trans_mean_m_m', 'vib->trans'),
             ('attitude_agitation_rad', 'rpe_rot_mean_deg_m', 'agit->rot'),
             ('attitude_agitation_rad', 'rpe_trans_mean_m_m', 'agit->trans')]
    for r in robots:
        for w in sorted(sweeps[r]):
            cs = sweeps[r][w]
            cells = []
            for pred, resp, nombre in pares:
                rr = np.nanmean([c.get('r_%s__%s' % (pred, resp)) for c in cs])
                pr = np.nanmean([c.get('pr_%s__%s' % (pred, resp)) for c in cs])
                cells.append('%s %.3f|%.3f' % (nombre, rr, pr))
            print('%-14s %5.0f ms  %s' % (short(r), w * 1000, '  '.join(cells)))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True)
    ap.add_argument('--output_dir', default=None,
                    help='default: <results>/paper_tables')
    ap.add_argument('--config',
                    default=os.path.join(here, '..', 'config', 'sim_config.yaml'))
    ap.add_argument('--margin', type=float, default=3.0,
                    help='metres of approach/exit around the step')
    args = ap.parse_args()
    out_dir = args.output_dir or os.path.join(args.results, 'paper_tables')

    runs = ag.discover(args.results)
    robots = [r for r in ORDER if r in runs]
    step = load_layout(args.config, robots[0])[1]

    prof, first, sweeps = {}, {}, {}
    stats = {r: {k: [] for k in STATS} for r in robots}
    for robot in robots:
        sweeps[robot] = {}
        for run_dir in runs[robot]:
            print('evaluating %s' % run_dir)
            res = ag.evaluate_run(run_dir, 1.0)
            if res is None or res.get('hr') is None:
                print('  skipped: no metrics or no gt_highrate')
                continue
            hr = res['hr']
            t = hr['timestamp'].values
            keep = (t - t[0]) >= ag.WARMUP_S
            stats[robot]['pitch'].append(np.degrees(hr['gt_pitch'].values[keep]))
            stats[robot]['roll'].append(np.degrees(hr['gt_roll'].values[keep]))
            stats[robot]['pitch_rate'].append(
                rate_deg_s(t[keep], hr['gt_pitch'].values[keep]))
            stats[robot]['roll_rate'].append(
                rate_deg_s(t[keep], hr['gt_roll'].values[keep]))
            for w, cs in ag._barrido_ventanas({robot: [res]}, robot).items():
                sweeps[robot].setdefault(w, []).extend(cs)
            if robot not in first:
                prof[robot] = profile(res, step, args.margin)
                first[robot] = res
            res['hr'] = None          # 1 kHz frames are large; keep RAM bounded
            del hr
        for k in STATS:
            stats[robot][k] = np.concatenate(stats[robot][k])

    prof = {r: p for r, p in prof.items() if p is not None}
    plot_profiles(robots, prof, out_dir)
    plot_distribution(robots, stats, out_dir)
    plot_trajectories(robots, first, out_dir)
    print_numbers(robots, prof, stats, sweeps)
    print('\nwritten to %s' % out_dir)


if __name__ == '__main__':
    main()
