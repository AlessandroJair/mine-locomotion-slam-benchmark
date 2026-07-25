#!/usr/bin/env python3
"""Aggregate the repeated runs of each platform into the paper's tables.

The reviewer asked for repeatability: three runs per robot, each reported
individually, and a mean +/- standard deviation across them.  This script
produces exactly that, plus the per-run detail that lets a reader see whether
the spread comes from one bad run or from genuine variability.

INPUT layout, as written by metrics_logger.py:

    <results>/<robot>/run01/metrics.csv, gt_traj.tum, events.csv, run_meta.yaml
    <results>/<robot>/run02/...
    <results>/<robot>/run03/...

OUTPUT (into --output_dir):

    per_run_metrics.csv        one row per (robot, run), every metric, with units
    summary_mean_std.csv       one row per (robot, metric): mean, std, n
    table_repeatability.txt    the formatted table for the paper
    table_slam.txt             SLAM quality table, mean +/- std
    ate_vs_time.eps/.png       instantaneous ATE against time, all runs
    ate_vs_distance.eps/.png   instantaneous ATE against distance travelled
    stability_vs_error.eps     windowed attitude agitation vs SLAM error

Usage:
    ./aggregate_runs.py --results ~/metrics_output --output_dir ~/paper_tables
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics_io import load_metrics                      # noqa: E402
import slam_metrics as sm                                # noqa: E402

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402

COLUMN_WIDTH_IN = 6.5
COLORS = ['#1f77b4', '#d62728', '#2ca02c']
LINE_STYLES = ['-', '--', '-.']
MARKERS = ['o', 's', '^']

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'lines.linewidth': 1.2, 'axes.grid': True,
    'grid.color': '0.85', 'grid.linewidth': 0.4,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

DISPLAY_NAME = {
    'differential': 'Husky (differential)',
    'husky': 'Husky (differential)',
    'tracked': 'Tracked',
    'rocker_bogie': 'Rocker-bogie',
}
ROBOT_ORDER = ['differential', 'husky', 'tracked', 'rocker_bogie']

WARMUP_S = 5.0

# metric key -> (label, unit, format)
REPORTED = [
    ('ate_origin_trans_rmse', 'ATE translational RMSE', 'm', '.4f'),
    ('ate_origin_trans_max', 'ATE translational max', 'm', '.4f'),
    ('ate_origin_rot_rmse_deg', 'ATE rotational RMSE', 'deg', '.3f'),
    ('ate_origin_rot_max_deg', 'ATE rotational max', 'deg', '.3f'),
    ('ate_umeyama_trans_rmse', 'ATE trans. RMSE (aligned)', 'm', '.4f'),
    ('rpe_trans_rmse', 'RPE translational RMSE', 'm/m', '.4f'),
    ('rpe_rot_rmse_deg', 'RPE rotational RMSE', 'deg/m', '.4f'),
    ('final_drift_m', 'Final drift', 'm', '.4f'),
    ('drift_pct_of_distance', 'Drift / distance', '%', '.3f'),
    ('gt_distance_m', 'Distance travelled (GT)', 'm', '.2f'),
    ('duration_s', 'Run duration', 's', '.1f'),
    ('tracking_losses', 'Tracking losses', 'count', '.0f'),
    ('relocalizations', 'Relocalizations', 'count', '.0f'),
    ('loop_closures_total', 'Loop closures reported', 'count', '.0f'),
    ('loop_closures_true', 'Loop closures true pos.', 'count', '.0f'),
    ('loop_closures_false', 'Loop closures false pos.', 'count', '.0f'),
    ('pitch_std_deg', 'Pitch std', 'deg', '.3f'),
    ('roll_std_deg', 'Roll std', 'deg', '.3f'),
    ('pitch_max_deg', 'Pitch max abs', 'deg', '.3f'),
    ('vibration_rms', 'Vibration RMS', 'm/s^2', '.4f'),
    ('accel_z_max', 'Peak |a_z - g|', 'm/s^2', '.4f'),
]

GRAVITY = 9.81


# ---------------------------------------------------------------------------

def discover(results_dir):
    """{robot: [run_dir, ...]} sorted by run index."""
    runs = {}
    for meta in sorted(glob.glob(os.path.join(results_dir, '*', 'run*',
                                              'metrics.csv'))):
        run_dir = os.path.dirname(meta)
        robot = os.path.basename(os.path.dirname(run_dir))
        runs.setdefault(robot, []).append(run_dir)
    for r in runs:
        runs[r].sort()
    return runs


def trim_warmup(df):
    """Drop the spawn transient, identically for every robot."""
    t = df['timestamp'].values
    t = t - t[0]
    if t[-1] > 2 * WARMUP_S:
        df = df[t >= WARMUP_S].reset_index(drop=True)
    return df


def read_events(run_dir):
    path = os.path.join(run_dir, 'events.csv')
    out = {'tracking_losses': 0, 'relocalizations': 0,
           'loop_closures_total': 0, 'loop_closures_true': 0,
           'loop_closures_false': 0, 'loop_closures_unknown': 0,
           'events': []}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            kind = row.get('event[-]', '')
            detail = row.get('detail[-]', '')
            out['events'].append(row)
            if kind == 'tracking_loss':
                out['tracking_losses'] += 1
            elif kind == 'relocalization':
                out['relocalizations'] += 1
            elif kind == 'loop_closure':
                out['loop_closures_total'] += 1
                if 'verdict=true' in detail:
                    out['loop_closures_true'] += 1
                elif 'verdict=false' in detail:
                    out['loop_closures_false'] += 1
                else:
                    out['loop_closures_unknown'] += 1
    return out


def locomotion_metrics(df):
    pitch = df['pitch'].values
    roll = df['roll'].values
    ax = df['accel_x'].values
    ay = df['accel_y'].values if 'accel_y' in df.columns else np.zeros(len(df))
    az = df['accel_z'].values - GRAVITY
    return {
        'pitch_std_deg': float(np.degrees(np.std(pitch))),
        'roll_std_deg': float(np.degrees(np.std(roll))),
        'pitch_max_deg': float(np.degrees(np.max(np.abs(pitch)))),
        'roll_max_deg': float(np.degrees(np.max(np.abs(roll)))),
        'vibration_rms': float(np.sqrt(np.mean(ax ** 2 + ay ** 2 + az ** 2))),
        'accel_z_max': float(np.max(np.abs(az))),
        'accel_x_rms': float(np.sqrt(np.mean(ax ** 2))),
    }


def evaluate_run(run_dir, rpe_delta_m):
    df = trim_warmup(load_metrics(os.path.join(run_dir, 'metrics.csv')))
    if len(df) < 20:
        return None

    res = sm.evaluate(df, rpe_delta_m=rpe_delta_m)
    if res is None:
        return None

    # Radians are the natural unit internally; the tables report degrees.
    for k in list(res.keys()):
        if k.endswith(('_rot_mean', '_rot_rmse', '_rot_max')):
            res[k + '_deg'] = float(np.degrees(res[k]))

    res.update(locomotion_metrics(df))
    res.update(read_events(run_dir))
    res['duration_s'] = float(df['timestamp'].values[-1] -
                              df['timestamp'].values[0])
    res['run_dir'] = run_dir
    res['run_label'] = os.path.basename(run_dir)
    res['corr'] = sm.stability_error_correlation(df, res['ate_trans_series'])
    res['df'] = df
    return res


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def fmt(value, spec):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return 'n/a'
    return ('{:' + spec + '}').format(value)


def per_run_table(robots, results, out_path):
    lines = []
    lines.append('=' * 100)
    lines.append('PER-RUN RESULTS (each repetition reported separately)')
    lines.append('=' * 100)
    for robot in robots:
        runs = results[robot]
        lines.append('')
        lines.append('{}  ({} runs)'.format(DISPLAY_NAME.get(robot, robot),
                                            len(runs)))
        header = '{:<32s}{:>10s}'.format('Metric [unit]', '')
        header = '{:<34s}'.format('Metric [unit]') + ''.join(
            '{:>12s}'.format(r['run_label']) for r in runs)
        lines.append('-' * len(header))
        lines.append(header)
        lines.append('-' * len(header))
        for key, label, unit, spec in REPORTED:
            vals = [r.get(key) for r in runs]
            if all(v is None for v in vals):
                continue
            lbl = '{} [{}]'.format(label, unit)
            lines.append('{:<34s}'.format(lbl) +
                         ''.join('{:>12s}'.format(fmt(v, spec)) for v in vals))
        lines.append('-' * len(header))

    text = '\n'.join(lines)
    with open(out_path, 'w') as f:
        f.write(text + '\n')
    return text


def mean_std(values):
    v = np.array([x for x in values
                  if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float('nan'), float('nan'), 0
    # Sample standard deviation (ddof=1): with n=3 repetitions the population
    # form would understate the spread.
    return (float(v.mean()),
            float(v.std(ddof=1)) if v.size > 1 else 0.0,
            int(v.size))


def summary_table(robots, results, out_path):
    lines = []
    header = '{:<34s}'.format('Metric [unit]') + ''.join(
        '{:>26s}'.format(DISPLAY_NAME.get(r, r)) for r in robots)
    lines.append('=' * len(header))
    lines.append('REPEATABILITY: mean +/- standard deviation over the runs')
    lines.append('=' * len(header))
    lines.append(header)
    lines.append('-' * len(header))

    rows_csv = []
    for key, label, unit, spec in REPORTED:
        cells = []
        any_value = False
        for robot in robots:
            m, s, n = mean_std([r.get(key) for r in results[robot]])
            if n == 0:
                cells.append('{:>26s}'.format('n/a'))
                continue
            any_value = True
            cells.append('{:>26s}'.format(
                '{} +/- {}  (n={})'.format(fmt(m, spec), fmt(s, spec), n)))
            rows_csv.append([DISPLAY_NAME.get(robot, robot), label, unit,
                             m, s, n])
        if any_value:
            lines.append('{:<34s}'.format('{} [{}]'.format(label, unit)) +
                         ''.join(cells))

    lines.append('=' * len(header))
    text = '\n'.join(lines)
    with open(out_path, 'w') as f:
        f.write(text + '\n')
    return text, rows_csv


def correlation_table(robots, results, out_path):
    """The stability -> SLAM performance evidence."""
    lines = []
    lines.append('=' * 96)
    lines.append('CHASSIS STABILITY vs INSTANTANEOUS SLAM ERROR')
    lines.append('=' * 96)
    lines.append('Pearson r between windowed attitude agitation and windowed')
    lines.append('SLAM error, with a 95% percentile-bootstrap CI (2000 resamples).')
    lines.append('Predictor: std of pitch and roll inside a 1 s window [rad].')
    lines.append('Response : mean instantaneous ATE inside the same window [m].')
    lines.append('')
    header = ('{:<24s}{:<8s}{:>10s}{:>22s}{:>10s}'
              .format('Robot', 'run', 'r', '95% CI', 'windows'))
    lines.append(header)
    lines.append('-' * len(header))

    pooled = {}
    for robot in robots:
        rs = []
        for r in results[robot]:
            c = r.get('corr')
            if not c:
                continue
            key = 'r_attitude_agitation_rad__ate_mean_m'
            ci = c.get('ci_attitude_agitation_rad__ate_mean_m', (np.nan, np.nan))
            rval = c.get(key, float('nan'))
            rs.append(rval)
            lines.append('{:<24s}{:<8s}{:>10s}{:>22s}{:>10d}'.format(
                DISPLAY_NAME.get(robot, robot), r['run_label'],
                fmt(rval, '.3f'),
                '[{}, {}]'.format(fmt(ci[0], '.3f'), fmt(ci[1], '.3f')),
                c.get('n_windows', 0)))
        if rs:
            m, s, n = mean_std(rs)
            pooled[robot] = (m, s, n)
            lines.append('{:<24s}{:<8s}{:>10s}{:>22s}{:>10s}'.format(
                '', 'mean', fmt(m, '.3f'),
                '+/- {} (n={})'.format(fmt(s, '.3f'), n), ''))
            lines.append('')

    lines.append('=' * 96)
    text = '\n'.join(lines)
    with open(out_path, 'w') as f:
        f.write(text + '\n')
    return text


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def save(fig, out_dir, name):
    for ext in ('eps', 'png'):
        fig.savefig(os.path.join(out_dir, '{}.{}'.format(name, ext)), dpi=200)
    plt.close(fig)


def plot_ate_vs(robots, results, out_dir, against='time'):
    """Instantaneous ATE against time or against distance travelled.

    The aggregate RMSE hides where the error is produced.  Plotting the
    instantaneous value shows whether it grows steadily (odometry drift) or
    jumps at specific places (the ramp and the step).
    """
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 3.2))
    for i, robot in enumerate(robots):
        for k, r in enumerate(results[robot]):
            err = r['ate_trans_series']
            if against == 'time':
                t = r['df']['timestamp'].values
                x = t - t[0]
            else:
                x = r['distance_series'][:len(err)]
            ax.plot(x[:len(err)], err[:len(x)],
                    ls=LINE_STYLES[k % 3], color=COLORS[i % 3],
                    linewidth=1.0, alpha=0.9,
                    label=DISPLAY_NAME.get(robot, robot) if k == 0 else None)
    ax.set_xlabel('Time (s)' if against == 'time'
                  else 'Distance travelled (m)')
    ax.set_ylabel('Instantaneous ATE (m)')
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir, 'ate_vs_' + against)


def plot_stability_vs_error(robots, results, out_dir):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 3.4))
    for i, robot in enumerate(robots):
        xs, ys = [], []
        for r in results[robot]:
            c = r.get('corr')
            if not c:
                continue
            xs += [np.degrees(w['attitude_agitation_rad']) for w in c['windows']]
            ys += [w['ate_mean_m'] for w in c['windows']]
        if not xs:
            continue
        ax.scatter(xs, ys, s=9, alpha=0.35, color=COLORS[i % 3],
                   marker=MARKERS[i % 3], linewidths=0,
                   label=DISPLAY_NAME.get(robot, robot))
        if len(xs) > 2:
            k, b = np.polyfit(np.array(xs), np.array(ys), 1)
            xf = np.linspace(min(xs), max(xs), 50)
            ax.plot(xf, k * xf + b, color=COLORS[i % 3], linewidth=1.2)
    ax.set_xlabel(r'Attitude agitation per 1 s window, '
                  r'$\sqrt{\sigma_\theta^2+\sigma_\phi^2}$ (deg)')
    ax.set_ylabel('Mean instantaneous ATE (m)')
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir, 'stability_vs_error')


def plot_run_spread(robots, results, out_dir):
    """Bar chart of mean +/- std across runs: the repeatability figure."""
    keys = [('ate_origin_trans_rmse', 'ATE trans. RMSE (m)'),
            ('ate_origin_rot_rmse_deg', 'ATE rot. RMSE (deg)'),
            ('rpe_trans_rmse', 'RPE trans. RMSE (m/m)'),
            ('pitch_std_deg', r'Pitch $\sigma$ (deg)')]
    fig, axes = plt.subplots(1, len(keys), figsize=(COLUMN_WIDTH_IN, 2.6))
    x = np.arange(len(robots))
    for ax, (key, title) in zip(axes, keys):
        means, stds = [], []
        for robot in robots:
            m, s, _ = mean_std([r.get(key) for r in results[robot]])
            means.append(m)
            stds.append(s)
        ax.bar(x, means, yerr=stds, capsize=3, width=0.55,
               color=COLORS[:len(robots)], edgecolor='black', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAME.get(r, r).split(' ')[0]
                            for r in robots], rotation=35, ha='right',
                           fontsize=7)
        ax.set_title(title, fontsize=8, pad=4)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, axis='y')
    fig.tight_layout()
    save(fig, out_dir, 'repeatability_summary')


# ---------------------------------------------------------------------------

def write_per_run_csv(robots, results, path):
    keys = [k for k, _, _, _ in REPORTED]
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['robot', 'run'] +
                   ['{}[{}]'.format(k, u)
                    for k, _, u, _ in REPORTED])
        for robot in robots:
            for r in results[robot]:
                w.writerow([DISPLAY_NAME.get(robot, robot), r['run_label']] +
                           [r.get(k, '') for k in keys])


def write_summary_csv(rows, path):
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['robot', 'metric', 'unit', 'mean', 'std', 'n_runs'])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=os.path.expanduser('~/metrics_output'),
                    help='directory holding <robot>/run<NN>/ subdirectories')
    ap.add_argument('--output_dir', default='./paper_tables')
    ap.add_argument('--rpe_delta_m', type=float, default=1.0,
                    help='travelled distance defining an RPE pair (m)')
    args = ap.parse_args()

    runs = discover(args.results)
    if not runs:
        print('No runs found under {}.'.format(args.results))
        print('Expected <results>/<robot>/run<NN>/metrics.csv - run the '
              'campaign first (run_campaign.sh).')
        return 1

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    results = {}
    for robot, run_dirs in runs.items():
        evald = []
        for d in run_dirs:
            r = evaluate_run(d, args.rpe_delta_m)
            if r is None:
                print('  skipping {} (too little data)'.format(d))
                continue
            evald.append(r)
        if evald:
            results[robot] = evald

    if not results:
        print('No usable runs.')
        return 1

    robots = [r for r in ROBOT_ORDER if r in results]
    robots += [r for r in sorted(results) if r not in robots]

    print('Runs found:')
    for robot in robots:
        print('  {:<16s} {} run(s)'.format(robot, len(results[robot])))
        if len(results[robot]) < 3:
            print('    WARNING: fewer than 3 repetitions; the standard '
                  'deviation reported for this platform is not meaningful.')

    t1 = per_run_table(robots, results,
                       os.path.join(args.output_dir, 'table_per_run.txt'))
    t2, rows_csv = summary_table(
        robots, results, os.path.join(args.output_dir,
                                      'table_repeatability.txt'))
    t3 = correlation_table(robots, results,
                           os.path.join(args.output_dir,
                                        'table_stability_correlation.txt'))
    print(t1)
    print(t2)
    print(t3)

    write_per_run_csv(robots, results,
                      os.path.join(args.output_dir, 'per_run_metrics.csv'))
    write_summary_csv(rows_csv,
                      os.path.join(args.output_dir, 'summary_mean_std.csv'))

    plot_ate_vs(robots, results, args.output_dir, 'time')
    plot_ate_vs(robots, results, args.output_dir, 'distance')
    plot_stability_vs_error(robots, results, args.output_dir)
    plot_run_spread(robots, results, args.output_dir)

    print('\nWritten to {}:'.format(os.path.abspath(args.output_dir)))
    for f in sorted(os.listdir(args.output_dir)):
        print('  - {}'.format(f))
    return 0


if __name__ == '__main__':
    sys.exit(main())
