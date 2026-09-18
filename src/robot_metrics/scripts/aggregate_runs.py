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
    rpe_vs_agitation.eps       windowed rotational and translational RPE
                               against vibration and attitude agitation
    rpe_corr_vs_window.eps     how those correlations move with the window
                               length, raw and partialled on velocity

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

from metrics_io import (load_metrics, load_gt_highrate,  # noqa: E402
                        vibration_rms)
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
    # The rows above are the ONLINE estimate: the pose the SLAM published at
    # each instant, which is what a robot navigating on it would have had.
    # The rows below are the OPTIMISED GRAPH, after the loop closures have
    # been applied backwards over the trajectory - which is what "ATE" means
    # in the SLAM literature, and the only place closing the loop shows up.
    # Measured 2026-08-23, rocker_bogie, one two-lap run: 0.998 m online
    # against 0.177 m optimised, on the same run.  Report both, and say which
    # is which.
    ('ate_opt_trans_rmse', 'ATE optimised graph RMSE', 'm', '.4f'),
    ('ate_opt_trans_mean', 'ATE optimised graph mean', 'm', '.4f'),
    ('ate_opt_trans_max', 'ATE optimised graph max', 'm', '.4f'),
    ('ate_opt_n_poses', 'Optimised graph poses', 'count', '.0f'),
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
    # Proximity detections are the LiDAR closure mechanism, and on this sensor
    # suite they are the ONLY one that fires - the visual rows above read 0 by
    # construction, because the camera publishes no depth.  Reporting the
    # visual rows alone made a working campaign look like a broken one.
    ('proximity_total', 'Proximity detections', 'count', '.0f'),
    ('proximity_true', 'Proximity true pos.', 'count', '.0f'),
    ('proximity_false', 'Proximity false pos.', 'count', '.0f'),
    ('pitch_std_deg', 'Pitch std', 'deg', '.3f'),
    ('roll_std_deg', 'Roll std', 'deg', '.3f'),
    ('pitch_max_deg', 'Pitch max abs', 'deg', '.3f'),
    ('vibration_rms', 'Vibration RMS', 'm/s^2', '.4f'),
    # Body-frame correction slam_metrics removed before measuring anything.
    # Near 0 for a platform whose base_link faces the way it drives, near
    # 180 for the Rocker-Bogie, whose URDF does not.  Printed so the
    # correction is visible rather than silent: a value that is neither, or
    # one that moves between runs of the same platform, means there is no
    # fixed convention to remove and that run's ATE, RPE and drift are not
    # trustworthy.
    ('gt_yaw_offset_deg', 'GT yaw offset removed', 'deg', '.1f'),
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
           'proximity_total': 0, 'proximity_true': 0,
           'proximity_false': 0, 'proximity_unknown': 0,
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
            elif kind == 'proximity_detection':
                out['proximity_total'] += 1
                if 'verdict=true' in detail:
                    out['proximity_true'] += 1
                elif 'verdict=false' in detail:
                    out['proximity_false'] += 1
                else:
                    out['proximity_unknown'] += 1
    return out


def locomotion_metrics(df):
    # Actitud de la verdad-terreno cuando el logger la escribio; el IMU queda
    # de reserva solo para la ACTITUD, que no depende de donde este montado.
    #
    # La vibracion NO tiene reserva: sale de metrics_io.vibration_rms, unica
    # definicion del repositorio, y vale n/a en las corridas sin gt_a*.  El
    # IMU no sirve porque va en un punto distinto de cada plataforma y porque
    # restar 9.81 de z deja dentro la proyeccion de g del cabeceo.
    pitch = (df['gt_pitch'] if 'gt_pitch' in df.columns else df['pitch']).values
    roll = (df['gt_roll'] if 'gt_roll' in df.columns else df['roll']).values
    # accel_z_max y accel_x_rms se quedan COMO ESTABAN -verdad-terreno cuando
    # la hay- para no mover cifras que ya estan en las tablas.  Ojo: en
    # analyze_metrics las claves con el mismo nombre son del IMU; los dos
    # scripts no informan lo mismo bajo "accel_*", y eso sigue sin unificar.
    if 'gt_ax' in df.columns:
        ax, az = df['gt_ax'].values, df['gt_az'].values
    else:
        ax = df['accel_x'].values
        az = df['accel_z'].values - GRAVITY
    return {
        'pitch_std_deg': float(np.degrees(np.std(pitch))),
        'roll_std_deg': float(np.degrees(np.std(roll))),
        'pitch_max_deg': float(np.degrees(np.max(np.abs(pitch)))),
        'roll_max_deg': float(np.degrees(np.max(np.abs(roll)))),
        'vibration_rms': vibration_rms(df),
        'accel_z_max': float(np.max(np.abs(az))),
        'accel_x_rms': float(np.sqrt(np.mean(ax ** 2))),
    }


def _load_tum(path):
    """TUM: timestamp tx ty tz qx qy qz qw -> stamps, 4x4 transforms."""
    stamps, mats = [], []
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
            mats.append(M)
    return np.array(stamps), np.array(mats)


def optimised_ate(run_dir, match_tol_s=0.05):
    """ATE of the OPTIMISED graph, which is where loop closure shows up.

    est_traj.tum holds what the SLAM published live, and a loop closure
    rewrites the graph behind it, so that file never sees the correction.
    run_campaign.sh exports opt_traj.tum from the preserved database for
    exactly this comparison; a run collected before that existed simply has
    no opt_traj.tum and these fields come back as NaN.
    """
    opt = os.path.join(run_dir, 'opt_traj.tum')
    gt = os.path.join(run_dir, 'gt_traj.tum')
    blank = {'ate_opt_trans_rmse': float('nan'),
             'ate_opt_trans_mean': float('nan'),
             'ate_opt_trans_max': float('nan'),
             'ate_opt_n_poses': 0}
    if not (os.path.isfile(opt) and os.path.isfile(gt)):
        return blank
    try:
        es, eT = _load_tum(opt)
        gs, gT = _load_tum(gt)
    except (OSError, ValueError):
        return blank
    if len(es) < 5 or len(gs) < 5:
        return blank

    # nearest ground-truth sample in time for each optimised pose
    idx = np.clip(np.searchsorted(gs, es), 1, len(gs) - 1)
    idx = np.where(np.abs(es - gs[idx - 1]) < np.abs(es - gs[idx]),
                   idx - 1, idx)
    keep = np.abs(es - gs[idx]) < match_tol_s
    if keep.sum() < 5:
        return blank

    trans, _ = sm.ate(sm.align_umeyama(eT[keep], gT[idx[keep]]), gT[idx[keep]])
    return {'ate_opt_trans_rmse': float(np.sqrt(np.mean(trans ** 2))),
            'ate_opt_trans_mean': float(np.mean(trans)),
            'ate_opt_trans_max': float(np.max(trans)),
            'ate_opt_n_poses': int(keep.sum())}


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
    res.update(optimised_ate(run_dir))
    res['duration_s'] = float(df['timestamp'].values[-1] -
                              df['timestamp'].values[0])
    res['run_dir'] = run_dir
    res['run_label'] = os.path.basename(run_dir)
    # La agitacion y la vibracion salen de gt_highrate.csv -verdad-terreno a
    # 1000 Hz- con ventana de 100 ms, no de metrics.csv a 50 Hz con ventana de
    # 1 s.  load_gt_highrate devuelve None en corridas viejas y entonces
    # stability_error_correlation cae al camino anterior.
    hr = load_gt_highrate(os.path.join(run_dir, 'metrics.csv'))
    res['hr'] = hr
    res['corr'] = sm.stability_error_correlation(
        df, res['ate_trans_series'], hr=hr,
        ate_rot=res.get('ate_rot_series'),
        rpe_rot=res.get('rpe_rot_series'),
        rpe_trans=res.get('rpe_trans_series'))
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


def _etiqueta_ventana(results):
    """'100 ms' o '1 s', leido de las propias corridas.

    El rotulo tiene que salir del dato: el 2026-09-01 la ventana bajo a
    100 ms y las figuras siguieron diciendo "1 s" mientras dibujaban datos
    de 100 ms.  Una corrida vieja que caiga a la ventana de 1 s lo dira
    correctamente sin que nadie tenga que acordarse.
    """
    for corridas in (results or {}).values():
        for c in corridas:
            if c.get('corr'):
                w = c['corr']['window_s']
                return ('%.0f ms' % (w * 1000.0)) if w < 1.0 else ('%g s' % w)
    return '1 s'


def correlation_table(robots, results, out_path):
    """The stability -> SLAM performance evidence."""
    lines = []
    lines.append('=' * 96)
    lines.append('CHASSIS STABILITY vs INSTANTANEOUS SLAM ERROR')
    lines.append('=' * 96)
    lines.append('Pearson r between windowed chassis agitation and windowed')
    lines.append('SLAM error, with a 95% percentile-bootstrap CI (2000 resamples).')
    lines.append('')
    lines.append('Predictors, inside a ' + _etiqueta_ventana(results)
                 + ' window:')
    lines.append('  pitch_std / roll_std / agitation  std of attitude [rad]')
    lines.append('  vibration_rms                     RMS |gt_a| [m/s^2], la '
                 'aceleracion propia')
    lines.append('                                    de verdad-terreno, NO el '
                 'IMU: ver metrics_io')
    lines.append('')
    lines.append('Responses, over the same window:')
    lines.append('  ate_mean    mean instantaneous ATE [m].  ACCUMULATED error, so')
    lines.append('              it carries the whole run up to that window; it')
    lines.append('              correlates with WHERE on the route the window is')
    lines.append('              as much as with what happened during it.')
    lines.append('  ate_growth  ATE gained across the window [m].  This is the one')
    lines.append('              that asks whether agitation DURING this window cost')
    lines.append('              accuracy, and the one to quote.')
    lines.append('')
    lines.append('A CI that spans 0 means this run cannot distinguish the effect')
    lines.append('from none.  Read the sign only when it does not.')
    lines.append('')
    lines.append('r|vel is the PARTIAL correlation with the windowed speed')
    lines.append('discounted from both sides, r(predictor, response | speed).')
    lines.append('It exists because agitation and per-metre error share an')
    lines.append('obvious common cause: driving faster shakes the chassis AND')
    lines.append('leaves fewer sweeps per metre, so a raw r can be the speed')
    lines.append('looking at itself.  r|vel close to r means speed was not the')
    lines.append('explanation; r|vel collapsing towards 0 means it was.')
    lines.append('')
    header = ('{:<22s}{:<7s}{:<14s}{:<12s}{:>8s}{:>20s}{:>9s}{:>8s}'
              .format('Robot', 'run', 'predictor', 'response', 'r',
                      '95% CI', 'r|vel', 'wins'))
    lines.append(header)
    lines.append('-' * len(header))

    PREDICTORS = (('pitch_std_rad', 'pitch_std'),
                  ('roll_std_rad', 'roll_std'),
                  ('attitude_agitation_rad', 'agitation'),
                  ('vibration_rms_m_s2', 'vibration_rms'))
    # rpe_rot va el ultimo pero es el que responde a la pregunta local:
    # el ATE es acumulado, asi que su correlacion con lo que agita el
    # chasis en ESTA ventana esta diluida por toda la historia previa.
    RESPONSES = (('ate_mean_m', 'ate_mean'),
                 ('ate_growth_m', 'ate_growth'),
                 ('rpe_rot_mean_deg_m', 'rpe_rot'),
                 ('rpe_trans_mean_m_m', 'rpe_trans'))

    pooled = {}
    for robot in robots:
        # pooled[] keeps the headline pairing: agitation against the growth of
        # the error, which is the one the paper should quote.
        rs = []
        first = True
        for r in results[robot]:
            c = r.get('corr')
            if not c:
                continue
            for pkey, plabel in PREDICTORS:
                for rkey, rlabel in RESPONSES:
                    rval = c.get('r_{}__{}'.format(pkey, rkey), float('nan'))
                    ci = c.get('ci_{}__{}'.format(pkey, rkey),
                               (float('nan'), float('nan')))
                    if pkey == 'attitude_agitation_rad' and rkey == 'ate_growth_m':
                        rs.append(rval)
                    prval = c.get('pr_{}__{}'.format(pkey, rkey),
                                  float('nan'))
                    lines.append(
                        '{:<22s}{:<7s}{:<14s}{:<12s}{:>8s}{:>20s}{:>9s}{:>8s}'
                        .format(
                            DISPLAY_NAME.get(robot, robot) if first else '',
                            r['run_label'] if first else '',
                            plabel, rlabel, fmt(rval, '.3f'),
                            '[{}, {}]'.format(fmt(ci[0], '.3f'),
                                              fmt(ci[1], '.3f')),
                            fmt(prval, '.3f'),
                            str(c.get('n_windows', 0)) if first else ''))
                    first = False
            lines.append('')
        if rs:
            m, s, n = mean_std(rs)
            pooled[robot] = (m, s, n)
            lines.append('{:<22s}{:<7s}{:<14s}{:<12s}{:>8s}{:>20s}'.format(
                DISPLAY_NAME.get(robot, robot), 'mean', 'agitation',
                'ate_growth', fmt(m, '.3f'),
                '+/- {} (n={})'.format(fmt(s, '.3f'), n)))
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
    # Ver la nota en plot_agitation_heading._etiqueta_ventana(): el
    # rotulo tiene que salir del dato, no de una constante en el texto.
    ax.set_xlabel('Attitude agitation per ' + _etiqueta_ventana(results)
                  + ' window, '
                  r'$\sqrt{\sigma_\theta^2+\sigma_\phi^2}$ (deg)')
    ax.set_ylabel('Mean instantaneous ATE (m)')
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir, 'stability_vs_error')


def _nubes_rpe(results, robot, clave_x, clave_y, a_grados):
    """(x, y) de todas las ventanas de un robot para un par predictor/respuesta."""
    xs, ys = [], []
    for r in results[robot]:
        c = r.get('corr')
        if not c:
            continue
        for w in c['windows']:
            x, y = w.get(clave_x), w.get(clave_y)
            if x is None or y is None:
                continue
            if np.isfinite(x) and np.isfinite(y):
                xs.append(np.degrees(x) if a_grados else x)
                ys.append(y)
    return np.array(xs), np.array(ys)


def plot_rpe_vs_agitation(robots, results, out_dir):
    """RPE por ventana contra lo que zarandea al chasis.  Rejilla 2x2.

    Filas: la componente del RPE -rotacional arriba, traslacional abajo-.
    Columnas: el predictor -vibracion a la izquierda, agitacion de actitud a
    la derecha-.  Cada fila comparte el eje Y y cada columna el eje X, que es
    lo que permite comparar de un vistazo sin releer los rotulos.

    UNA REJILLA, NO DOS EJES X SUPERPUESTOS.  Vibracion (m/s^2) y agitacion
    (deg) no comparten escala ni unidad; ponerlas sobre un mismo eje obligaria
    a leer cada nube contra un eje distinto.

    POR QUE EL RPE Y NO EL ATE.  stability_vs_error ya enfrenta la agitacion
    contra el ATE, que es un error ACUMULADO: cuando el rumbo ya se ha ido
    sigue siendo grande aunque el metro actual se estime perfecto, asi que esa
    nube arrastra toda la historia de la corrida y su r sale indistinguible de
    cero.  El RPE mide el movimiento RELATIVO sobre un metro, asi que responde
    a la pregunta local: esta sacudida, aqui, cuanto costo.

    La identidad de cada plataforma va en color Y en marcador: la paleta del
    paper no separa el rojo del verde en deuteranopia (Delta E 3.9, medido),
    asi que el color por si solo no distingue dos de las tres series.
    """
    filas = [
        ('rpe_rot_mean_deg_m', 'Rotational RPE per %.0f m (deg)'),
        ('rpe_trans_mean_m_m', 'Translational RPE per %.0f m (m)'),
    ]
    cols = [
        ('vibration_rms_m_s2', False,
         'Vibration RMS per %s window,\n' r'$|a_{\rm gt}|$ (m/s$^2$)'),
        ('attitude_agitation_rad', True,
         'Attitude agitation per %s window,\n'
         r'$\sqrt{\sigma_\theta^2+\sigma_\phi^2}$ (deg)'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH_IN, 5.6),
                             sharex='col', sharey='row')
    hay = False
    for fi, (clave_y, rot_y) in enumerate(filas):
        for ci, (clave_x, a_grados, rot_x) in enumerate(cols):
            ax = axes[fi][ci]
            for i, robot in enumerate(robots):
                xs, ys = _nubes_rpe(results, robot, clave_x, clave_y, a_grados)
                if len(xs) < 3:
                    continue
                hay = True
                rho = float(np.corrcoef(xs, ys)[0, 1])
                # rasterized: el backend PostScript descarta el alpha y las
                # nubes salian opacas en el .eps, tapandose entre plataformas.
                # Rasterizadas, Agg mezcla la transparencia y el resto del
                # panel sigue siendo vectorial.
                ax.scatter(xs, ys, s=7, alpha=0.30, color=COLORS[i % 3],
                           marker=MARKERS[i % 3], linewidths=0, rasterized=True,
                           label='%s  (r = %+.2f)'
                                 % (DISPLAY_NAME.get(robot, robot), rho))
                k, b = np.polyfit(xs, ys, 1)
                xf = np.linspace(xs.min(), xs.max(), 50)
                ax.plot(xf, k * xf + b, color=COLORS[i % 3], linewidth=1.2)
            # Una leyenda por panel: la r de la vibracion y la de la agitacion
            # son distintas, y una leyenda compartida las daria por iguales.
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=6, loc='best')
            if fi == len(filas) - 1:
                ax.set_xlabel(rot_x % _etiqueta_ventana(results))
        axes[fi][0].set_ylabel(rot_y % _delta_rpe(results))
    if not hay:
        plt.close(fig)
        return
    fig.tight_layout()
    save(fig, out_dir, 'rpe_vs_agitation')


VENTANAS_BARRIDO = (0.05, 0.1, 0.2, 0.5, 1.0)


def _barrido_ventanas(results, robot, ventanas=VENTANAS_BARRIDO):
    """Recalcula las correlaciones a varios tamanos de ventana.

    Devuelve {window_s: corr}.  El tamano de ventana no es un detalle de
    implementacion: fija QUE se esta preguntando.  Una ventana corta mide si
    la sacudida de este instante costo error en este instante; una larga mide
    si el tramo agitado costo error en el tramo.  Que r crezca con la ventana
    dice que el efecto es acumulativo y que a 50 ms el ruido de medida se
    come la senal, no que la relacion sea mas fuerte.

    El bootstrap va apagado aqui (con_ci=False): son cinco ventanas por robot
    y por corrida, y el intervalo ya se reporta en la tabla a la ventana
    nominal.
    """
    salida = {}
    for r in results[robot]:
        df = r.get('df')
        if df is None:
            continue
        for w in ventanas:
            c = sm.stability_error_correlation(
                df, r['ate_trans_series'], window_s=w, hr=r.get('hr'),
                ate_rot=r.get('ate_rot_series'),
                rpe_rot=r.get('rpe_rot_series'),
                rpe_trans=r.get('rpe_trans_series'),
                con_ci=False)
            if c:
                salida.setdefault(w, []).append(c)
    return salida


def plot_rpe_corr_vs_window(robots, results, out_dir):
    """Como cambia la correlacion con el tamano de ventana, cruda y parcial.

    LA LINEA DISCONTINUA ES LA QUE IMPORTA.  La agitacion del chasis y el
    error por metro comparten una causa: la velocidad.  Ir mas rapido agita
    mas Y deja menos barridos por metro, asi que parte de la r cruda puede ser
    la velocidad mirandose al espejo.  La discontinua es
    r(predictor, RPE | velocidad): lo que queda a igualdad de velocidad.  Si
    las dos lineas van juntas, la velocidad no era la explicacion; si la
    discontinua se hunde, si lo era.
    """
    filas = [
        ('vibration_rms_m_s2', 'Vibration'),
        ('attitude_agitation_rad', 'Attitude agitation'),
    ]
    cols = [
        ('rpe_rot_mean_deg_m', 'Rotational RPE'),
        ('rpe_trans_mean_m_m', 'Translational RPE'),
    ]
    barridos = {}
    for robot in robots:
        try:
            barridos[robot] = _barrido_ventanas(results, robot)
        except Exception:
            barridos[robot] = {}
    if not any(barridos.values()):
        return
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH_IN, 5.2),
                             sharex=True, sharey=True)
    ventanas = sorted(VENTANAS_BARRIDO)
    for fi, (pred, rot_p) in enumerate(filas):
        for ci, (resp, rot_r) in enumerate(cols):
            ax = axes[fi][ci]
            for i, robot in enumerate(robots):
                for clave, estilo, sufijo in (('r_', '-', ''),
                                              ('pr_', '--', ' | velocity')):
                    ys = []
                    for w in ventanas:
                        cs = barridos.get(robot, {}).get(w, [])
                        vals = [c.get('%s%s__%s' % (clave, pred, resp))
                                for c in cs]
                        vals = [v for v in vals
                                if v is not None and np.isfinite(v)]
                        ys.append(float(np.mean(vals)) if vals else np.nan)
                    if not np.any(np.isfinite(ys)):
                        continue
                    ax.plot(ventanas, ys, estilo, color=COLORS[i % 3],
                            marker=MARKERS[i % 3], markersize=4, linewidth=1.2,
                            label='%s%s' % (DISPLAY_NAME.get(robot, robot),
                                            sufijo))
            ax.set_xscale('log')
            ax.set_xticks(ventanas)
            ax.set_xticklabels(['%g ms' % (w * 1000) if w < 1 else '1 s'
                                for w in ventanas], fontsize=7)
            ax.axhline(0.0, color='0.6', linewidth=0.6, zorder=0)
            ax.set_title('%s %s %s' % (rot_p, r'$\rightarrow$', rot_r),
                         fontsize=8)
            if fi == len(filas) - 1:
                ax.set_xlabel('Window length')
            if ci == 0:
                ax.set_ylabel('Pearson r')
    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        axes[0][0].legend(h, l, fontsize=6, loc='best')
    fig.tight_layout()
    save(fig, out_dir, 'rpe_corr_vs_window')


def _delta_rpe(results):
    """El delta_m con el que se calculo el RPE: el rotulo sale del dato."""
    for runs in results.values():
        for r in runs:
            if r.get('rpe_delta_m'):
                return float(r['rpe_delta_m'])
    return 1.0


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
    plot_rpe_vs_agitation(robots, results, args.output_dir)
    plot_rpe_corr_vs_window(robots, results, args.output_dir)
    plot_run_spread(robots, results, args.output_dir)

    print('\nWritten to {}:'.format(os.path.abspath(args.output_dir)))
    for f in sorted(os.listdir(args.output_dir)):
        print('  - {}'.format(f))
    return 0


if __name__ == '__main__':
    sys.exit(main())
