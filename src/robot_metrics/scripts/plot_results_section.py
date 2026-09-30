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
import re
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


# Ancho de banda de la aceleracion.  La de las tablas deriva la velocidad a
# 1 kHz, y un impulso de contacto da un pico ~1/dt: el RMS y sobre todo los
# picos dependen del ancho de banda (critica de revision).  Como sensibilidad
# se da tambien con una media movil de LP_WINDOW_MS (-3 dB a ~44 Hz, primer
# cero a 100 Hz para 10 ms).
LP_WINDOW_MS = 10


def boxcar(x, k):
    """Media movil centrada de k muestras que ignora los NaN."""
    ok = np.isfinite(x)
    w = np.ones(k)
    num = np.convolve(np.where(ok, x, 0.0), w, 'same')
    den = np.convolve(ok * 1.0, w, 'same')
    return np.where(den >= k / 2.0, num / np.maximum(den, 1.0), np.nan)


def accel_lowpass(hr):
    """gt_ax/ay/az de gt_highrate pasados por boxcar(LP_WINDOW_MS)."""
    k = int(round(LP_WINDOW_MS / (1e3 * np.median(np.diff(hr['timestamp'].values)))))
    return {c: boxcar(hr[c].values, max(k, 1)) for c in ('gt_ax', 'gt_ay', 'gt_az')}


# Vuelta: primera vez que la verdad-terreno vuelve a menos de LAP_RADIUS_M del
# inicio despues de recorrer LAP_MIN_TRAVEL_M.
LAP_RADIUS_M = 3.0
LAP_MIN_TRAVEL_M = 50.0


def review_checks(res):
    """Cifras que el texto cita en respuesta a la revision, por corrida.

    pairing   RPE/ATE emparejados fila a fila (lo que habia) frente a por
              estimacion (las tablas): el sesgo de retencion.
    latency   RPE con la verdad-terreno retrasada tau (sensibilidad).
    offset    delta de la estimacion y el ATE segun como se corrija.
    prox      detecciones de proximidad: cuantas en la primera vuelta y a que
              distancia real estan los nodos que enlazan (gt_gap del logger;
              las de gt_gap=nan no tienen nodo conocido y no cuentan).
    autocorr  autocorrelacion de las series por barrido al lag del bloque del
              bootstrap: justifica su longitud.
    """
    df, hr = res['df'], res['hr']
    old = sm.evaluate(df)
    tr_tau, ro_tau = sm.rpe_at_delay(df, hr, res['latency_s'])

    xy = df[['gt_x', 'gt_y']].values
    s = res['distance_series']
    back = np.flatnonzero((np.linalg.norm(xy - xy[0], axis=1) < LAP_RADIUS_M)
                          & (s > LAP_MIN_TRAVEL_M))
    t_lap = df['timestamp'].values[back[0]] if back.size else np.inf
    gaps, lap1 = [], []
    for row in res.get('events', []):
        if row.get('event[-]') != 'proximity_detection':
            continue
        m = re.search(r'gt_gap=([\d.]+)m', row.get('detail[-]', ''))
        if m:
            gaps.append(float(m.group(1)))
            lap1.append(float(row['timestamp[s]']) < t_lap)
    gaps, lap1 = np.array(gaps), np.array(lap1, dtype=bool)

    W = res['corr']['windows']
    L = res['corr']['block']
    ac = {}
    for k in ('attitude_agitation_rad', 'sweep_rot_deg', 'sweep_trans_m'):
        x = np.array([w[k] for w in W], dtype=float)
        x = x - np.nanmean(x)
        ac[k] = float(np.nansum(x[:-L] * x[L:]) / np.nansum(x * x))

    # vibracion de toda la corrida, mismo tramo que la tabla, con y sin filtro
    t = df['timestamp'].values
    tt = hr['timestamp'].values
    tramo = (tt >= t[0]) & (tt <= t[-1])
    lp = accel_lowpass(hr)
    a_lp = np.sqrt(sum(lp[c][tramo] ** 2 for c in lp))
    vib_lp = float(np.sqrt(np.nanmean(a_lp ** 2)))

    # contacto de ruedas (solo el rocker los lleva), filas de metrics.csv
    cc = [c for c in df.columns if c.startswith('contact_')]
    if cc:
        C = df[cc].values > 0.5
        contact = {'all': float(C.all(axis=1).mean()),
                   'min_wheel': float(C.mean(axis=0).min()),
                   'n_wheels': len(cc)}
    else:
        contact = None

    return {
        'vib': res.get('vibration_rms', float('nan')), 'vib_lp': vib_lp,
        'contact': contact, 'corr': res['corr'],
        'rpe_t_row': old['rpe_trans_rmse'], 'rpe_r_row': old['rpe_rot_rmse'],
        'ate_row': old['ate_origin_trans_rmse'],
        'rpe_t': res['rpe_trans_rmse'], 'rpe_r': res['rpe_rot_rmse'],
        'ate': res['ate_origin_trans_rmse'],
        'rpe_t_tau': tr_tau, 'rpe_r_tau': ro_tau,
        'offset': sm.ate_offset_sensitivity(df, hr),
        'lap_m': float(s[back[0]]) if back.size else float('nan'),
        'gaps': gaps, 'lap1': lap1, 'block': L, 'autocorr': ac,
    }


def print_review(robots, chk):
    """Imprime, por plataforma, las cifras de review_checks tal y como las
    cita el texto: medias sobre corridas o rangos entre corridas."""
    def mean(r, k):
        return np.mean([c[k] for c in chk[r]])

    def pct(a, b):
        return 100.0 * (b - a) / a

    print('\n=== RPE PAIRING: row-by-row (held odometry) vs per estimate '
          '(tables); means over runs ===')
    for r in robots:
        print('%-14s RPE trans %.4f -> %.4f m/m (%+.1f%%)   rot %.3f -> %.3f '
              'deg/m (%+.1f%%)   ATE max |change| over runs %.2f%%'
              % (short(r), mean(r, 'rpe_t_row'), mean(r, 'rpe_t'),
                 pct(mean(r, 'rpe_t_row'), mean(r, 'rpe_t')),
                 np.degrees(mean(r, 'rpe_r_row')), np.degrees(mean(r, 'rpe_r')),
                 pct(mean(r, 'rpe_r_row'), mean(r, 'rpe_r')),
                 max(abs(pct(c['ate_row'], c['ate'])) for c in chk[r])))

    print('\n=== LATENCY-CORRECTED RPE (GT delayed by tau per run; '
          'sensitivity only) ===')
    for r in robots:
        print('%-14s RPE trans %.4f m/m (%+.1f%% vs table)   rot %.3f deg/m '
              '(%+.1f%%)' % (short(r), mean(r, 'rpe_t_tau'),
                             pct(mean(r, 'rpe_t'), mean(r, 'rpe_t_tau')),
                             np.degrees(mean(r, 'rpe_r_tau')),
                             pct(mean(r, 'rpe_r'), mean(r, 'rpe_r_tau'))))
    for k, name in (('rpe_t', 'trans'), ('rpe_r', 'rot')):
        for suf, cual in (('', 'table'), ('_tau', 'tau')):
            orden = sorted(robots, key=lambda r: mean(r, k + suf))
            print('  ranking %-5s %-5s : %s' % (name, cual,
                                                ' < '.join(short(r) for r in orden)))

    print('\n=== ESTIMATE HEADING OFFSET and start-aligned ATE by correction ===')
    for r in robots:
        o = [c['offset'] for c in chk[r]]
        d = [x['est_offset_deg'] for x in o]
        dg = [abs(x['ate_gt_offset'] - x['ate_own']) for x in o]
        dn = [abs(x['ate_uncorrected'] - x['ate_own']) for x in o]
        dnp = [100 * abs(x['ate_uncorrected'] - x['ate_own']) / x['ate_own'] for x in o]
        print('%-14s est offset %.2f..%.2f deg   |ATE gt-offset - own| max %.3f m   '
              '|ATE uncorrected - own| max %.3f m (%.1f%%)'
              % (short(r), min(d), max(d), max(dg), max(dn), max(dnp)))
    for k in ('ate_own', 'ate_gt_offset', 'ate_uncorrected'):
        orden = sorted(robots, key=lambda r: np.mean([c['offset'][k] for c in chk[r]]))
        print('  ranking %-16s: %s' % (k, ' < '.join(short(r) for r in orden)))

    print('\n=== PROXIMITY DETECTIONS with known node gap: lap split at first '
          'return within %.0f m of start after %.0f m ===' % (LAP_RADIUS_M,
                                                              LAP_MIN_TRAVEL_M))
    for r in robots:
        cs = chk[r]
        rng = lambda f: '%.1f..%.1f' % (min(f(c) for c in cs), max(f(c) for c in cs))
        print('%-14s lap at %s m   lap1 share %s%%   lap1 gap median %s m   '
              'gap median %s m   <=1 m %s%%   in (1,5] m %d'
              % (short(r), rng(lambda c: c['lap_m']),
                 rng(lambda c: 100 * c['lap1'].mean()),
                 rng(lambda c: np.median(c['gaps'][c['lap1']])),
                 rng(lambda c: np.median(c['gaps'])),
                 rng(lambda c: 100 * np.mean(c['gaps'] <= 1.0)),
                 sum(int(np.sum((c['gaps'] > 1) & (c['gaps'] <= 5))) for c in cs)))

    print('\n=== YAW AGITATION without the commanded turn, sigma(psi - int '
          'omega_cmd), one sweep; means over runs ===')
    for r in robots:
        cs = [c['corr'] for c in chk[r]]
        g = lambda k: np.nanmean([c.get(k, np.nan) for c in cs])
        print('%-14s rot: r %.3f (raw sigma_psi %.3f)  r|attitude %.3f (raw %.3f)   '
              'trans: r %.3f (raw %.3f)'
              % (short(r),
                 g('r_yaw_resid_agitation_rad__sweep_rot_deg'),
                 g('r_yaw_agitation_rad__sweep_rot_deg'),
                 g('pa_yaw_resid_agitation_rad__sweep_rot_deg'),
                 g('pa_yaw_agitation_rad__sweep_rot_deg'),
                 g('r_yaw_resid_agitation_rad__sweep_trans_m'),
                 g('r_yaw_agitation_rad__sweep_trans_m')))

    print('\n=== VIBRATION RMS: 1 kHz derivative (tables) vs %d ms moving '
          'average; means over runs ===' % LP_WINDOW_MS)
    for r in robots:
        print('%-14s %.3f -> %.3f m/s^2' % (short(r), mean(r, 'vib'), mean(r, 'vib_lp')))

    print('\n=== WHEEL CONTACT (contact sensors, 50 Hz rows, after warm-up) ===')
    for r in robots:
        cs = [c['contact'] for c in chk[r] if c['contact']]
        if cs:
            print('%-14s all %d wheels in contact %.2f..%.2f%% of the time   '
                  'least-contact wheel %.2f..%.2f%%'
                  % (short(r), cs[0]['n_wheels'], 100 * min(c['all'] for c in cs),
                     100 * max(c['all'] for c in cs),
                     100 * min(c['min_wheel'] for c in cs),
                     100 * max(c['min_wheel'] for c in cs)))

    print('\n=== AUTOCORRELATION of per-sweep series at the bootstrap block lag '
          '(max over runs) ===')
    for r in robots:
        cs = chk[r]
        print('%-14s block %d sweeps   %s' % (
            short(r), cs[0]['block'],
            '  '.join('%s %.3f' % (k, max(c['autocorr'][k] for c in cs))
                      for k in cs[0]['autocorr'])))


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
    lp = accel_lowpass(hr)
    return {
        'x': d - d0, 'slab_len': d1 - d0,
        'az': hr['gt_az'].values[near], 'ax': hr['gt_ax'].values[near],
        'az_lp': lp['gt_az'][near], 'ax_lp': lp['gt_ax'][near],
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
    axes[-1].set_xlabel('Distance from the slab footprint edge along the path (m); '
                        'shaded: base over the footprint')
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


def print_numbers(robots, prof, stats, sweeps, latency, noshift, rms_rot):
    print('\n=== STEP CROSSING (run01, first pass) ===')
    for r in robots:
        p = prof.get(r)
        if p is None:
            print('%-14s no crossing found' % short(r))
            continue
        print('%-14s |a_z|max %6.2f  |a_x|max %6.2f  pitch [%6.2f, %6.2f]  '
              'roll [%6.2f, %6.2f]  |th_dot|max %7.1f  |ph_dot|max %7.1f  '
              'slab %.2f m  window %.1f s'
              % (short(r), np.nanmax(np.abs(p['az'])), np.nanmax(np.abs(p['ax'])),
                 p['pitch'].min(), p['pitch'].max(), p['roll'].min(),
                 p['roll'].max(), np.max(np.abs(p['pitch_rate'])),
                 np.max(np.abs(p['roll_rate'])), p['slab_len'], p['duration_s']))
    print('low-pass (%d ms moving average):' % LP_WINDOW_MS)
    for r in robots:
        p = prof.get(r)
        if p is not None:
            print('%-14s |a_z|max %6.2f  |a_x|max %6.2f'
                  % (short(r), np.nanmax(np.abs(p['az_lp'])),
                     np.nanmax(np.abs(p['ax_lp']))))

    print('\n=== ATTITUDE DISTRIBUTION (all runs, gt_highrate) ===')
    print('%-14s %-10s %8s %8s %8s %8s %8s %8s'
          % ('robot', 'signal', 'median', 'p25', 'p75', 'p1', 'p99', 'std'))
    for r in robots:
        for k in STATS:
            v = stats[r][k]
            q = np.percentile(v, [50, 25, 75, 1, 99])
            print('%-14s %-10s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f'
                  % ((short(r), k) + tuple(q) + (np.std(v),)))

    print('\n=== WINDOW SWEEP (sweeps per window, matched response): '
          'mean over runs of r and r|velocity ===')
    pares = [('vibration_rms_m_s2', 'sweep_rot_deg', 'vib->rot'),
             ('vibration_rms_m_s2', 'sweep_trans_m', 'vib->trans'),
             ('attitude_agitation_rad', 'sweep_rot_deg', 'agit->rot'),
             ('attitude_agitation_rad', 'sweep_trans_m', 'agit->trans'),
             ('yaw_agitation_rad', 'sweep_rot_deg', 'yaw->rot'),
             ('yaw_agitation_rad', 'sweep_trans_m', 'yaw->trans')]
    for r in robots:
        for w in sorted(sweeps[r]):
            cs = sweeps[r][w]
            cells = []
            for pred, resp, nombre in pares:
                rr = np.nanmean([c.get('r_%s__%s' % (pred, resp)) for c in cs])
                pr = np.nanmean([c.get('pr_%s__%s' % (pred, resp)) for c in cs])
                cells.append('%s %.3f|%.3f' % (nombre, rr, pr))
            print('%-14s %2d sw  %s' % (short(r), w, '  '.join(cells)))

    print('\n=== LATENCY and SENSITIVITY (one sweep per window) ===')
    print('tau: GT delay minimising the 1 m translational RPE, per run.  '
          'noshift: same correlation with tau = 0.')
    for r in robots:
        taus = [1e3 * x for x in latency[r]]
        ns = noshift[r]
        cell = lambda k: np.nanmean([c.get(k) for c in ns])
        print('%-14s tau %s ms (mean %.0f)   noshift agit->rot %.3f  '
              'agit->trans %.3f   per-sweep rot RMS %.3f -> %.3f deg'
              % (short(r), ' '.join('%.0f' % x for x in taus), np.mean(taus),
                 cell('r_attitude_agitation_rad__sweep_rot_deg'),
                 cell('r_attitude_agitation_rad__sweep_trans_m'),
                 np.mean([x[0] for x in rms_rot[r]]),
                 np.mean([x[1] for x in rms_rot[r]])))


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
    latency, noshift, rms_rot, chk = {}, {}, {}, {}
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
            latency.setdefault(robot, []).append(res['latency_s'])
            noshift.setdefault(robot, []).append(sm.sweep_error_correlation(
                res['sweeps_noshift'], hr, con_ci=False) or {})
            chk.setdefault(robot, []).append(review_checks(res))
            rms_rot.setdefault(robot, []).append(tuple(
                float(np.sqrt(np.nanmean(res[k]['rot'] ** 2)))
                for k in ('sweeps_noshift', 'sweeps')))
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
    print_numbers(robots, prof, stats, sweeps, latency, noshift, rms_rot)
    print_review(robots, chk)
    print('\nwritten to %s' % out_dir)


if __name__ == '__main__':
    main()
