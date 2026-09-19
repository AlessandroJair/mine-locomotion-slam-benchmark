#!/usr/bin/env python3
"""Trajectory-error metrics for the three-platform comparison.

Everything here is numpy-only on purpose: the workstation running the
simulations has no scipy, and a dependency that has to be installed before the
paper's numbers can be reproduced is a dependency that will not be installed.

WHAT IS COMPUTED, AND WHY IT IS SPLIT
=====================================
A single scalar "ATE in metres" hides the effect the paper is actually arguing
for.  A platform that pitches and rolls its sensor mast degrades the ROTATIONAL
part of the estimate first; the translational error is a downstream consequence
that accumulates over distance.  So every metric is reported as a translational
and a rotational component:

  ATE  absolute trajectory error, after aligning the estimate to ground truth
       - ate_trans  [m]    ||t_gt,i - t_est,i||
       - ate_rot    [rad]  angle of R_gt,i^T R_est,i
       Global consistency of the whole trajectory.

  RPE  relative pose error over a fixed travelled distance delta
       - rpe_trans  [m]    translation of the relative-pose residual
       - rpe_rot    [rad]  rotation of the same residual
       Local drift rate, independent of where earlier errors put the estimate.
       Reported per delta metres of travel (default 1 m), which is the fair
       comparison when platforms cover the route at different speeds.

ALIGNMENT
=========
ATE is reported two ways because they answer different questions:

  origin-aligned  the estimate is placed at the ground-truth start pose, and
                  nothing else is fitted.  This is what a navigation stack
                  actually experiences, and it is what the drift plots use.
  Umeyama-aligned the rigid SE(3) transform minimising the sum of squared
                  position residuals over the whole run (Umeyama 1991,
                  without scale, since these are metric sensors).  This is the
                  convention used by the TUM/KITTI tooling, so the numbers are
                  comparable with the SLAM literature.

Both are reported.  If they disagree strongly the run has a large constant
offset, which is worth knowing.

BODY-FRAME CORRECTION
=====================
Before any of this, both trajectories are rotated about their own z so that
each pose's x axis points along the direction that trajectory actually travels,
and the angle removed is reported as gt_yaw_offset_deg / est_yaw_offset_deg.

This is not cosmetic.  gt_traj.tum stores /gazebo/model_states unmodified, and
the Rocker-Bogie's URDF authors base_link facing -x, so its logged yaw is 180
deg from its heading.  align_origin places the estimate using T_gt[0], and rpe
takes relative motion in the body frame, so both inherited that 180 deg:
measured on the 2026-08-25 campaign, ATE translational RMSE read 56.33 m and
RPE 1.995 m/m for a run whose real trajectory error was 0.31 m.  1.995 is not a
coincidence - it is the error of travelling one metre backwards, per metre.

Expect gt_yaw_offset_deg near 0 for the differential and the tracked platform
and near 180 for the Rocker-Bogie.  A value that is neither, or one that moves
between runs of the same platform, means the estimate is not picking up a fixed
convention and the numbers should not be trusted.
"""

import math

import numpy as np

from metrics_io import ventanas_alta_frecuencia, vibration_magnitude


# ---------------------------------------------------------------------------
# rotation helpers
# ---------------------------------------------------------------------------

def quat_to_rot(q):
    """(qx, qy, qz, qw) -> 3x3 rotation matrix."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def rot_to_angle(R):
    """Magnitude of the rotation encoded by R, in radians, in [0, pi]."""
    t = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(t, -1.0, 1.0)))


def rpy_to_rot(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def poses_from_rpy(x, y, z, roll, pitch, yaw):
    """Stack of Nx4x4 homogeneous poses from position + RPY arrays."""
    n = len(x)
    T = np.zeros((n, 4, 4))
    T[:, 3, 3] = 1.0
    T[:, 0, 3] = x
    T[:, 1, 3] = y
    T[:, 2, 3] = z
    for i in range(n):
        T[i, :3, :3] = rpy_to_rot(roll[i], pitch[i], yaw[i])
    return T


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------

def umeyama(src, dst, with_scale=False):
    """Rigid transform mapping src onto dst (both Nx3), least squares.

    Returns (R, t, s).  Umeyama (1991); the SVD-based solution, with the
    reflection guard that a naive SVD solution gets wrong.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d

    cov = (dc.T @ sc) / len(src)
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    if with_scale:
        var_s = (sc ** 2).sum() / len(src)
        s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def heading_offset(T, min_travel_m=1.0):
    """Constant angle between a trajectory's logged yaw and where it travels.

    Returns radians, positive when the logged yaw leads the direction of
    travel.  Near 0 for a platform whose base_link faces the way it drives;
    near pi for one that does not.

    Why this is estimated instead of read from a per-platform constant: the
    constant exists (base_yaw_offset, in run_campaign.sh) but nothing forces
    the evaluation to agree with it, and a platform added later would inherit
    the bug silently.  Measuring it from the trajectory cannot go out of date.

    Each sample is WEIGHTED by the distance it covered, so a stationary
    vehicle - which has no direction of travel - contributes nothing without
    needing a cutoff anywhere.

    THE CUTOFF THIS REPLACES.  The previous version kept only samples whose
    step exceeded a fixed 20 mm.  That is not scale free: metrics.csv is
    logged at 50 Hz, so at 0.45 m/s a sample advances 9 mm and the filter
    rejected the ENTIRE trajectory, fell through its `< 20 samples` guard and
    returned 0.0.  Silently, and only where it mattered - 0.0 is the right
    answer for the two platforms whose base_link faces forward, so the defect
    was invisible on them.  Measured 2026-09-02 on the SAME rocker_bogie
    route: over one lap 1 sample cleared 20 mm, no correction was applied and
    ATE read 61.03 m with RPE 2.0006 m/m; over two laps 32 samples cleared it,
    the correction fired and ATE read 10.88 m.  The published number depended
    on how many samples happened to cross the cutoff.

    The average is circular, so it does not care that the values straddle
    +-pi.  A trajectory that never covers min_travel_m has no heading to
    measure and returns 0.0.
    """
    d = np.diff(T[:, :2, 3], axis=0)
    step = np.hypot(d[:, 0], d[:, 1])
    if step.sum() < min_travel_m:
        return 0.0
    travel = np.arctan2(d[:, 1], d[:, 0])
    yaw = np.arctan2(T[:-1, 1, 0], T[:-1, 0, 0])
    diff = yaw - travel
    return float(np.arctan2(np.sum(step * np.sin(diff)),
                            np.sum(step * np.cos(diff))))

def correct_body_frame(T, offset_rad):
    """Rotate each pose about its own z so its x axis points along travel.

    Post-multiplied, not pre-multiplied: the defect is a body-frame convention
    - base_link built facing the wrong way - not a world-frame rotation.  For
    a pure-yaw pose the two are identical; they differ once roll and pitch are
    present, which on this terrain they always are.
    """
    if abs(offset_rad) < 1e-9:
        return T
    c, s = math.cos(-offset_rad), math.sin(-offset_rad)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = T.copy()
    out[:, :3, :3] = np.einsum('nij,jk->nik', T[:, :3, :3], Rz)
    return out


def align_origin(T_est, T_gt):
    """Place the estimate at the ground-truth start pose."""
    delta = T_gt[0] @ np.linalg.inv(T_est[0])
    return np.einsum('ij,njk->nik', delta, T_est)


def align_umeyama(T_est, T_gt):
    """Least-squares rigid alignment of the whole trajectory."""
    R, t, _ = umeyama(T_est[:, :3, 3], T_gt[:, :3, 3])
    delta = np.eye(4)
    delta[:3, :3] = R
    delta[:3, 3] = t
    return np.einsum('ij,njk->nik', delta, T_est)


# ---------------------------------------------------------------------------
# ATE / RPE
# ---------------------------------------------------------------------------

def ate(T_est_aligned, T_gt):
    """Per-sample absolute trajectory error, split into translation/rotation."""
    dt = T_gt[:, :3, 3] - T_est_aligned[:, :3, 3]
    trans = np.linalg.norm(dt, axis=1)
    rot = np.array([rot_to_angle(T_gt[i, :3, :3].T @ T_est_aligned[i, :3, :3])
                    for i in range(len(T_gt))])
    return trans, rot


def _cumulative_distance(T):
    d = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def rpe(T_est, T_gt, delta_m=1.0):
    """Relative pose error over `delta_m` metres of ground-truth travel.

    Using travelled distance rather than a sample count keeps the comparison
    fair when the three platforms move at different speeds: a slower robot
    would otherwise be evaluated over shorter relative motions and look better.

    Returns (rpe_trans [m per delta], rpe_rot [rad per delta], pair_count).
    """
    s = _cumulative_distance(T_gt)
    n = len(s)
    if n < 2 or s[-1] < delta_m:
        return np.array([]), np.array([]), 0

    # For each i, the first j with s[j] - s[i] >= delta_m.
    j_of = np.searchsorted(s, s + delta_m, side='left')
    pairs = [(i, j) for i, j in enumerate(j_of) if j < n]
    if not pairs:
        return np.array([]), np.array([]), 0

    trans = np.empty(len(pairs))
    rot = np.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        gt_rel = np.linalg.inv(T_gt[i]) @ T_gt[j]
        est_rel = np.linalg.inv(T_est[i]) @ T_est[j]
        err = np.linalg.inv(gt_rel) @ est_rel
        trans[k] = np.linalg.norm(err[:3, 3])
        rot[k] = rot_to_angle(err[:3, :3])
    return trans, rot, len(pairs)


def rmse(a):
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float('nan')


def summarize(trans, rot, prefix):
    """mean / rmse / max for a translation+rotation error pair."""
    out = {}
    if trans.size:
        out[prefix + '_trans_mean'] = float(np.mean(trans))
        out[prefix + '_trans_rmse'] = rmse(trans)
        out[prefix + '_trans_max'] = float(np.max(trans))
    if rot.size:
        out[prefix + '_rot_mean'] = float(np.mean(rot))
        out[prefix + '_rot_rmse'] = rmse(rot)
        out[prefix + '_rot_max'] = float(np.max(rot))
    return out


def evaluate(df, rpe_delta_m=1.0):
    """Full trajectory evaluation for one run.

    `df` must carry gt_x/gt_y/gt_z, gt_roll/gt_pitch/gt_yaw and the matching
    odom_* columns (already unit-stripped by metrics_io.load_metrics).
    """
    need = ['gt_x', 'gt_y', 'gt_z', 'odom_x', 'odom_y', 'odom_z']
    if any(c not in df.columns for c in need) or len(df) < 10:
        return None

    def col(name, default=0.0):
        return (df[name].values if name in df.columns
                else np.full(len(df), default))

    T_gt = poses_from_rpy(df['gt_x'].values, df['gt_y'].values,
                          df['gt_z'].values, col('gt_roll'),
                          col('gt_pitch'), col('gt_yaw'))
    T_est = poses_from_rpy(df['odom_x'].values, df['odom_y'].values,
                           df['odom_z'].values, col('odom_roll'),
                           col('odom_pitch'), col('odom_yaw'))

    # A platform whose base_link does not face the way it drives logs a yaw that
    # is a constant angle from its direction of travel, and align_origin, rpe
    # and final_drift all take that yaw as their frame of reference.  Measured
    # on rocker_bogie, whose URDF faces -x: ATE 56.33 m and RPE 1.995 m/m, both
    # pure artefact, against a true 0.31 m.  Correct both trajectories onto
    # their own direction of travel before anything is measured.
    res = {}
    off_gt = heading_offset(T_gt)
    off_est = heading_offset(T_est)
    res['gt_yaw_offset_deg'] = float(np.degrees(off_gt))
    res['est_yaw_offset_deg'] = float(np.degrees(off_est))
    T_gt = correct_body_frame(T_gt, off_gt)
    T_est = correct_body_frame(T_est, off_est)

    T_org = align_origin(T_est, T_gt)
    tr, ro = ate(T_org, T_gt)
    res.update(summarize(tr, ro, 'ate_origin'))
    res['ate_trans_series'] = tr
    res['ate_rot_series'] = ro

    T_um = align_umeyama(T_est, T_gt)
    tr_u, ro_u = ate(T_um, T_gt)
    res.update(summarize(tr_u, ro_u, 'ate_umeyama'))

    rt, rr, npairs = rpe(T_est, T_gt, delta_m=rpe_delta_m)
    res.update(summarize(rt, rr, 'rpe'))
    res['rpe_pairs'] = npairs
    res['rpe_delta_m'] = rpe_delta_m
    # Serie por muestra del RPE rotacional, para poder mirarlo POR VENTANA en
    # vez de solo como un RMSE de toda la corrida.  rpe() empareja la muestra i
    # con la primera que esta delta_m mas alla; como la distancia acumulada no
    # decrece, los pares que existen son exactamente los primeros len(rr), y al
    # final de la trayectoria ya no queda un metro por delante.  Ese hueco se
    # deja en NaN, no en cero: un cero se promediaria como si fuese un acierto.
    serie_rot = np.full(len(T_gt), np.nan)
    serie_rot[:len(rr)] = np.degrees(rr)
    res['rpe_rot_series'] = serie_rot
    serie_tr = np.full(len(T_gt), np.nan)
    serie_tr[:len(rt)] = rt
    res['rpe_trans_series'] = serie_tr

    dist = _cumulative_distance(T_gt)
    res['gt_distance_m'] = float(dist[-1])
    res['distance_series'] = dist
    res['final_drift_m'] = float(np.linalg.norm(
        T_gt[-1, :3, 3] - T_org[-1, :3, 3]))
    # Drift expressed as a fraction of distance travelled is the number that
    # can be compared against other papers.
    res['drift_pct_of_distance'] = (
        100.0 * res['final_drift_m'] / dist[-1] if dist[-1] > 0 else float('nan'))

    return res


# ---------------------------------------------------------------------------
# stability <-> SLAM-error correlation
# ---------------------------------------------------------------------------

def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan'), 0
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def partial_correlation(x, y, z):
    """r(x, y | z): la correlacion que queda tras descontar z de ambas.

    POR QUE HACE FALTA AQUI.  La agitacion del chasis y el error por metro
    comparten una causa obvia: la velocidad.  Ir mas rapido zarandea mas el
    chasis Y deja menos barridos por metro recorrido, asi que una r cruda
    entre agitacion y RPE puede ser en buena parte la velocidad mirandose al
    espejo.  La parcial responde a la pregunta que de verdad se hace el
    paper: a IGUALDAD de velocidad, un chasis mas agitado estima peor?

    Formula estandar a partir de las tres correlaciones simples; no hace
    falta scipy.  Devuelve (r, n) como pearson().
    """
    r_xy, n = pearson(x, y)
    r_xz, _ = pearson(x, z)
    r_yz, _ = pearson(y, z)
    if not all(np.isfinite(v) for v in (r_xy, r_xz, r_yz)):
        return float('nan'), n
    den = math.sqrt(max(0.0, (1.0 - r_xz ** 2) * (1.0 - r_yz ** 2)))
    if den < 1e-12:
        return float('nan'), n
    return float((r_xy - r_xz * r_yz) / den), n


def bootstrap_ci(x, y, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for Pearson r.

    Used instead of an analytic p-value so that no scipy dependency is needed;
    it also makes no normality assumption, which windowed IMU statistics
    certainly do not satisfy.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    rs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xs, ys = x[idx], y[idx]
        if np.std(xs) == 0 or np.std(ys) == 0:
            rs[b] = np.nan
        else:
            rs[b] = np.corrcoef(xs, ys)[0, 1]
    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        return float('nan'), float('nan')
    return (float(np.percentile(rs, 100 * alpha / 2)),
            float(np.percentile(rs, 100 * (1 - alpha / 2))))


def _media_sin_nan(v):
    """Media de v ignorando los NaN; NaN si no queda ninguna muestra.

    np.nanmean sobre un vector entero de NaN avisa por stderr y devuelve NaN
    de todas formas; la ultima ventana de la corrida, que se queda sin par de
    RPE, es exactamente ese caso y saldria en cada corrida.
    """
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float('nan')


def stability_error_correlation(df, ate_trans, window_s=None, hr=None,
                                ate_rot=None, rpe_rot=None,
                                rpe_trans=None, con_ci=True):
    """Correlate chassis attitude agitation with instantaneous SLAM error.

    This is the quantitative evidence for the paper's central claim that
    stability drives SLAM performance.  Both signals are reduced to
    non-overlapping windows of `window_s` seconds:

        predictor : std(pitch) and std(roll) inside the window, and the RMS of
                    their rates of change - i.e. how much the mast moved, not
                    where it was;
        response  : the mean instantaneous ATE inside the same window, and the
                    growth of that error across the window (which is closer to
                    a local drift rate than the absolute value).

    Correlating windowed statistics rather than raw samples matters: the raw
    ATE is a slowly-drifting integral, so a sample-wise correlation against a
    zero-mean vibration signal is dominated by autocorrelation and reports a
    meaningless number.
    """
    if 'timestamp' not in df.columns or len(df) < 20:
        return None

    t = df['timestamp'].values
    t = t - t[0]
    n = min(len(t), len(ate_trans))
    t = t[:n]
    err = np.asarray(ate_trans[:n], dtype=float)
    # ATE rotacional por ventana, en grados.  Se anade 2026-09-02: la
    # correlacion se venia mirando solo contra el error de TRASLACION, y la
    # pregunta que interesa -si el chasis sacudido degrada la estimacion- se
    # responde antes en el rumbo, que es lo que un LiDAR de barrido pierde
    # primero.  None en corridas viejas y entonces la columna sale NaN.
    err_ro = (np.degrees(np.asarray(ate_rot[:n], dtype=float))
              if ate_rot is not None else np.full(n, np.nan))
    # RPE rotacional por ventana, en grados por metro recorrido.  Se anade
    # 2026-09-03.  El ATE rotacional de la linea anterior es un error
    # ABSOLUTO: cuando el rumbo ya se ha ido sigue siendo grande aunque el
    # tramo actual se estime perfecto, asi que correlacionarlo con lo que
    # agita el chasis AHORA mezcla la historia con el presente.  El RPE es
    # local por construccion -movimiento relativo sobre un metro- y es por
    # tanto el que responde a "esta sacudida, en esta ventana, cuanto costo".
    # None en corridas viejas, y entonces la columna sale NaN.
    rpe_ro = (np.asarray(rpe_rot[:n], dtype=float)
              if rpe_rot is not None else np.full(n, np.nan))
    rpe_tr = (np.asarray(rpe_trans[:n], dtype=float)
              if rpe_trans is not None else np.full(n, np.nan))
    # Velocidad por ventana, para poder descontarla en la correlacion parcial.
    # Del twist de la verdad-terreno cuando esta, que no lleva ruido de sensor.
    if 'gt_vx' in df.columns and 'gt_vy' in df.columns:
        vel = np.hypot(df['gt_vx'].values[:n], df['gt_vy'].values[:n])
    else:
        vel = np.full(n, np.nan)
    # Actitud de la verdad-terreno: ver la nota de VIBRACION mas abajo.
    pitch = (df['gt_pitch'] if 'gt_pitch' in df.columns else df['pitch']).values[:n]
    roll = (df['gt_roll'] if 'gt_roll' in df.columns else df['roll']).values[:n]

    # VIBRACION.  La misma definicion que las tablas y que las figuras, porque
    # sale de la misma funcion: metrics_io.vibration_magnitude, o sea |gt_a|.
    # La agitacion de actitud dice cuanto se BALANCEO el chasis; esto dice
    # cuanta aceleracion propia se llevo dentro de la ventana.
    #
    # Sin reserva al IMU: hasta 2026-09-04 este bloque caia a
    # `|(a_x, a_y, a_z - 9.81)|` en las corridas viejas, y esa cifra es en su
    # mayor parte la proyeccion de g que el cabeceo deja sin quitar, con un
    # RMS de ~1.9 m/s^2 en la ruta de la mina.  Ahora sale NaN, pearson()
    # descarta las ventanas no finitas y la casilla queda en n/a.
    vib_muestra = vibration_magnitude(df)[:n]

    # VENTANA.  100 ms cuando hay gt_highrate, 1 s cuando no.
    #
    # 1 s no corresponde a nada del sistema: el LiDAR barre a 10 Hz -100 ms
    # por nube- y el troceado emite a 400 Hz, asi que una ventana de 1 s
    # promedia diez barridos y borra la variacion por barrido, que es
    # justamente la que produce distorsion de movimiento.  Era 1 s porque a
    # los 50 Hz de metrics.csv una de 100 ms tiene 5 muestras, el corte de
    # m.sum() < 5.  Con gt_highrate a 1000 Hz lleva 100.
    hr_win = {}
    if window_s is None:
        window_s = 0.1 if hr is not None else 1.0
    if hr is not None:
        hr_win = ventanas_alta_frecuencia(hr, df['timestamp'].values[0],
                                          window_s)

    edges = np.arange(0.0, t[-1] + window_s, window_s)
    idx = np.digitize(t, edges) - 1

    # A 100 ms sobre metrics.csv a 50 Hz quedan 5 muestras: suficiente para
    # el crecimiento del error (e[-1] - e[0]) que es lo unico que sale de
    # aqui cuando la agitacion viene de gt_highrate.
    min_muestras = 3 if hr_win else 5

    rows = []
    for k in range(len(edges) - 1):
        m = idx == k
        if m.sum() < min_muestras:
            continue
        p, r, e = pitch[m], roll[m], err[m]
        clave = round(float(edges[k]), 6)
        if hr_win:
            if clave not in hr_win:
                continue
            agit, vib, _n = hr_win[clave]
            p_std = r_std = float('nan')
        else:
            p_std, r_std = float(np.std(p)), float(np.std(r))
            agit = float(np.hypot(p_std, r_std))
            v = vib_muestra[m]
            v = v[np.isfinite(v)]
            vib = float(np.sqrt(np.mean(v ** 2))) if v.size else float('nan')
        rows.append({
            'window_start_s': edges[k],
            'pitch_std_rad': p_std,
            'roll_std_rad': r_std,
            'vibration_rms_m_s2': vib,
            'attitude_agitation_rad': agit,
            'pitch_rate_rms_rad_s': float(np.sqrt(np.mean(
                (np.diff(p) / window_s * m.sum()) ** 2))) if m.sum() > 1 else 0.0,
            'ate_mean_m': float(np.mean(e)),
            'ate_growth_m': float(e[-1] - e[0]),
            'ate_rot_mean_deg': float(np.mean(err_ro[m])),
            'rpe_rot_mean_deg_m': _media_sin_nan(rpe_ro[m]),
            'rpe_trans_mean_m_m': _media_sin_nan(rpe_tr[m]),
            'speed_mean_m_s': _media_sin_nan(vel[m]),
        })

    if len(rows) < 5:
        return None

    def colof(key):
        return np.array([r[key] for r in rows])

    out = {'windows': rows, 'window_s': window_s, 'n_windows': len(rows)}
    vel_w = colof('speed_mean_m_s')
    for pred in ('pitch_std_rad', 'roll_std_rad', 'attitude_agitation_rad',
                 'vibration_rms_m_s2'):
        for resp in ('ate_mean_m', 'ate_growth_m', 'ate_rot_mean_deg',
                     'rpe_rot_mean_deg_m', 'rpe_trans_mean_m_m'):
            r, n_used = pearson(colof(pred), colof(resp))
            out['r_{}__{}'.format(pred, resp)] = r
            out['n_{}__{}'.format(pred, resp)] = n_used
            # La misma r con la velocidad descontada de ambos lados.
            pr, _ = partial_correlation(colof(pred), colof(resp), vel_w)
            out['pr_{}__{}'.format(pred, resp)] = pr
            # El bootstrap es lo caro; el barrido de ventanas lo apaga.
            if con_ci:
                lo, hi = bootstrap_ci(colof(pred), colof(resp))
            else:
                lo = hi = float('nan')
            out['ci_{}__{}'.format(pred, resp)] = (lo, hi)
    return out
