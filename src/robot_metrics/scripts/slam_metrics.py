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

    # For each i, the first j with s[j] - s[i] >= delta_m.  s does not
    # decrease, so the i that have a partner are a prefix.
    j_of = np.searchsorted(s, s + delta_m, side='left')
    i = np.flatnonzero(j_of < n)
    if not i.size:
        return np.array([]), np.array([]), 0
    trans, rot = relative_error(T_est[i], T_est[j_of[i]], T_gt[i], T_gt[j_of[i]])
    return trans, rot, len(i)


def relative_error(E_i, E_j, G_i, G_j):
    """Residual of the relative motion i->j, estimate against ground truth,
    for stacks of poses.  Returns (translation [m], rotation angle [rad])."""
    inv = np.linalg.inv
    err = inv(inv(G_i) @ G_j) @ (inv(E_i) @ E_j)
    tr = np.einsum('nii->n', err[:, :3, :3])
    return (np.linalg.norm(err[:, :3, 3], axis=1),
            np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


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


def _pair_at_estimates(df, hr, col, tau=0.0):
    """Una muestra por estimacion de odometria, con la verdad-terreno en SU
    instante.

    POR QUE.  metrics.csv va a 50 Hz y la odometria llega a 10 Hz, asi que
    cada estimacion se repite ~5 filas mientras la verdad-terreno de esas filas
    sigue avanzando.  Emparejar fila a fila cuenta como error lo que es solo
    una estimacion que aun no se ha actualizado (hasta v * 100 ms).  Medido en
    la campaña de swept_lidar: el RPE salia ~4 % mas alto en traslacion y
    ~15 % en rotacion, igual en las tres plataformas.

    Se toma la fila en que aparece cada estimacion nueva y la verdad-terreno
    de gt_highrate (1 kHz) interpolada en ese instante.  Queda sin corregir la
    latencia entre el barrido y la recepcion de su odometria: el logger no
    guarda el stamp de cabecera, solo la hora de llegada.

    Con `tau` la verdad-terreno se toma `tau` segundos ANTES de la llegada de
    cada estimacion: la latencia supuesta entre el barrido y su odometria.

    Devuelve (indices de fila, T_est, T_gt) de las estimaciones.
    """
    o = np.column_stack([df['odom_x'].values, df['odom_y'].values,
                         df['odom_z'].values, col('odom_yaw')])
    idx = np.flatnonzero(np.r_[True, np.any(np.diff(o, axis=0) != 0, axis=1)])
    t = df['timestamp'].values[idx] - tau
    th = hr['timestamp'].values
    order = np.argsort(th)
    th = th[order]

    def g(k, ang=False):
        v = hr[k].values[order]
        return np.interp(t, th, np.unwrap(v) if ang else v)

    T_gt = poses_from_rpy(g('gt_x'), g('gt_y'), g('gt_z'), g('gt_roll', True),
                          g('gt_pitch', True), g('gt_yaw', True))
    T_est = poses_from_rpy(df['odom_x'].values[idx], df['odom_y'].values[idx],
                           df['odom_z'].values[idx], col('odom_roll')[idx],
                           col('odom_pitch')[idx], col('odom_yaw')[idx])
    return idx, T_est, T_gt


def _body(T):
    return correct_body_frame(T, heading_offset(T))


LATENCY_GRID_S = np.arange(0.0, 0.2501, 0.01)


def estimate_latency(df, hr, delta_m=1.0):
    """Retardo de la verdad-terreno que minimiza el RPE traslacional.

    El logger no guarda el stamp de cabecera de la odometria, solo su hora
    de llegada, y entre el barrido y la llegada pasan el ensamblado y el ICP.
    Sobre 1 m ese retardo pesa poco; sobre un barrido es mas de la mitad del
    error (medido: RMS rotacional por barrido 0.60 -> 0.25 deg en el
    diferencial al retrasar 108 ms).  Se estima por corrida en una rejilla de
    10 ms; en la campana swept_lidar sale 66-146 ms segun la plataforma.
    """
    best = (np.inf, 0.0)
    for tau in LATENCY_GRID_S:
        best = min(best, (rpe_at_delay(df, hr, tau, delta_m)[0], float(tau)))
    return best[1]


def _colfn(df):
    return lambda name, default=0.0: (df[name].values if name in df.columns
                                      else np.full(len(df), default))


def rpe_at_delay(df, hr, tau, delta_m=1.0):
    """(RMSE trans [m/m], RMSE rot [rad/m]) del RPE por estimacion con la
    verdad-terreno retrasada `tau`.  tau = 0 es el RPE de las tablas."""
    _, Te, Tg = _pair_at_estimates(df, hr, _colfn(df), tau)
    tr, ro, _ = rpe(_body(Te), _body(Tg), delta_m)
    return rmse(tr), rmse(ro)


def ate_offset_sensitivity(df, hr):
    """ATE alineado al inicio (RMSE, m) segun que correccion de rumbo reciba
    la ESTIMACION: la suya propia (la de las tablas), la de la verdad-terreno,
    o ninguna.  Critica de revision: si el delta de la estimacion absorbiera
    el sesgo de rumbo que domina el ATE, las tres diferirian mucho."""
    _, Te, Tg = _pair_at_estimates(df, hr, _colfn(df))
    d_gt, d_est = heading_offset(Tg), heading_offset(Te)
    G = correct_body_frame(Tg, d_gt)

    def ate_of(E):
        return rmse(ate(align_origin(E, G), G)[0])
    return {'est_offset_deg': float(np.degrees(d_est)),
            'ate_own': ate_of(correct_body_frame(Te, d_est)),
            'ate_gt_offset': ate_of(correct_body_frame(Te, d_gt)),
            'ate_uncorrected': ate_of(Te)}


def sweep_increments(df, hr, col, tau):
    """Error de cada barrido: el incremento entre dos estimaciones seguidas
    contra el de la verdad-terreno retrasada `tau`.

    Es la respuesta de la correlacion por ventanas.  La anterior, el RPE del
    metro SIGUIENTE, mira ~2 s (unos 22 barridos) hacia delante desde una
    ventana de 100 ms: predictor y respuesta no cubrian el mismo intervalo
    (critica de revision).  Aqui el intervalo es el mismo: el barrido que
    produjo la estimacion k+1 ocupa, en la verdad-terreno retrasada, el
    intervalo [t_k, t_k+1).

    Devuelve dict de arrays por barrido: t0, t1 (hora de la verdad-terreno,
    ya retrasada), trans [m], rot [deg], speed [m/s], y el ATE (alineado al
    inicio) al final de cada barrido, para el crecimiento del ATE.
    """
    idx, Te, Tg = _pair_at_estimates(df, hr, col, tau)
    Te, Tg = _body(Te), _body(Tg)
    t = df['timestamp'].values[idx] - tau
    tr, ro = relative_error(Te[:-1], Te[1:], Tg[:-1], Tg[1:])
    dt = np.diff(t)
    ate_t, ate_r = ate(align_origin(Te, Tg), Tg)
    return {'t0': t[:-1], 't1': t[1:], 'trans': tr, 'rot': np.degrees(ro),
            'speed': np.linalg.norm(np.diff(Tg[:, :3, 3], axis=0), axis=1)
            / np.where(dt > 0, dt, np.nan),
            'ate': ate_t[1:], 'ate_prev': ate_t[:-1],
            'ate_rot': np.degrees(ate_r[1:])}


def evaluate(df, rpe_delta_m=1.0, hr=None):
    """Full trajectory evaluation for one run.

    `df` must carry gt_x/gt_y/gt_z, gt_roll/gt_pitch/gt_yaw and the matching
    odom_* columns (already unit-stripped by metrics_io.load_metrics).

    With `hr` (gt_highrate) the errors are computed once per odometry estimate
    against the interpolated ground truth (_pair_at_estimates) and the
    per-sample series are held over the 50 Hz rows while each estimate is in
    force.  Without it, rows are paired as logged.
    """
    need = ['gt_x', 'gt_y', 'gt_z', 'odom_x', 'odom_y', 'odom_z']
    if any(c not in df.columns for c in need) or len(df) < 10:
        return None

    def col(name, default=0.0):
        return (df[name].values if name in df.columns
                else np.full(len(df), default))

    T_gt_rows = poses_from_rpy(df['gt_x'].values, df['gt_y'].values,
                               df['gt_z'].values, col('gt_roll'),
                               col('gt_pitch'), col('gt_yaw'))
    if hr is not None:
        idx, T_est, T_gt = _pair_at_estimates(df, hr, col)
        # fila -> estimacion vigente en esa fila
        held = np.searchsorted(idx, np.arange(len(df)), side='right') - 1
    else:
        T_gt = T_gt_rows
        T_est = poses_from_rpy(df['odom_x'].values, df['odom_y'].values,
                               df['odom_z'].values, col('odom_roll'),
                               col('odom_pitch'), col('odom_yaw'))
        held = np.arange(len(df))

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
    res['ate_trans_series'] = tr[held]
    res['ate_rot_series'] = ro[held]

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
    res['rpe_rot_series'] = serie_rot[held]
    serie_tr = np.full(len(T_gt), np.nan)
    serie_tr[:len(rt)] = rt
    res['rpe_trans_series'] = serie_tr[held]

    # Distancia de la verdad-terreno fila a fila (50 Hz), que no depende del
    # emparejamiento; el drift final, de la ultima estimacion.
    dist = _cumulative_distance(T_gt_rows)
    res['gt_distance_m'] = float(dist[-1])
    res['distance_series'] = dist
    res['final_drift_m'] = float(np.linalg.norm(
        T_gt[-1, :3, 3] - T_org[-1, :3, 3]))
    # Drift expressed as a fraction of distance travelled is the number that
    # can be compared against other papers.
    res['drift_pct_of_distance'] = (
        100.0 * res['final_drift_m'] / dist[-1] if dist[-1] > 0 else float('nan'))

    # Error por barrido para la correlacion por ventanas, con la latencia
    # estimada y, como sensibilidad, sin ella.
    if hr is not None:
        res['latency_s'] = estimate_latency(df, hr, rpe_delta_m)
        res['sweeps'] = sweep_increments(df, hr, col, res['latency_s'])
        res['sweeps_noshift'] = sweep_increments(df, hr, col, 0.0)

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


def bootstrap_ci(x, y, n_boot=2000, alpha=0.05, seed=0, block=1):
    """Percentile bootstrap CI for Pearson r.

    Used instead of an analytic p-value so that no scipy dependency is needed;
    it also makes no normality assumption, which windowed IMU statistics
    certainly do not satisfy.

    `block` > 1 gives a moving-block bootstrap: consecutive windows are
    autocorrelated (the terrain changes over metres, not over one sweep), and
    resampling them one by one treats them as independent and makes the
    interval too narrow.  Measured on the 1 m RPE response: iid +-0.04,
    5 s blocks +-0.09.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return float('nan'), float('nan')
    block = int(min(max(block, 1), n))
    k = -(-n // block)
    rng = np.random.default_rng(seed)
    rs = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, n - block + 1, k)
        idx = (s[:, None] + np.arange(block)).ravel()[:n]
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
    yaw = np.unwrap((df['gt_yaw'] if 'gt_yaw' in df.columns
                     else df['yaw']).values[:n])

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
            agit, vib, _n, yaw_std = hr_win[clave]
            p_std = r_std = float('nan')
        else:
            p_std, r_std = float(np.std(p)), float(np.std(r))
            yaw_std = float(np.std(yaw[m]))
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
            'yaw_agitation_rad': yaw_std,
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
    out = _correlate(rows, ('ate_mean_m', 'ate_growth_m', 'ate_rot_mean_deg',
                            'rpe_rot_mean_deg_m', 'rpe_trans_mean_m_m'), con_ci)
    out['window_s'] = window_s
    return out


BLOCK_S = 5.0


def sweep_error_correlation(sw, hr, group=1, con_ci=True, cmd=None):
    """Agitacion contra el error de los MISMOS barridos (sweep_increments).

    Cada ventana son `group` barridos consecutivos: los predictores salen de
    gt_highrate sobre [t0 del primero, t1 del ultimo) y la respuesta es la
    SUMA del error de esos barridos, asi que predictor y respuesta cubren
    siempre el mismo intervalo.  Con group > 1 la r sube, pero eso es lo que
    se espera al promediar el ruido de registro de cada barrido, que no
    depende de la agitacion; no demuestra por si solo un efecto acumulativo.

    Barridos de menos de 50 ms o mas de 200 ms (huecos de la odometria) o con
    menos de 20 muestras de verdad-terreno se descartan, y un grupo no cruza
    un barrido descartado.

    `cmd` = (t, omega_cmd) de metrics.csv: con el, la agitacion de guinada se
    da tambien sin el giro MANDADO, sigma(psi - integral de omega_cmd)
    (critica de revision: sigma_psi incluye las curvas de la ruta).
    """
    t0, t1 = sw['t0'], sw['t1']
    dt = t1 - t0
    th = hr['timestamp'].values
    o = np.argsort(th)
    th = th[o]
    p = np.unwrap(hr['gt_pitch'].values[o])
    r = np.unwrap(hr['gt_roll'].values[o])
    y = np.unwrap(hr['gt_yaw'].values[o])
    if cmd is not None:
        w = np.interp(th, cmd[0], cmd[1])
        yr = y - np.r_[0.0, np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(th))]
    else:
        yr = np.full_like(y, np.nan)
    a2 = vibration_magnitude(hr, 'gt_highrate.csv')[o] ** 2
    fin = np.isfinite(a2)
    csum = {k: np.r_[0.0, np.cumsum(v)] for k, v in (
        ('p', p), ('p2', p * p), ('r', r), ('r2', r * r), ('y', y),
        ('y2', y * y), ('yr', yr), ('yr2', yr * yr),
        ('a2', np.where(fin, a2, 0.0)), ('na', fin * 1.0))}
    i_of = np.searchsorted(th, t0)
    j_of = np.searchsorted(th, t1)
    ok = (dt > 0.05) & (dt < 0.2) & (j_of - i_of > 20)

    # grupos de `group` barridos validos y contiguos
    runs = np.split(np.arange(len(ok)), np.flatnonzero(np.diff(ok * 1)) + 1)
    grupos = [g[k:k + group] for g in runs if ok[g[0]]
              for k in range(0, len(g) - group + 1, group)]
    rows = []
    for g in grupos:
        i, j = i_of[g[0]], j_of[g[-1]]
        n = j - i

        def std(k):
            m = (csum[k][j] - csum[k][i]) / n
            return math.sqrt(max((csum[k + '2'][j] - csum[k + '2'][i]) / n - m * m, 0.0))
        na = csum['na'][j] - csum['na'][i]
        rows.append({
            'window_start_s': float(t0[g[0]]),
            'attitude_agitation_rad': math.hypot(std('p'), std('r')),
            'yaw_agitation_rad': std('y'),
            'yaw_resid_agitation_rad': std('yr'),
            'vibration_rms_m_s2': (math.sqrt((csum['a2'][j] - csum['a2'][i]) / na)
                                   if na else float('nan')),
            'sweep_rot_deg': float(sw['rot'][g].sum()),
            'sweep_trans_m': float(sw['trans'][g].sum()),
            'ate_mean_m': float(sw['ate'][g].mean()),
            'ate_growth_m': float(sw['ate'][g[-1]] - sw['ate_prev'][g[0]]),
            'ate_rot_mean_deg': float(sw['ate_rot'][g].mean()),
            'speed_mean_m_s': float(np.nanmean(sw['speed'][g])),
        })
    if len(rows) < 5:
        return None
    win = float(np.median(dt[ok])) * group
    out = _correlate(rows, ('ate_mean_m', 'ate_growth_m', 'ate_rot_mean_deg',
                            'sweep_rot_deg', 'sweep_trans_m'), con_ci,
                     block=max(1, int(round(BLOCK_S / win))))
    out.update(window_s=win, group=group, per_sweep=True)
    return out


def _correlate(rows, responses, con_ci, block=1):
    """r, r|velocidad, r|otra agitacion y su IC para cada par
    predictor/respuesta de las ventanas `rows`."""
    def colof(key):
        return np.array([r.get(key, np.nan) for r in rows], dtype=float)

    out = {'windows': rows, 'n_windows': len(rows), 'block': block}
    vel_w = colof('speed_mean_m_s')
    for pred in ('pitch_std_rad', 'roll_std_rad', 'attitude_agitation_rad',
                 'yaw_agitation_rad', 'yaw_resid_agitation_rad',
                 'vibration_rms_m_s2'):
        for resp in responses:
            r, n_used = pearson(colof(pred), colof(resp))
            out['r_{}__{}'.format(pred, resp)] = r
            out['n_{}__{}'.format(pred, resp)] = n_used
            # La misma r con la velocidad descontada de ambos lados.
            pr, _ = partial_correlation(colof(pred), colof(resp), vel_w)
            out['pr_{}__{}'.format(pred, resp)] = pr
            # Guinada con la actitud descontada, y al reves: lo que cada
            # agitacion explica del RPE que la otra no explica ya.  Es la
            # prueba de si la guinada DOMINA, no solo de si correlaciona.
            otra = {'yaw_agitation_rad': 'attitude_agitation_rad',
                    'yaw_resid_agitation_rad': 'attitude_agitation_rad',
                    'attitude_agitation_rad': 'yaw_agitation_rad'}.get(pred)
            if otra:
                out['pa_{}__{}'.format(pred, resp)], _ = partial_correlation(
                    colof(pred), colof(resp), colof(otra))
            # El bootstrap es lo caro; el barrido de ventanas lo apaga.
            if con_ci:
                lo, hi = bootstrap_ci(colof(pred), colof(resp), block=block)
            else:
                lo = hi = float('nan')
            out['ci_{}__{}'.format(pred, resp)] = (lo, hi)
    return out
