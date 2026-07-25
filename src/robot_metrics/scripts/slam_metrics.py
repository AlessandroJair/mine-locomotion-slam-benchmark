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
"""

import numpy as np


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

    res = {}

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


def stability_error_correlation(df, ate_trans, window_s=1.0):
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
    pitch = df['pitch'].values[:n]
    roll = df['roll'].values[:n]

    edges = np.arange(0.0, t[-1] + window_s, window_s)
    idx = np.digitize(t, edges) - 1

    rows = []
    for k in range(len(edges) - 1):
        m = idx == k
        if m.sum() < 5:
            continue
        p, r, e = pitch[m], roll[m], err[m]
        rows.append({
            'window_start_s': edges[k],
            'pitch_std_rad': float(np.std(p)),
            'roll_std_rad': float(np.std(r)),
            'attitude_agitation_rad': float(np.hypot(np.std(p), np.std(r))),
            'pitch_rate_rms_rad_s': float(np.sqrt(np.mean(
                (np.diff(p) / window_s * m.sum()) ** 2))) if m.sum() > 1 else 0.0,
            'ate_mean_m': float(np.mean(e)),
            'ate_growth_m': float(e[-1] - e[0]),
        })

    if len(rows) < 5:
        return None

    def colof(key):
        return np.array([r[key] for r in rows])

    out = {'windows': rows, 'window_s': window_s, 'n_windows': len(rows)}
    for pred in ('pitch_std_rad', 'roll_std_rad', 'attitude_agitation_rad'):
        for resp in ('ate_mean_m', 'ate_growth_m'):
            r, n_used = pearson(colof(pred), colof(resp))
            lo, hi = bootstrap_ci(colof(pred), colof(resp))
            out['r_{}__{}'.format(pred, resp)] = r
            out['ci_{}__{}'.format(pred, resp)] = (lo, hi)
            out['n_{}__{}'.format(pred, resp)] = n_used
    return out
