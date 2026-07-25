#!/usr/bin/env python3
"""Single-run locomotion and SLAM plots.

NOTE: for the paper's tables use aggregate_runs.py instead.  This script
analyses ONE csv per robot and reports no repeatability; aggregate_runs.py
reads the per-run directories written by metrics_logger.py, reports mean +/-
standard deviation over the three repetitions, and splits ATE and RPE into
translational and rotational components.  This one is kept for quick looks at
a single run.
"""

#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from metrics_io import load_metrics  # noqa: E402

# ---- Journal-quality plot style (single-column paper) ----
COLUMN_WIDTH_IN = 6.5   # single-column text width (inches)
FIG_HEIGHT_SINGLE = 3.0
FIG_HEIGHT_DOUBLE = 5.0
FIG_HEIGHT_SQUARE = 5.5

LINE_STYLES = ['-', '--', '-.']
MARKERS = ['o', 's', '^']
COLORS = ['#1f77b4', '#d62728', '#2ca02c']  # blue, red, green

matplotlib.rcParams.update({
    # LaTeX-like serif font
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    # Font sizes
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    # Lines
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    # Axes
    'axes.linewidth': 0.6,
    'axes.grid': True,
    'grid.alpha': 1.0,
    'grid.color': '0.85',
    'grid.linewidth': 0.4,
    # Ticks
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    # Legend
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': '0.8',
    'legend.fancybox': False,
    # Save
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
})


GRAVITY = 9.81
SLIP_CMD_THRESHOLD = 0.05  # m/s minimum commanded velocity to compute slip
WARMUP_SECONDS = 5.0  # seconds to discard after spawn (solver transients)


def load_csv(filepath):
    # Read through metrics_io so the unit suffixes in the header ("pitch[rad]")
    # are stripped consistently and older, pre-units CSVs still load.
    df = load_metrics(filepath)
    # Normalize time to start at 0
    df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    total_duration = df['time'].iloc[-1]
    # Discard initial warmup period (Gazebo spawn transients)
    if total_duration > WARMUP_SECONDS * 2:
        df = df[df['time'] >= WARMUP_SECONDS].reset_index(drop=True)
        df['time'] = df['time'] - df['time'].iloc[0]
    else:
        print("Warning: {} is only {:.1f}s long, skipping warmup trim.".format(
            filepath, total_duration))
    return df


def has_slam_columns(df):
    slam_cols = ['loop_closure_count', 'slam_inliers', 'tf_correction_magnitude']
    return all(c in df.columns for c in slam_cols)


def compute_stability(df):
    pitch = df['pitch'].values
    roll = df['roll'].values if 'roll' in df.columns else np.zeros(len(df))
    yaw = df['yaw'].values if 'yaw' in df.columns else np.zeros(len(df))
    ax = df['accel_x'].values
    ay = df['accel_y'].values if 'accel_y' in df.columns else np.zeros(len(df))
    az = df['accel_z'].values
    angular_accel_x = df['angular_accel_x'].values if 'angular_accel_x' in df.columns else np.zeros(len(df))
    angular_accel_z = df['angular_accel_z'].values if 'angular_accel_z' in df.columns else np.zeros(len(df))

    return {
        'pitch_mean': np.mean(pitch),
        'pitch_std': np.std(pitch),
        'pitch_max_abs': np.max(np.abs(pitch)),
        'roll_std': np.std(roll),
        'roll_max_abs': np.max(np.abs(roll)),
        'yaw_max_abs': np.max(np.abs(yaw)),
        'accel_x_rms': np.sqrt(np.mean(ax**2)),
        'accel_x_max': np.max(np.abs(ax)),
        'accel_y_rms': np.sqrt(np.mean(ay**2)),
        'accel_z_rms': np.sqrt(np.mean((az - GRAVITY)**2)),
        'accel_z_max': np.max(np.abs(az - GRAVITY)),
        'vibration_rms': np.sqrt(np.mean(ax**2 + ay**2 + (az - GRAVITY)**2)),
        'angular_accel_x_max': np.max(np.abs(angular_accel_x)),
        'angular_accel_z_max': np.max(np.abs(angular_accel_z)),
    }


def compute_slip(df):
    mask = np.abs(df['cmd_vx']) > SLIP_CMD_THRESHOLD
    if mask.sum() == 0:
        return np.array([]), 0.0

    cmd = df.loc[mask, 'cmd_vx'].values
    actual = df.loc[mask, 'odom_vx'].values
    slip = 1.0 - (actual / cmd)
    slip_pct = slip * 100.0
    return slip_pct, np.mean(slip_pct)


def compute_trajectory(df):
    dx = np.diff(df['odom_x'].values)
    dy = np.diff(df['odom_y'].values)
    increments = np.sqrt(dx**2 + dy**2)
    total_distance = np.sum(increments)

    final_x = df['odom_x'].iloc[-1]
    final_y = df['odom_y'].iloc[-1]
    start_x = df['odom_x'].iloc[0]
    start_y = df['odom_y'].iloc[0]
    drift = np.sqrt((final_x - start_x)**2 + (final_y - start_y)**2)

    return {
        'total_distance': total_distance,
        'final_drift': drift,
    }


def has_ground_truth(df):
    gt_cols = ['gt_x', 'gt_y', 'gt_z']
    return all(c in df.columns for c in gt_cols)


def align_gt_to_odom(df):
    """Align ground truth trajectory to the odometry frame using SE(2).

    The GT starts at (gt_x0, gt_y0) with yaw0 in world frame, while odom
    starts at (odom_x0, odom_y0).  We translate GT to its origin and rotate
    by -yaw0 so that both trajectories start at the origin and face the same
    direction as the odom frame.
    """
    # Initial poses
    gt_x0, gt_y0 = df['gt_x'].iloc[0], df['gt_y'].iloc[0]
    odom_x0, odom_y0 = df['odom_x'].iloc[0], df['odom_y'].iloc[0]

    # Initial yaw of GT (used to rotate GT into odom frame)
    if 'gt_yaw' in df.columns:
        yaw0 = df['gt_yaw'].iloc[0]
    else:
        yaw0 = 0.0

    # Initial yaw of odom
    # odom frame yaw at start is 0 by definition for most SLAM systems,
    # but if odom has an initial yaw we need to account for it too.
    # We compute the relative rotation: odom_yaw0 - gt_yaw0
    # For simplicity, we rotate GT by -gt_yaw0 (bringing it to 0) then
    # the odom is already near 0, so both are aligned.
    cos_y = np.cos(-yaw0)
    sin_y = np.sin(-yaw0)

    # Translate GT to origin, then rotate
    dx = df['gt_x'].values - gt_x0
    dy = df['gt_y'].values - gt_y0
    gt_x_aligned = cos_y * dx - sin_y * dy
    gt_y_aligned = sin_y * dx + cos_y * dy
    gt_z_aligned = df['gt_z'].values - df['gt_z'].iloc[0]

    # Translate odom to origin (no rotation needed, odom frame is reference)
    odom_x_aligned = df['odom_x'].values - odom_x0
    odom_y_aligned = df['odom_y'].values - odom_y0
    odom_z_aligned = df['odom_z'].values - df['odom_z'].iloc[0]

    return (gt_x_aligned, gt_y_aligned, gt_z_aligned,
            odom_x_aligned, odom_y_aligned, odom_z_aligned)


def compute_ate(df):
    """Absolute Trajectory Error: Euclidean distance between aligned gt and odom."""
    if not has_ground_truth(df) or len(df) < 3:
        return None
    gt_x, gt_y, gt_z, odom_x, odom_y, odom_z = align_gt_to_odom(df)
    dx = gt_x - odom_x
    dy = gt_y - odom_y
    dz = gt_z - odom_z
    errors = np.sqrt(dx**2 + dy**2 + dz**2)
    return {
        'ate_mean': np.mean(errors),
        'ate_max': np.max(errors),
        'ate_rmse': np.sqrt(np.mean(errors**2)),
        'ate_errors': errors,
    }


def compute_rpe(df):
    """Relative Pose Error: error between consecutive relative poses (gt vs estimated)."""
    if not has_ground_truth(df) or len(df) < 3:
        return None
    gt_x, gt_y, _, odom_x, odom_y, _ = align_gt_to_odom(df)
    # Relative displacements
    gt_dx = np.diff(gt_x)
    gt_dy = np.diff(gt_y)
    odom_dx = np.diff(odom_x)
    odom_dy = np.diff(odom_y)
    # RPE per step
    err_x = gt_dx - odom_dx
    err_y = gt_dy - odom_dy
    errors = np.sqrt(err_x**2 + err_y**2)
    return {
        'rpe_mean': np.mean(errors),
        'rpe_max': np.max(errors),
        'rpe_rmse': np.sqrt(np.mean(errors**2)),
    }


def compute_slam_metrics(df):
    if not has_slam_columns(df):
        return None

    lc = df['loop_closure_count'].values
    total_loop_closures = int(lc[-1]) if len(lc) > 0 else 0

    prox = df['proximity_detection_count'].values
    total_proximity = int(prox[-1]) if len(prox) > 0 else 0

    inliers = df['slam_inliers'].values
    matches = df['slam_matches'].values
    icp_inliers_ratio = df['slam_icp_inliers_ratio'].values if 'slam_icp_inliers_ratio' in df.columns else df['slam_icp_rms'].values if 'slam_icp_rms' in df.columns else np.zeros(len(df))
    icp_correspondences = df['slam_icp_correspondences'].values if 'slam_icp_correspondences' in df.columns else np.zeros(len(df))
    proc_time = df['slam_processing_time'].values
    tf_corr = df['tf_correction_magnitude'].values
    tf_yaw = df['tf_correction_yaw'].values if 'tf_correction_yaw' in df.columns else np.zeros(len(df))

    # Filter non-zero processing times for stats
    proc_nonzero = proc_time[proc_time > 0]

    return {
        'loop_closures': total_loop_closures,
        'proximity_detections': total_proximity,
        'inliers_mean': np.mean(inliers[inliers > 0]) if np.any(inliers > 0) else 0.0,
        'inliers_std': np.std(inliers[inliers > 0]) if np.any(inliers > 0) else 0.0,
        'matches_mean': np.mean(matches[matches > 0]) if np.any(matches > 0) else 0.0,
        'icp_inliers_ratio_mean': np.mean(icp_inliers_ratio[icp_inliers_ratio > 0]) if np.any(icp_inliers_ratio > 0) else 0.0,
        'icp_inliers_ratio_std': np.std(icp_inliers_ratio[icp_inliers_ratio > 0]) if np.any(icp_inliers_ratio > 0) else 0.0,
        'icp_correspondences_mean': np.mean(icp_correspondences[icp_correspondences > 0]) if np.any(icp_correspondences > 0) else 0.0,
        'proc_time_mean': np.mean(proc_nonzero) if len(proc_nonzero) > 0 else 0.0,
        'proc_time_max': np.max(proc_nonzero) if len(proc_nonzero) > 0 else 0.0,
        'tf_correction_mean': np.mean(np.abs(tf_corr)),
        'tf_correction_max': np.max(np.abs(tf_corr)),
        'tf_correction_std': np.std(tf_corr),
        'tf_yaw_correction_std': np.std(tf_yaw),
    }


# ---- Print tables ----

def print_mechanical_stability_table(names, stabilities):
    """Print the mechanical stability table matching the journal paper format.

    Variables:
      a_{x,max}     — Peak longitudinal linear acceleration (m/s^2)
      a_{z,max}     — Peak vertical linear acceleration deviation from g (m/s^2)
      theta_{x,max} — Peak roll angle (rad)
      theta_{z,max} — Peak yaw angle (rad)
      alpha_{x,max} — Peak roll angular acceleration (rad/s^2)
      alpha_{z,max} — Peak yaw angular acceleration (rad/s^2)
    """
    header = '{:<28s}'.format('Variable') + ''.join('{:>14s}'.format(n) for n in names)
    sep = '-' * len(header)

    print('\n' + sep)
    print('MECHANICAL STABILITY DATA')
    print(sep)
    print(header)
    print(sep)

    rows = [
        ('a_x,max  (m/s2)',    [s['accel_x_max'] for s in stabilities]),
        ('a_z,max  (m/s2)',    [s['accel_z_max'] for s in stabilities]),
        ('theta_x,max  (rad)', [s['roll_max_abs'] for s in stabilities]),
        ('theta_z,max  (rad)', [s['yaw_max_abs'] for s in stabilities]),
        ('alpha_x,max  (rad/s2)', [s['angular_accel_x_max'] for s in stabilities]),
        ('alpha_z,max  (rad/s2)', [s['angular_accel_z_max'] for s in stabilities]),
    ]

    for label, vals in rows:
        fmt_vals = ''.join('{:>14.4f}'.format(v) for v in vals)
        print('{:<28s}'.format(label) + fmt_vals)

    print(sep + '\n')


def print_comparison_table(names, stabilities, slips, trajectories):
    header = '{:<25s}'.format('Metric') + ''.join('{:>14s}'.format(n) for n in names)
    sep = '-' * len(header)

    print('\n' + sep)
    print('LOCOMOTION METRICS')
    print(sep)
    print(header)
    print(sep)

    rows = [
        ('Pitch mean (rad)',     [s['pitch_mean'] for s in stabilities]),
        ('Pitch std (rad)',      [s['pitch_std'] for s in stabilities]),
        ('Pitch max abs (rad)',  [s['pitch_max_abs'] for s in stabilities]),
        ('Roll std (rad)',       [s['roll_std'] for s in stabilities]),
        ('Yaw max abs (rad)',    [s['yaw_max_abs'] for s in stabilities]),
        ('Vibration RMS (m/s2)', [s['vibration_rms'] for s in stabilities]),
        ('Accel X RMS (m/s2)',   [s['accel_x_rms'] for s in stabilities]),
        ('Accel Z RMS (m/s2)',   [s['accel_z_rms'] for s in stabilities]),
        ('Ang accel X max (r/s2)', [s['angular_accel_x_max'] for s in stabilities]),
        ('Ang accel Z max (r/s2)', [s['angular_accel_z_max'] for s in stabilities]),
        ('Mean slip (%)',        slips),
        ('Distance (m)',         [t['total_distance'] for t in trajectories]),
        ('Final drift (m)',      [t['final_drift'] for t in trajectories]),
    ]

    for label, vals in rows:
        fmt_vals = ''.join('{:>14.4f}'.format(v) for v in vals)
        print('{:<25s}'.format(label) + fmt_vals)

    print(sep + '\n')


def print_slam_table(names, slam_metrics, rpe_metrics=None, ate_metrics=None):
    header = '{:<30s}'.format('SLAM Metric') + ''.join('{:>14s}'.format(n) for n in names)
    sep = '-' * len(header)

    print(sep)
    print('SLAM QUALITY METRICS')
    print(sep)
    print(header)
    print(sep)

    rows = [
        ('Loop closures',          [s['loop_closures'] for s in slam_metrics], 'd'),
        ('Proximity detections',   [s['proximity_detections'] for s in slam_metrics], 'd'),
        ('ICP inliers (mean)',     [s['inliers_mean'] for s in slam_metrics], '.1f'),
        ('ICP inliers (std)',      [s['inliers_std'] for s in slam_metrics], '.1f'),
        ('ICP matches (mean)',     [s['matches_mean'] for s in slam_metrics], '.1f'),
        ('ICP inliers ratio (mean)', [s['icp_inliers_ratio_mean'] for s in slam_metrics], '.4f'),
        ('ICP inliers ratio (std)',  [s['icp_inliers_ratio_std'] for s in slam_metrics], '.4f'),
        ('ICP correspondences (mean)', [s['icp_correspondences_mean'] for s in slam_metrics], '.1f'),
        ('Processing time (mean)', [s['proc_time_mean'] for s in slam_metrics], '.4f'),
        ('Processing time (max)',  [s['proc_time_max'] for s in slam_metrics], '.4f'),
        ('TF correction (mean m)', [s['tf_correction_mean'] for s in slam_metrics], '.4f'),
        ('TF correction (max m)',  [s['tf_correction_max'] for s in slam_metrics], '.4f'),
        ('TF correction (std m)',  [s['tf_correction_std'] for s in slam_metrics], '.4f'),
        ('TF yaw corr. (std rad)',[s['tf_yaw_correction_std'] for s in slam_metrics], '.4f'),
    ]

    # RPE metrics
    if rpe_metrics and all(r is not None for r in rpe_metrics):
        rows += [
            ('RPE mean (m)',   [r['rpe_mean'] for r in rpe_metrics], '.4f'),
            ('RPE max (m)',    [r['rpe_max'] for r in rpe_metrics], '.4f'),
            ('RPE RMSE (m)',   [r['rpe_rmse'] for r in rpe_metrics], '.4f'),
        ]

    # ATE metrics
    if ate_metrics and all(a is not None for a in ate_metrics):
        rows += [
            ('ATE mean (m)',   [a['ate_mean'] for a in ate_metrics], '.4f'),
            ('ATE max (m)',    [a['ate_max'] for a in ate_metrics], '.4f'),
            ('ATE RMSE (m)',   [a['ate_rmse'] for a in ate_metrics], '.4f'),
        ]

    for label, vals, fmt in rows:
        fmt_vals = ''.join(('{:>14' + fmt + '}').format(v) for v in vals)
        print('{:<30s}'.format(label) + fmt_vals)

    print(sep + '\n')


# ---- Locomotion plots ----

def plot_pitch(datasets, names, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, FIG_HEIGHT_DOUBLE), sharex=True)
    for i, (df, name) in enumerate(zip(datasets, names)):
        axes[0].plot(df['time'].values, np.degrees(df['pitch'].values),
                     ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
        if 'roll' in df.columns:
            axes[1].plot(df['time'].values, np.degrees(df['roll'].values),
                         ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
    axes[0].set_ylabel('Pitch (deg)')
    axes[0].legend()
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Roll (deg)')
    axes[1].legend()
    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(output_dir, 'pitch_roll_comparison.eps'))
    plt.close(fig)


def plot_acceleration(datasets, names, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, FIG_HEIGHT_DOUBLE), sharex=True)
    for i, (df, name) in enumerate(zip(datasets, names)):
        axes[0].plot(df['time'].values, df['accel_x'].values,
                     ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
        axes[1].plot(df['time'].values, (df['accel_z'] - GRAVITY).values,
                     ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
    axes[0].set_ylabel(r'$a_x$ (m/s$^2$)')
    axes[0].legend()
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel(r'$a_z - g$ (m/s$^2$)')
    axes[1].legend()
    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(output_dir, 'acceleration_comparison.eps'))
    plt.close(fig)


def plot_vibration(datasets, names, output_dir):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, FIG_HEIGHT_SINGLE))
    for i, (df, name) in enumerate(zip(datasets, names)):
        ax_val = df['accel_x'].values
        ay_val = df['accel_y'].values if 'accel_y' in df.columns else np.zeros(len(df))
        az_val = df['accel_z'].values - GRAVITY
        vib = np.sqrt(ax_val**2 + ay_val**2 + az_val**2)
        window = min(50, len(vib) // 10)
        if window > 1:
            vib_smooth = np.convolve(vib, np.ones(window)/window, mode='same')
        else:
            vib_smooth = vib
        ax.plot(df['time'].values, vib_smooth,
                ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Vibration magnitude (m/s$^2$)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'vibration_comparison.eps'))
    plt.close(fig)


def plot_slip(datasets, names, output_dir):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, FIG_HEIGHT_SINGLE))
    for i, (df, name) in enumerate(zip(datasets, names)):
        mask = np.abs(df['cmd_vx']) > SLIP_CMD_THRESHOLD
        if mask.sum() == 0:
            continue
        cmd = df.loc[mask, 'cmd_vx'].values
        actual = df.loc[mask, 'odom_vx'].values
        slip_pct = (1.0 - actual / cmd) * 100.0
        time_masked = df.loc[mask, 'time'].values
        ax.plot(time_masked, slip_pct,
                ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Slip (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'slip_comparison.eps'))
    plt.close(fig)


def plot_trajectory(datasets, names, output_dir):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, FIG_HEIGHT_SQUARE))
    for i, (df, name) in enumerate(zip(datasets, names)):
        # Align odom trajectory using GT initial yaw so all robots
        # are plotted in the same reference frame
        ox = df['odom_x'].values - df['odom_x'].iloc[0]
        oy = df['odom_y'].values - df['odom_y'].iloc[0]
        if has_ground_truth(df) and 'gt_yaw' in df.columns:
            yaw0 = df['gt_yaw'].iloc[0]
            cos_y = np.cos(yaw0)
            sin_y = np.sin(yaw0)
            rx = cos_y * ox - sin_y * oy
            ry = sin_y * ox + cos_y * oy
        else:
            rx, ry = ox, oy
        ax.plot(rx, ry,
                ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
        ax.plot(rx[0], ry[0], 'o', color=COLORS[i % 3], markersize=5)
        ax.plot(rx[-1], ry[-1], 's', color=COLORS[i % 3], markersize=5)
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.legend()
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'trajectory_comparison.eps'))
    plt.close(fig)


# ---- SLAM plots ----

def plot_slam_icp(datasets, names, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH_IN, 6.5), sharex=True)
    for i, (df, name) in enumerate(zip(datasets, names)):
        icp_col = 'slam_icp_inliers_ratio' if 'slam_icp_inliers_ratio' in df.columns else 'slam_icp_rms'
        if icp_col not in df.columns:
            continue
        axes[0].plot(df['time'].values, df[icp_col].values,
                     ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
        axes[1].plot(df['time'].values, df['slam_inliers'].values,
                     ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
        if 'slam_icp_correspondences' in df.columns:
            axes[2].plot(df['time'].values, df['slam_icp_correspondences'].values,
                         ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
    axes[0].set_ylabel('ICP inliers ratio')
    axes[0].legend()
    axes[1].set_ylabel('ICP inliers')
    axes[1].legend()
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('ICP correspondences')
    axes[2].legend()
    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(output_dir, 'slam_icp_comparison.eps'))
    plt.close(fig)


def plot_slam_tf_corrections(datasets, names, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, FIG_HEIGHT_DOUBLE), sharex=True)
    for i, (df, name) in enumerate(zip(datasets, names)):
        if 'tf_correction_magnitude' not in df.columns:
            continue
        axes[0].plot(df['time'].values, df['tf_correction_magnitude'].values,
                     ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
        if 'tf_correction_yaw' in df.columns:
            axes[1].plot(df['time'].values, np.degrees(df['tf_correction_yaw'].values),
                         ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
    axes[0].set_ylabel('TF correction (m)')
    axes[0].legend()
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Yaw correction (deg)')
    axes[1].legend()
    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(output_dir, 'slam_tf_corrections.eps'))
    plt.close(fig)


def plot_slam_loop_closures(datasets, names, output_dir):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, FIG_HEIGHT_SINGLE))
    for i, (df, name) in enumerate(zip(datasets, names)):
        if 'loop_closure_count' not in df.columns:
            continue
        ax.plot(df['time'].values, df['loop_closure_count'].values,
                ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative loop closures')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'slam_loop_closures.eps'))
    plt.close(fig)


# ---- New plots for paper tables ----

def plot_yaw(datasets, names, output_dir):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, FIG_HEIGHT_SINGLE))
    for i, (df, name) in enumerate(zip(datasets, names)):
        if 'yaw' not in df.columns:
            continue
        ax.plot(df['time'].values, np.degrees(df['yaw'].values),
                ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\psi$ (deg)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'yaw_comparison.eps'))
    plt.close(fig)


def plot_angular_acceleration(datasets, names, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, FIG_HEIGHT_DOUBLE), sharex=True)
    for i, (df, name) in enumerate(zip(datasets, names)):
        if 'angular_accel_x' not in df.columns:
            continue
        axes[0].plot(df['time'].values, df['angular_accel_x'].values,
                     ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
        axes[1].plot(df['time'].values, df['angular_accel_z'].values,
                     ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
    axes[0].set_ylabel(r'$\dot{\omega}_x$ (rad/s$^2$)')
    axes[0].legend()
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel(r'$\dot{\omega}_z$ (rad/s$^2$)')
    axes[1].legend()
    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(output_dir, 'angular_acceleration_comparison.eps'))
    plt.close(fig)


def plot_ate_over_time(datasets, names, ate_results, output_dir):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, FIG_HEIGHT_SINGLE))
    for i, (df, name, ate) in enumerate(zip(datasets, names, ate_results)):
        if ate is None:
            continue
        ax.plot(df['time'].values, ate['ate_errors'],
                ls=LINE_STYLES[i % 3], color=COLORS[i % 3], label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ATE (m)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ate_over_time.eps'))
    plt.close(fig)


def plot_gt_vs_estimated(datasets, names, output_dir):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, FIG_HEIGHT_SQUARE))
    for i, (df, name) in enumerate(zip(datasets, names)):
        if not has_ground_truth(df) or len(df) < 3:
            continue
        c = COLORS[i % len(COLORS)]
        gt_x, gt_y, _, odom_x, odom_y, _ = align_gt_to_odom(df)
        ax.plot(gt_x, gt_y, '-', color=c, linewidth=1.4,
                label='{} (GT)'.format(name))
        ax.plot(odom_x, odom_y, '--', color=c, linewidth=0.9,
                label='{} (odom)'.format(name))
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'gt_vs_estimated_trajectory.eps'))
    plt.close(fig)


# ---- Correlation plot: vibration vs SLAM degradation ----

def plot_vibration_vs_ate(names, stabilities, ate_metrics, output_dir):
    """Scatter plot of vibration RMS vs ATE RMSE per robot.

    This is the key figure that visualises the paper's central hypothesis:
    higher chassis vibration degrades SLAM accuracy.
    """
    vib = [s['vibration_rms'] for s in stabilities]
    ate = [a['ate_rmse'] for a in ate_metrics]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, FIG_HEIGHT_SINGLE + 0.4))

    # Plot each robot as a labelled marker
    for i, name in enumerate(names):
        ax.scatter(vib[i], ate[i],
                   marker=MARKERS[i % 3], s=80, color=COLORS[i % 3],
                   edgecolors='black', linewidths=0.5, zorder=3,
                   label=name)

    # Linear trend line
    vib_arr = np.array(vib)
    ate_arr = np.array(ate)
    if len(vib_arr) >= 2:
        coeffs = np.polyfit(vib_arr, ate_arr, 1)
        x_fit = np.linspace(vib_arr.min() * 0.9, vib_arr.max() * 1.1, 50)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, 'k--', linewidth=0.8,
                label='Linear fit')

        # Pearson correlation coefficient
        if len(vib_arr) >= 3:
            r = np.corrcoef(vib_arr, ate_arr)[0, 1]
            ax.text(0.97, 0.05, '$r = {:.3f}$'.format(r),
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='0.7'))

    ax.set_xlabel(r'Vibration RMS (m/s$^2$)')
    ax.set_ylabel('ATE RMSE (m)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'vibration_vs_ate.eps'))
    plt.close(fig)


# ---- Bar summary ----

def plot_bar_summary(names, stabilities, slips, trajectories, slam_metrics, output_dir):
    metrics = {
        r'$\sigma_\theta$ (rad)':       [s['pitch_std'] for s in stabilities],
        r'Vibration RMS (m/s$^2$)':     [s['vibration_rms'] for s in stabilities],
        r'$a_z$ RMS (m/s$^2$)':         [s['accel_z_rms'] for s in stabilities],
        'Mean slip (%)':                 slips,
    }

    if slam_metrics and slam_metrics[0] is not None:
        metrics['ICP inliers ratio'] = [s['icp_inliers_ratio_mean'] for s in slam_metrics]
        metrics[r'TF corr. $\sigma$ (m)'] = [s['tf_correction_std'] for s in slam_metrics]

    n_plots = len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(COLUMN_WIDTH_IN, 2.8))
    if n_plots == 1:
        axes = [axes]
    x = np.arange(len(names))

    for ax, (title, values) in zip(axes, metrics.items()):
        bars = ax.bar(x, values, color=COLORS[:len(names)], width=0.55, edgecolor='black', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha='right', fontsize=7)
        ax.set_title(title, fontsize=8, pad=4)
        ax.grid(True, axis='y')
        ax.tick_params(axis='y', labelsize=7)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    '{:.3f}'.format(val), ha='center', va='bottom', fontsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'metrics_bar_comparison.eps'))
    plt.close(fig)


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description='Analyze and compare robot locomotion + SLAM metrics from CSV files.')
    parser.add_argument('--files', nargs='+', required=True,
                        help='CSV files to compare (one per robot)')
    parser.add_argument('--names', nargs='+', default=None,
                        help='Robot names (defaults to filenames without extension)')
    parser.add_argument('--output_dir', default='.',
                        help='Directory to save plots (default: current directory)')
    args = parser.parse_args()

    # Canonical display names keyed by filename patterns
    CANONICAL_ORDER = [
        (['differential', 'husky'], 'Husky (differential)'),
        (['tracked'],               'Tracked'),
        (['rocker', 'rocker_bogie'],'Rocker-bogie'),
    ]

    if args.names:
        names = args.names
    else:
        names = []
        for f in args.files:
            base = os.path.splitext(os.path.basename(f))[0].lower()
            matched = False
            for patterns, display_name in CANONICAL_ORDER:
                if any(p in base for p in patterns):
                    names.append(display_name)
                    matched = True
                    break
            if not matched:
                names.append(os.path.splitext(os.path.basename(f))[0])

    # Sort files into canonical order: Husky, Tracked, Rocker-bogie
    if len(args.files) == 3 and not args.names:
        canonical_names = [cn for _, cn in CANONICAL_ORDER]
        order = []
        for cn in canonical_names:
            for i, n in enumerate(names):
                if n == cn and i not in order:
                    order.append(i)
                    break
        if len(order) == len(args.files):
            args.files = [args.files[i] for i in order]
            names = [names[i] for i in order]

    if len(args.files) != 3:
        print("Warning: expected 3 CSV files, got {}. Proceeding anyway.".format(
            len(args.files)))

    # Load data, filter out datasets with insufficient data
    datasets_raw = [load_csv(f) for f in args.files]
    valid = [(df, n) for df, n in zip(datasets_raw, names) if len(df) >= 3]
    if len(valid) < len(names):
        skipped = set(names) - {n for _, n in valid}
        print("Warning: skipping datasets with insufficient data: {}".format(
            ', '.join(skipped)))
    datasets = [v[0] for v in valid]
    names = [v[1] for v in valid]

    if len(datasets) == 0:
        print("Error: no datasets with sufficient data. Exiting.")
        return

    # Compute locomotion metrics
    stabilities = [compute_stability(df) for df in datasets]
    slip_data = [compute_slip(df) for df in datasets]
    slips_mean = [s[1] for s in slip_data]
    trajectories = [compute_trajectory(df) for df in datasets]

    # Compute SLAM metrics
    slam_available = all(has_slam_columns(df) for df in datasets)
    slam_metrics = [compute_slam_metrics(df) for df in datasets] if slam_available else None

    # Compute RPE and ATE
    gt_available = all(has_ground_truth(df) for df in datasets)
    rpe_metrics = [compute_rpe(df) for df in datasets] if gt_available else None
    ate_metrics = [compute_ate(df) for df in datasets] if gt_available else None

    # Print tables
    print_mechanical_stability_table(names, stabilities)
    print_comparison_table(names, stabilities, slips_mean, trajectories)
    if slam_metrics and all(s is not None for s in slam_metrics):
        print_slam_table(names, slam_metrics, rpe_metrics, ate_metrics)

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate locomotion plots
    plot_pitch(datasets, names, args.output_dir)
    plot_acceleration(datasets, names, args.output_dir)
    plot_vibration(datasets, names, args.output_dir)
    plot_slip(datasets, names, args.output_dir)
    plot_trajectory(datasets, names, args.output_dir)
    plot_yaw(datasets, names, args.output_dir)
    plot_angular_acceleration(datasets, names, args.output_dir)

    plots = [
        'pitch_roll_comparison.eps',
        'acceleration_comparison.eps',
        'vibration_comparison.eps',
        'slip_comparison.eps',
        'trajectory_comparison.eps',
        'yaw_comparison.eps',
        'angular_acceleration_comparison.eps',
    ]

    # Generate SLAM plots if data available
    if slam_available:
        plot_slam_icp(datasets, names, args.output_dir)
        plot_slam_tf_corrections(datasets, names, args.output_dir)
        plot_slam_loop_closures(datasets, names, args.output_dir)
        plots += [
            'slam_icp_comparison.eps',
            'slam_tf_corrections.eps',
            'slam_loop_closures.eps',
        ]

    # Generate ground truth plots if available
    if gt_available and ate_metrics:
        plot_ate_over_time(datasets, names, ate_metrics, args.output_dir)
        plot_gt_vs_estimated(datasets, names, args.output_dir)
        plot_vibration_vs_ate(names, stabilities, ate_metrics, args.output_dir)
        plots += [
            'ate_over_time.eps',
            'gt_vs_estimated_trajectory.eps',
            'vibration_vs_ate.eps',
        ]

    plot_bar_summary(names, stabilities, slips_mean, trajectories,
                     slam_metrics if slam_metrics else [None]*len(names), args.output_dir)
    plots.append('metrics_bar_comparison.eps')

    print("Plots saved to: {}".format(os.path.abspath(args.output_dir)))
    for p in plots:
        print("  - {}".format(p))


if __name__ == '__main__':
    main()
