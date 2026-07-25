#!/usr/bin/env python3
"""Shared schema and I/O for the robot_metrics CSV files.

Every column carries its unit in the header, as "name[unit]", so a CSV is
self-describing and the units propagate automatically into the generated
tables (and from there into Table 3 of the paper) instead of living only in
someone's memory.

Dimensionless counts and flags use "[-]".

Both the logger and the analysis scripts import this module, so the schema
cannot drift between the code that writes the data and the code that reads it.

    from metrics_io import METRIC_COLUMNS, csv_header, load_metrics

    load_metrics(path) returns a DataFrame whose column names have the unit
    suffix stripped ("pitch" not "pitch[rad]"), with the units available as
    df.attrs['units'].  Legacy files written before the unit convention are
    read transparently.
"""

import csv
import os

# ---------------------------------------------------------------------------
# (name, unit, description)
# ---------------------------------------------------------------------------
METRIC_COLUMNS = [
    ('timestamp', 's', 'ROS sim time (use_sim_time:=true)'),

    # --- attitude / IMU ---
    ('pitch', 'rad', 'Chassis pitch from IMU orientation'),
    ('roll', 'rad', 'Chassis roll from IMU orientation'),
    ('yaw', 'rad', 'Chassis yaw from IMU orientation'),
    ('accel_x', 'm/s^2', 'IMU linear acceleration, body x'),
    ('accel_y', 'm/s^2', 'IMU linear acceleration, body y'),
    ('accel_z', 'm/s^2', 'IMU linear acceleration, body z (includes g)'),
    ('angular_vel_x', 'rad/s', 'IMU angular velocity, body x'),
    ('angular_vel_y', 'rad/s', 'IMU angular velocity, body y'),
    ('angular_vel_z', 'rad/s', 'IMU angular velocity, body z'),
    ('angular_accel_x', 'rad/s^2', 'd(angular_vel_x)/dt, numerical'),
    ('angular_accel_z', 'rad/s^2', 'd(angular_vel_z)/dt, numerical'),

    # --- SLAM odometry estimate ---
    ('odom_x', 'm', 'SLAM position estimate, x'),
    ('odom_y', 'm', 'SLAM position estimate, y'),
    ('odom_z', 'm', 'SLAM position estimate, z'),
    ('odom_roll', 'rad', 'SLAM orientation estimate, roll'),
    ('odom_pitch', 'rad', 'SLAM orientation estimate, pitch'),
    ('odom_yaw', 'rad', 'SLAM orientation estimate, yaw'),
    ('odom_vx', 'm/s', 'SLAM linear velocity estimate, body x'),
    ('odom_vyaw', 'rad/s', 'SLAM yaw rate estimate'),

    # --- command ---
    ('cmd_vx', 'm/s', 'Commanded linear velocity'),
    ('cmd_vyaw', 'rad/s', 'Commanded yaw rate'),

    # --- ground truth (exact Gazebo model pose, unfiltered) ---
    ('gt_x', 'm', 'Gazebo ground-truth position, x (world frame)'),
    ('gt_y', 'm', 'Gazebo ground-truth position, y (world frame)'),
    ('gt_z', 'm', 'Gazebo ground-truth position, z (world frame)'),
    ('gt_roll', 'rad', 'Gazebo ground-truth roll'),
    ('gt_pitch', 'rad', 'Gazebo ground-truth pitch'),
    ('gt_yaw', 'rad', 'Gazebo ground-truth yaw'),
    ('gt_vx', 'm/s', 'Gazebo ground-truth linear velocity, world x'),
    ('gt_vy', 'm/s', 'Gazebo ground-truth linear velocity, world y'),
    ('gt_distance', 'm', 'Cumulative ground-truth path length'),

    # --- SLAM internals ---
    ('loop_closure_count', '-', 'Cumulative loop closures accepted'),
    ('proximity_detection_count', '-', 'Cumulative proximity detections'),
    ('slam_inliers', '-', 'Visual/ICP inliers of the last odometry update'),
    ('slam_matches', '-', 'Feature matches of the last odometry update'),
    ('slam_icp_inliers_ratio', '-', 'ICP inlier ratio'),
    ('slam_icp_correspondences', '-', 'ICP correspondences'),
    ('slam_processing_time', 's', 'Odometry estimation time'),
    ('tracking_lost', '-', '1 while odometry tracking is lost, else 0'),
    ('tracking_loss_count', '-', 'Cumulative tracking-loss events'),
    ('relocalization_count', '-', 'Cumulative recoveries after a loss'),

    # --- map->odom correction ---
    ('tf_correction_magnitude', 'm', 'Per-sample |delta| of the map->odom TF'),
    ('tf_correction_yaw', 'rad', 'Per-sample yaw delta of the map->odom TF'),
]

# Per-wheel contact columns are appended dynamically: one contact flag and one
# normal force per wheel link.
CONTACT_COLUMN_TEMPLATES = [
    ('contact_{}', '-', 'Wheel {} in contact with the ground (1/0)'),
    ('normal_force_{}', 'N', 'Wheel {} contact normal force magnitude'),
]


def contact_columns(wheel_links):
    cols = []
    for link in wheel_links:
        for name, unit, desc in CONTACT_COLUMN_TEMPLATES:
            cols.append((name.format(link), unit, desc.format(link)))
    return cols


def csv_header(columns):
    """['timestamp[s]', 'pitch[rad]', ...]"""
    return ['{}[{}]'.format(name, unit) for name, unit, _ in columns]


def strip_unit(column_name):
    """'pitch[rad]' -> ('pitch', 'rad');  'pitch' -> ('pitch', '')"""
    if column_name.endswith(']') and '[' in column_name:
        base, unit = column_name.rsplit('[', 1)
        return base.strip(), unit[:-1]
    return column_name.strip(), ''


# Columns whose name changed when units were introduced.  Older CSVs are
# remapped on load so previously collected runs stay usable.
LEGACY_ALIASES = {
    'pitch_rad': 'pitch',
    'roll_rad': 'roll',
    'yaw_rad': 'yaw',
    'slam_processing_time_s': 'slam_processing_time',
    'slam_icp_rms': 'slam_icp_inliers_ratio',
}


def load_metrics(path):
    """Read a metrics CSV into a DataFrame with unit-free column names.

    Units end up in df.attrs['units'] as {column: unit}.
    """
    import pandas as pd

    df = pd.read_csv(path)
    units = {}
    renamed = {}
    for col in df.columns:
        base, unit = strip_unit(col)
        base = LEGACY_ALIASES.get(base, base)
        renamed[col] = base
        units[base] = unit
    df = df.rename(columns=renamed)
    df.attrs['units'] = units
    df.attrs['source'] = os.path.abspath(path)
    return df


def unit_of(df, column, default=''):
    return df.attrs.get('units', {}).get(column, default)


def write_units_reference(columns, path):
    """Emit a machine-readable data dictionary next to the CSVs."""
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['column', 'unit', 'description'])
        for name, unit, desc in columns:
            w.writerow([name, unit, desc])


# ---------------------------------------------------------------------------
# TUM trajectory format:  timestamp tx ty tz qx qy qz qw
# The de-facto interchange format for trajectory evaluation, so the logged
# trajectories can be fed straight to evo / TUM tools for cross-checking the
# ATE and RPE numbers produced here.
# ---------------------------------------------------------------------------
TUM_HEADER = ('# timestamp[s] tx[m] ty[m] tz[m] qx[-] qy[-] qz[-] qw[-]\n')


def write_tum(path, rows):
    with open(path, 'w') as f:
        f.write(TUM_HEADER)
        for r in rows:
            f.write('%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n' % tuple(r))


def read_tum(path):
    import numpy as np

    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            data.append([float(v) for v in line.split()])
    return np.array(data)
