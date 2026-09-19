#!/usr/bin/env python3
"""Compare an RTAB-Map point cloud against the true geometry of the Gazebo world.

Trajectory error says how well the robot knew where it was.  It says nothing
about whether the map it produced is the right shape - and for a mine
exploration paper, the map is the deliverable.  This script measures that
directly, by sampling the collision geometry of the world the robot drove
through and comparing the reconstructed cloud against it.

METRICS
=======
With A = the map cloud and B = points sampled on the true surfaces:

  accuracy      for each map point, the distance to the nearest true surface.
                Large values mean the map contains geometry that is not there
                (drift smearing walls, or ghost surfaces after a bad loop
                closure).
  completeness  for each true surface point, the distance to the nearest map
                point.  Large values mean parts of the mine were never mapped.
  Chamfer       mean(accuracy) + mean(completeness).  The single number, but
                the two halves are reported separately because they fail for
                completely different reasons and a paper that quotes only the
                sum cannot distinguish "mapped the wrong shape" from "did not
                finish exploring".
  precision@t   fraction of map points within t of a true surface.
  recall@t      fraction of true surface within t of a map point.
  F-score@t     harmonic mean of the two - the standard summary in the
                reconstruction literature, and the one to quote.
  density       map points per square metre of true surface actually covered.

Only surface within `--range` of the robot's ground-truth trajectory is used as
the reference.  A robot cannot be blamed for failing to map a gallery it never
entered, and including unvisited geometry would make completeness a measure of
route coverage instead of mapping quality.

INPUT
=====
    --map        the cloud exported from RTAB-Map (.pcd or .ply).  Export with
                   rosrun rtabmap_ros rtabmap-export --cloud map.pcd ~/.ros/rtabmap.db
                 or by subscribing to /rtabmap/cloud_map and saving it.
    --run        the run directory, for the ground-truth trajectory
                 (gt_traj.tum) used to align the map and limit the reference.
    --world      the .world file describing the true geometry.

ALIGNMENT
=========
The map is expressed in the SLAM map frame, which is created at the robot's
spawn pose; the world geometry is in Gazebo world coordinates.  The two are
related by the spawn pose from sim_config.yaml, which is applied before
comparing.  Pass --icp-refine to additionally run a few rigid alignment
iterations; report whether you used it, since it flatters the accuracy figure
by absorbing a constant offset that a real deployment would suffer.
"""

import argparse
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pointcloud_io as pio                                # noqa: E402
from metrics_io import read_tum                            # noqa: E402
import model_parser as mp                                  # noqa: E402
import slam_metrics as sm                                  # noqa: E402

import xml.etree.ElementTree as ET


def find_model_dir(world_path, uri):
    """Resolve model://name against the directories beside the world file."""
    name = uri.replace('model://', '').split('/')[0]
    base = os.path.dirname(os.path.abspath(world_path))
    for cand in (os.path.join(base, name),
                 os.path.join(base, '..', name),
                 os.path.join(base, '..', '..', name)):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return None


def sample_world(world_path, density, verbose=True):
    """Surface samples of every collision body in the world, in world frame."""
    root = ET.parse(world_path).getroot()
    world = root.find('world')
    if world is None:
        world = root

    samples = []
    labels = []

    def add(points, label):
        if len(points):
            samples.append(points)
            labels.append((label, len(points)))

    # --- included models ---
    for inc in world.findall('include'):
        uri = inc.find('uri')
        if uri is None or not uri.text:
            continue
        pose = mp.parse_pose(inc.find('pose').text
                             if inc.find('pose') is not None else None)
        mdir = find_model_dir(world_path, uri.text.strip())
        if mdir is None:
            if verbose:
                print('  ! could not resolve {}'.format(uri.text.strip()))
            continue
        sdf = os.path.join(mdir, 'model.sdf')
        if not os.path.isfile(sdf):
            continue
        try:
            model = mp.load_sdf(sdf)
        except Exception as exc:                          # noqa: BLE001
            if verbose:
                print('  ! {}: {}'.format(os.path.basename(mdir), exc))
            continue

        name = inc.find('name')
        label = name.text.strip() if name is not None and name.text else model.name

        for link in model.links.values():
            for T_c, geom in link.collisions:
                T = pose @ link.pose @ T_c
                pts = _sample_geometry(geom, mdir, density)
                if pts is None or not len(pts):
                    continue
                add((T[:3, :3] @ pts.T).T + T[:3, 3], label)

    # --- models written inline in the world (ramp, step) ---
    for m in world.findall('model'):
        pose = mp.parse_pose(m.find('pose').text
                             if m.find('pose') is not None else None)
        label = m.get('name', 'inline')
        for link in m.findall('link'):
            lpose = mp.parse_pose(link.find('pose').text
                                  if link.find('pose') is not None else None)
            for coll in link.findall('collision'):
                cpose = mp.parse_pose(coll.find('pose').text
                                      if coll.find('pose') is not None else None)
                geom = mp._parse_geometry(coll.find('geometry'))
                if geom is None:
                    continue
                T = pose @ lpose @ cpose
                pts = _sample_geometry(geom, os.path.dirname(world_path), density)
                if pts is None or not len(pts):
                    continue
                add((T[:3, :3] @ pts.T).T + T[:3, 3], label)

    if verbose:
        print('Reference geometry sampled:')
        for label, n in labels:
            print('  {:<22s} {:>8d} points'.format(label, n))

    if not samples:
        return np.zeros((0, 3))
    return np.vstack(samples)


def _sample_geometry(geom, model_dir, density):
    kind = geom.get('type')
    if kind == 'box':
        return pio.sample_box(geom['size'], density)
    if kind == 'cylinder':
        r, h = geom['radius'], geom['length']
        n = max(8, int(2 * math.pi * r * h * density))
        rng = np.random.default_rng(2)
        a = rng.random(n) * 2 * math.pi
        z = (rng.random(n) - 0.5) * h
        return np.column_stack([r * np.cos(a), r * np.sin(a), z])
    if kind == 'mesh':
        uri = geom.get('uri', '')
        fn = uri.replace('model://', '')
        parts = fn.split('/')
        cand = os.path.join(model_dir, *parts[1:]) if len(parts) > 1 else None
        for p in filter(None, [cand, os.path.join(model_dir, os.path.basename(fn))]):
            if os.path.isfile(p):
                return pio.sample_mesh_file(p, density)
        return None
    return None


def limit_to_visited(reference, traj_xyz, max_range):
    """Keep only reference points the robot could plausibly have seen."""
    if not len(reference) or not len(traj_xyz):
        return reference, np.ones(len(reference), dtype=bool)
    idx = pio.VoxelIndex(traj_xyz, cell=max(1.0, max_range / 4.0))
    d, found = idx.query(reference, max_radius=max_range)
    keep = found & np.isfinite(d) & (d <= max_range)
    return reference[keep], keep


def icp_refine(source, target_index, iterations=10, max_pair=1.0):
    """A few point-to-point ICP iterations, reported separately."""
    T = np.eye(4)
    cur = source.copy()
    for _ in range(iterations):
        d, found = target_index.query(cur, max_radius=max_pair)
        m = found & np.isfinite(d)
        if m.sum() < 50:
            break
        # Recover the matched target points by querying again per matched
        # point is expensive; instead align on the subset using Umeyama with
        # the nearest neighbours found by a direct search on the subset.
        src = cur[m]
        tgt = _nearest_points(target_index, src, max_pair)
        R, t, _ = sm.umeyama(src, tgt)
        step = np.eye(4)
        step[:3, :3] = R
        step[:3, 3] = t
        cur = (R @ cur.T).T + t
        T = step @ T
    return cur, T


def _nearest_points(index, queries, max_radius):
    """The actual nearest reference points (not just their distances)."""
    out = np.zeros_like(queries)
    qk = index._keys(queries)
    groups = {}
    for i in range(len(queries)):
        groups.setdefault(tuple(qk[i]), []).append(i)
    ring = max(1, int(np.ceil(max_radius / index.cell)))
    for key, idx in groups.items():
        cand = index._candidates(key, ring)
        if cand is None or not len(cand):
            out[idx] = queries[idx]
            continue
        idx = np.asarray(idx)
        d = np.linalg.norm(queries[idx][:, None, :] - cand[None, :, :], axis=2)
        out[idx] = cand[d.argmin(axis=1)]
    return out


def stats(d, found, label):
    ok = found & np.isfinite(d)
    if ok.sum() == 0:
        return {label + '_mean_m': float('nan'),
                label + '_median_m': float('nan'),
                label + '_rms_m': float('nan'),
                label + '_matched_pct': 0.0, label + '_n': int(len(d))}
    v = d[ok]
    return {
        label + '_mean_m': float(v.mean()),
        label + '_median_m': float(np.median(v)),
        label + '_rms_m': float(np.sqrt(np.mean(v ** 2))),
        label + '_p95_m': float(np.percentile(v, 95)),
        label + '_matched_pct': float(100.0 * ok.sum() / len(d)),
        label + '_n': int(len(d)),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--map', required=True, help='RTAB-Map cloud (.pcd/.ply)')
    ap.add_argument('--run', required=True, help='run directory (for gt_traj.tum)')
    ap.add_argument('--world', default=None, help='path to lcmine.world')
    ap.add_argument('--robot', default=None)
    ap.add_argument('--config', default=os.path.join(here, '..', 'config',
                                                     'sim_config.yaml'))
    ap.add_argument('--density', type=float, default=40.0,
                    help='reference samples per square metre')
    ap.add_argument('--range', type=float, default=15.0,
                    help='sensor range used to limit the reference geometry (m)')
    ap.add_argument('--tau', type=float, default=0.20,
                    help='threshold for precision/recall/F-score (m)')
    ap.add_argument('--max_radius', type=float, default=5.0,
                    help='distance beyond which a point counts as unmatched (m)')
    ap.add_argument('--voxel', type=float, default=0.10,
                    help='downsample the map to this resolution before '
                         'comparing (m); 0 disables')
    ap.add_argument('--icp-refine', action='store_true')
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run)
    robot = args.robot or os.path.basename(os.path.dirname(run_dir))

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    world = args.world
    if world is None:
        guess = {
            'rocker_bogie': 'rocker_bogie/world/mine/lcmine.world',
            'differential': 'differential/world/mine/lcmine.world',
            'tracked': ('tracked/gazebo_continuous_track_example/'
                        'world/mine/lcmine.world'),
        }.get(robot)
        src = os.path.abspath(os.path.join(here, '..', '..'))
        world = os.path.join(src, guess) if guess else None
    if not world or not os.path.isfile(world):
        print('World file not found; pass --world explicitly.')
        return 1

    print('map    : {}'.format(args.map))
    print('run    : {}'.format(run_dir))
    print('world  : {}'.format(world))
    print('robot  : {}'.format(robot))
    print('')

    cloud = pio.read_cloud(args.map)
    print('Map cloud: {} points'.format(len(cloud)))
    if args.voxel > 0:
        keys = np.floor(cloud / args.voxel).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        cloud = cloud[np.sort(keep)]
        print('  downsampled to {} points at {:.2f} m'.format(len(cloud),
                                                              args.voxel))

    # --- map frame -> world frame, using the spawn pose ---
    spawn = cfg['spawn']
    sx, sy = spawn['common']['x'], spawn['common']['y']
    syaw = spawn[robot]['yaw_rad'] if robot in spawn else 0.0
    c, s = math.cos(syaw), math.sin(syaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    cloud_world = (R @ cloud.T).T + np.array([sx, sy, 0.0])
    print('  transformed to world frame (spawn {:.1f}, {:.1f}, yaw {:.1f} deg)'
          .format(sx, sy, math.degrees(syaw)))

    # --- ground-truth trajectory ---
    tum_path = os.path.join(run_dir, 'gt_traj.tum')
    if os.path.isfile(tum_path):
        traj = read_tum(tum_path)
        traj_xyz = traj[:, 1:4] if len(traj) else np.zeros((0, 3))
        print('  ground-truth trajectory: {} poses'.format(len(traj_xyz)))
    else:
        traj_xyz = np.zeros((0, 3))
        print('  WARNING: no gt_traj.tum; the reference will not be limited '
              'to the visited part of the mine, so completeness will be '
              'pessimistic.')

    print('')
    reference = sample_world(world, args.density)
    print('  total {} reference points'.format(len(reference)))

    if len(traj_xyz):
        reference, _ = limit_to_visited(reference, traj_xyz, args.range)
        print('  {} within {:.0f} m of the trajectory'.format(len(reference),
                                                              args.range))
    if len(reference) == 0:
        print('No reference geometry; cannot score the map.')
        return 1

    ref_index = pio.VoxelIndex(reference, cell=max(0.3, args.tau * 2))

    if args.icp_refine:
        print('\nRefining alignment with ICP...')
        cloud_world, T = icp_refine(cloud_world, ref_index)
        print('  translation applied: {}'.format(np.round(T[:3, 3], 4)))

    print('\nScoring...')
    d_acc, f_acc = ref_index.query(cloud_world, max_radius=args.max_radius)
    map_index = pio.VoxelIndex(cloud_world, cell=max(0.3, args.tau * 2))
    d_cmp, f_cmp = map_index.query(reference, max_radius=args.max_radius)

    res = {}
    res.update(stats(d_acc, f_acc, 'accuracy'))
    res.update(stats(d_cmp, f_cmp, 'completeness'))

    acc_ok = f_acc & np.isfinite(d_acc)
    cmp_ok = f_cmp & np.isfinite(d_cmp)
    precision = float(np.mean(acc_ok & (d_acc <= args.tau)))
    recall = float(np.mean(cmp_ok & (d_cmp <= args.tau)))
    fscore = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

    res['chamfer_m'] = res['accuracy_mean_m'] + res['completeness_mean_m']
    res['precision_at_tau'] = precision
    res['recall_at_tau'] = recall
    res['fscore_at_tau'] = fscore
    res['tau_m'] = args.tau
    covered_area = recall * len(reference) / args.density
    res['covered_surface_m2'] = float(covered_area)
    res['map_density_pts_per_m2'] = (float(len(cloud_world) / covered_area)
                                     if covered_area > 0 else float('nan'))

    unit = {
        'accuracy_mean_m': 'm', 'accuracy_median_m': 'm', 'accuracy_rms_m': 'm',
        'accuracy_p95_m': 'm', 'accuracy_matched_pct': '%', 'accuracy_n': '-',
        'completeness_mean_m': 'm', 'completeness_median_m': 'm',
        'completeness_rms_m': 'm', 'completeness_p95_m': 'm',
        'completeness_matched_pct': '%', 'completeness_n': '-',
        'chamfer_m': 'm', 'precision_at_tau': '-', 'recall_at_tau': '-',
        'fscore_at_tau': '-', 'tau_m': 'm', 'covered_surface_m2': 'm^2',
        'map_density_pts_per_m2': '1/m^2',
    }

    lines = []
    lines.append('=' * 62)
    lines.append('MAP vs GROUND-TRUTH GEOMETRY - {}'.format(robot))
    lines.append('=' * 62)
    lines.append('{:<34s}{:>16s}  {}'.format('Metric', 'Value', 'Unit'))
    lines.append('-' * 62)
    order = ['accuracy_mean_m', 'accuracy_median_m', 'accuracy_rms_m',
             'accuracy_p95_m', 'accuracy_matched_pct',
             'completeness_mean_m', 'completeness_median_m',
             'completeness_rms_m', 'completeness_p95_m',
             'completeness_matched_pct',
             'chamfer_m', 'tau_m', 'precision_at_tau', 'recall_at_tau',
             'fscore_at_tau', 'covered_surface_m2', 'map_density_pts_per_m2']
    for k in order:
        if k in res:
            lines.append('{:<34s}{:>16.4f}  {}'.format(k, res[k],
                                                       unit.get(k, '')))
    lines.append('-' * 62)
    lines.append('map points          : {}'.format(len(cloud_world)))
    lines.append('reference points    : {}'.format(len(reference)))
    lines.append('ICP refinement      : {}'.format(
        'yes' if args.icp_refine else 'no'))
    lines.append('=' * 62)
    text = '\n'.join(lines)
    print('\n' + text)

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        base = os.path.join(args.output_dir, 'map_accuracy_{}'.format(robot))
        with open(base + '.txt', 'w') as f:
            f.write(text + '\n')
        import csv
        with open(base + '.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['robot', 'metric', 'unit', 'value'])
            for k in order:
                if k in res:
                    w.writerow([robot, k, unit.get(k, ''), res[k]])
        print('\nWritten to {}.txt / .csv'.format(base))
    return 0


if __name__ == '__main__':
    sys.exit(main())
