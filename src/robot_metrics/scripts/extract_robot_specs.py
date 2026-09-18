#!/usr/bin/env python3
"""Mechanical specification table for the three platforms, straight from the
model files.

The submitted paper stated the platforms' masses and dimensions without saying
where they came from, which makes the fairness of the comparison impossible to
check.  This script derives them from the same URDF/SDF files Gazebo loads, so
the table in the paper and the robots in the simulation cannot disagree.

Extracted per robot:
    total mass            sum of every link's <mass>, plus any <inertial>
                          embedded in a <gazebo> block (the continuous-track
                          plugin builds the tracked robot's belts that way and
                          they are simulated bodies like any other)          [kg]
    centre of gravity     mass-weighted mean of the link inertial origins,
                          in the base_link frame                            [m]
    CoG height            the same, above the wheel contact plane           [m]
    footprint             axis-aligned bounding box of all COLLISION
                          geometry, in base_link                            [m]
    track / wheelbase     measured between wheel-link origins               [m]
    wheel radius          from the wheel collision cylinder                 [m]
    max wheel speed       joint velocity limit x wheel radius             [m/s]
    commanded max speed   max_linear_speed from the waypoint config        [m/s]
    ground clearance      lowest chassis collision point above the contact
                          plane                                             [m]

Two different "max speed" numbers are reported on purpose.  The kinematic limit
is what the model can do; the commanded limit is what the experiment actually
asked for.  If the paper quotes one it should say which.

Usage:
    ./extract_robot_specs.py
    ./extract_robot_specs.py --output_dir ~/paper_tables
    ./extract_robot_specs.py --wheel-map      # per-wheel positions only
"""

import argparse
import csv
import os
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_parser as mp                                  # noqa: E402


# Where each platform's description and configuration live, relative to src/.
ROBOTS = {
    'differential': {
        'display': 'Husky (differential)',
        # EL FICHERO QUE GAZEBO SPAWNEA (-database differential), no la copia
        # de model/model.sdf, que se quedo con el LiDAR a 0.9000 m.  Misma
        # lista que check_sim_parity.SPAWN_SOURCE.
        'description': 'differential/model/differential/model.sdf',
        'wheel_pattern': r'wheel',
        # Drop from the wheel axle to the ground.  Declared, not taken as the
        # wheel radius, because on the tracked platform it is not the wheel
        # radius - see that entry.  Kept in step with
        # sim_config.yaml: sensor_mounting.contact_plane_below_axle_m.
        'contact_offset_m': 0.148,
        'waypoints': 'differential/config/waypoints_mine.yaml',
        'launch': 'differential/launch/lcmine_husky_world.launch',
        # Was libgazebo_ros_skid_steer_drive.so until the actuation budget
        # went in: an ODE velocity motor has no gain, so its response to
        # load could not be matched to the other two platforms.
        'controller': 'JointVelocityController x4 + twist_to_wheels.py (skid-steer)',
        # Sign of the body-frame x axis that points forward. Needed to compare
        # sensor mounting positions across platforms: base_link origins do not
        # coincide and one model is built with forward = -x, so raw x values
        # are not comparable between robots.
        'forward_axis_x': +1,
    },
    'tracked': {
        'display': 'Tracked',
        # Se spawnea con -database two_track_robot, o sea el SDF del model
        # database: el xacro _gazebo describe otro robot (su CM sale 15.6 mm
        # mas adelante, los idlers lumpeados en left_body/right_body).
        'description': ('tracked/gazebo_continuous_track_example/model/'
                        'two_track_robot/model.sdf'),
        # Idlers are ground-contact bodies too: on a tracked vehicle the
        # sprocket-to-idler span is what the wheelbase means.
        'wheel_pattern': r'sprocket|idler|wheel|track',
        # NOT the sprocket radius.  This vehicle rests on its BELT, and the
        # belt elements do not exist in any file - gazebo_continuous_track
        # creates them at load time - so this cannot be derived from the model.
        # Chain of radii from the sprocket axle, all declared in model.sdf:
        #     sprocket / idler collision   0.128  = belt inner face
        #     trajectory = arc cylinder    0.138  = belt mid-line
        #     belt outer face              0.148  = the contact plane
        # (the elements are 0.02 thick and ride centred on the trajectory).
        # WAS 0.180 until 2026-09-17, when the belt was re-seated on the
        # sprocket; the arc cylinder had also been left at 0.2, 20 mm PROUD of
        # the belt, which made the drums the only thing touching the ground.
        'contact_offset_m': 0.148,
        'waypoints': ('tracked/gazebo_continuous_track_example/'
                      'config/waypoints_mine.yaml'),
        'launch': ('tracked/gazebo_continuous_track_example/launch/'
                   'lcmine_two_track_world.launch'),
        'controller': 'JointVelocityController x2 + twist_to_tracks.py',
        'forward_axis_x': +1,
    },
    'rocker_bogie': {
        'display': 'Rocker-bogie',
        'description': 'rocker_bogie/urdf/ensamblajeurdf.xacro',
        'wheel_pattern': r'^Rueda',
        'contact_offset_m': 0.148,
        'waypoints': 'rocker_bogie/config/waypoints_mine.yaml',
        'launch': 'rocker_bogie/launch/lcmine_rocker_bogie_world.launch',
        'controller': ('JointVelocityController x6 + JointPositionController x4'
                       ' + twist_to_wheels.py (Ackermann)'),
        # base_link mira a +x desde 2026-09-13: el chasis del CAD, que mira a -x,
        # cuelga de el girado 180 deg como chassis_link.
        'forward_axis_x': +1,
    },
}

# Controller parameters worth reporting, wherever they are declared.
CONTROLLER_KEYS = ['wheelSeparation', 'wheelDiameter', 'wheelTorque',
                   'wheelAcceleration', 'max_velocity', 'updateRate',
                   'wheelbase', 'front_track', 'rear_track', 'wheel_radius',
                   'max_steer_angle', 'track_separation', 'sprocket_radius',
                   # Command-path parity: min_angular must read 0.0 on all
                   # three, see actuation.command in sim_config.yaml.
                   'track_width', 'min_angular']

ORDER = ['differential', 'tracked', 'rocker_bogie']


def wheel_links(model, pattern):
    rx = re.compile(pattern)
    return {name: link for name, link in model.links.items() if rx.search(name)}


def wheel_geometry(model, pattern):
    """Wheel centres, radius, track and wheelbase.

    The wheel centre is the origin of its collision cylinder rather than the
    link origin: on CAD-exported models the link frame is often offset from the
    wheel by half its width, which would corrupt the track measurement.
    """
    wheels = wheel_links(model, pattern)
    centres = {}
    radii = []
    for name, link in wheels.items():
        for T_c, geom in link.collisions:
            if geom['type'] == 'cylinder':
                radii.append(geom['radius'])
                centres[name] = (link.pose @ T_c)[:3, 3]
                break
        else:
            if link.collisions:
                T_c, _ = link.collisions[0]
                centres[name] = (link.pose @ T_c)[:3, 3]

    if not centres:
        return {}

    P = np.array(list(centres.values()))
    xs, ys, zs = P[:, 0], P[:, 1], P[:, 2]
    return {
        'centres': centres,
        'radius': float(np.median(radii)) if radii else float('nan'),
        'track': float(ys.max() - ys.min()),
        'wheelbase': float(xs.max() - xs.min()),
        'axle_height': float(np.median(zs)),
        'count': len(centres),
    }


def max_joint_speed(model, pattern, radius):
    """Fastest a driven wheel may spin, times its radius."""
    rx = re.compile(pattern)
    vels = []
    for j in model.joints.values():
        if j.type not in ('continuous', 'revolute'):
            continue
        if not (rx.search(j.child or '') or rx.search(j.name or '')):
            continue
        v = j.limit.get('velocity')
        if v:
            vels.append(v)
    if not vels or not np.isfinite(radius):
        return float('nan'), float('nan')
    w = float(np.median(vels))
    return w, w * radius


def commanded_max_speed(src, rel_path):
    path = os.path.join(src, rel_path)
    if not os.path.isfile(path):
        return float('nan'), None
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    nav = cfg.get('navigation_params', {})
    return float(nav.get('max_linear_speed', float('nan'))), path


def ground_clearance(model, pattern, axle_height, contact_drop):
    """Lowest non-wheel collision point above the contact plane.

    `contact_drop` is the distance from the axle down to the ground, which is
    the wheel radius only on the platforms that run on wheels.
    """
    rx = re.compile(pattern)
    contact_z = (axle_height - contact_drop if np.isfinite(contact_drop)
                 else 0.0)
    lowest = None
    for name, link in model.links.items():
        if rx.search(name):
            continue
        for T_c, geom in link.collisions:
            corners = mp._geometry_corners(geom)
            if corners is None:
                continue
            T = link.pose @ T_c
            for c in corners:
                z = (T @ np.append(c, 1.0))[2]
                lowest = z if lowest is None else min(lowest, z)
    if lowest is None:
        return float('nan')
    return float(lowest - contact_z)


def controller_params(model, src, spec):
    """Controller configuration, from the SDF plugin and/or the launch file.

    The three platforms declare their drive limits in different places - the
    Husky inside a <plugin> in its SDF, the other two as <param> entries on the
    node that converts cmd_vel into joint commands - so both are scanned and
    merged.
    """
    params = {}

    for pl in model.plugins:
        if 'drive' in (pl.get('filename') or '') or 'drive' in (pl.get('name') or ''):
            for k, v in pl['params'].items():
                if k in CONTROLLER_KEYS:
                    params[k] = v

    launch_rel = spec.get('launch')
    if launch_rel:
        path = os.path.join(src, launch_rel)
        if os.path.isfile(path):
            text = open(path).read()
            for k in CONTROLLER_KEYS:
                m = re.search(
                    r'<param\s+name="{}"\s+value="([^"]+)"'.format(re.escape(k)),
                    text)
                if m:
                    params[k] = m.group(1)
    return params


def analyse(key, spec, src):
    path = os.path.join(src, spec['description'])
    if not os.path.isfile(path):
        return {'display': spec['display'], 'error':
                'description not found: {}'.format(spec['description'])}

    try:
        model = mp.load(path)
    except Exception as exc:                       # noqa: BLE001
        hint = ''
        if path.endswith('.xacro'):
            hint = ('  (xacro needs the catkin workspace on ROS_PACKAGE_PATH: '
                    'source devel/setup.bash first)')
        return {'display': spec['display'],
                'error': '{}: {}{}'.format(type(exc).__name__, exc, hint)}

    wg = wheel_geometry(model, spec['wheel_pattern'])
    radius = wg.get('radius', float('nan'))
    com = model.center_of_mass()
    lo, hi, skipped = model.collision_bbox()

    axle_h = wg.get('axle_height', float('nan'))
    # The drop to the ground, declared per platform: see 'contact_offset_m'.
    drop = spec.get('contact_offset_m', radius)
    contact_z = (axle_h - drop
                 if np.isfinite(axle_h) and np.isfinite(drop) else 0.0)

    w_max, v_max = max_joint_speed(model, spec['wheel_pattern'], radius)
    v_cmd, wp_path = commanded_max_speed(src, spec['waypoints'])

    out = {
        'display': spec['display'],
        'description_file': os.path.relpath(path, src),
        'links': len(model.links),
        'joints': len(model.joints),
        'total_mass_kg': model.total_mass(),
        'link_mass_kg': model.link_mass(),
        'embedded_sdf_mass_kg': model.embedded_mass(),
        'cog_x_m': com[0], 'cog_y_m': com[1], 'cog_z_m': com[2],
        'cog_height_above_contact_m': (com[2] - contact_z
                                       if np.isfinite(com[2]) else float('nan')),
        'wheel_count': wg.get('count', 0),
        'wheel_radius_m': radius,
        'track_m': wg.get('track', float('nan')),
        'wheelbase_m': wg.get('wheelbase', float('nan')),
        'max_wheel_rate_rad_s': w_max,
        'max_speed_kinematic_m_s': v_max,
        'max_speed_commanded_m_s': v_cmd,
        'waypoint_config': (os.path.relpath(wp_path, src) if wp_path else 'n/a'),
        'ground_clearance_m': ground_clearance(model, spec['wheel_pattern'],
                                               axle_h, drop),
        'meshes_skipped_in_bbox': skipped,
        'wheel_centres': wg.get('centres', {}),
        'controller': spec.get('controller', 'n/a'),
        'controller_params': controller_params(model, src, spec),
    }
    if lo is not None:
        out.update({
            'bbox_length_m': hi[0] - lo[0],
            'bbox_width_m': hi[1] - lo[1],
            'bbox_height_m': hi[2] - lo[2],
        })
    return out


# ---------------------------------------------------------------------------

ROWS = [
    ('total_mass_kg', 'Total mass', 'kg', '.3f'),
    ('link_mass_kg', '  of which URDF/SDF links', 'kg', '.3f'),
    ('embedded_sdf_mass_kg', '  of which embedded SDF bodies', 'kg', '.3f'),
    ('links', 'Links', '-', '.0f'),
    ('joints', 'Joints', '-', '.0f'),
    ('wheel_count', 'Ground-contact bodies', '-', '.0f'),
    ('bbox_length_m', 'Length (collision bbox)', 'm', '.4f'),
    ('bbox_width_m', 'Width (collision bbox)', 'm', '.4f'),
    ('bbox_height_m', 'Height (collision bbox)', 'm', '.4f'),
    ('wheelbase_m', 'Wheelbase', 'm', '.4f'),
    ('track_m', 'Track (max)', 'm', '.4f'),
    ('wheel_radius_m', 'Wheel/sprocket radius', 'm', '.4f'),
    ('ground_clearance_m', 'Ground clearance', 'm', '.4f'),
    ('cog_x_m', 'CoG x (base_link)', 'm', '.4f'),
    ('cog_y_m', 'CoG y (base_link)', 'm', '.4f'),
    ('cog_z_m', 'CoG z (base_link)', 'm', '.4f'),
    ('cog_height_above_contact_m', 'CoG height above contact plane', 'm', '.4f'),
    ('max_wheel_rate_rad_s', 'Wheel rate limit', 'rad/s', '.3f'),
    ('max_speed_kinematic_m_s', 'Max speed (kinematic limit)', 'm/s', '.3f'),
    ('max_speed_commanded_m_s', 'Max speed (commanded)', 'm/s', '.3f'),
]


def fmt(v, spec):
    if v is None:
        return 'n/a'
    if isinstance(v, float) and not np.isfinite(v):
        return 'n/a'
    try:
        return ('{:' + spec + '}').format(v)
    except (ValueError, TypeError):
        return str(v)


def render_table(results, keys):
    header = '{:<34s}'.format('Property [unit]') + ''.join(
        '{:>22s}'.format(results[k]['display']) for k in keys)
    lines = ['=' * len(header),
             'MECHANICAL SPECIFICATIONS (parsed from the model files)',
             '=' * len(header), header, '-' * len(header)]
    for key, label, unit, spec in ROWS:
        cells = [fmt(results[k].get(key), spec) for k in keys]
        if all(c == 'n/a' for c in cells):
            continue
        lines.append('{:<34s}'.format('{} [{}]'.format(label, unit)) +
                     ''.join('{:>22s}'.format(c) for c in cells))
    lines.append('-' * len(header))
    lines.append('{:<34s}'.format('Source file') +
                 ''.join('{:>22s}'.format(
                     os.path.basename(results[k].get('description_file', '?')))
                     for k in keys))
    lines.append('=' * len(header))

    # Controller configuration is reported as free text: the three platforms
    # declare it in genuinely different places and forcing it into the numeric
    # table above would misrepresent it.
    lines.append('')
    lines.append('DRIVE CONTROLLER CONFIGURATION')
    lines.append('-' * len(header))
    for k in keys:
        r = results[k]
        lines.append('{}:'.format(r['display']))
        lines.append('  controller: {}'.format(r.get('controller', 'n/a')))
        params = r.get('controller_params') or {}
        if params:
            for pk in sorted(params):
                lines.append('    {:<20s} {}'.format(pk, params[pk]))
        else:
            lines.append('    (no explicit limits declared; the drive is '
                         'torque-limited, so the effective ceiling is the '
                         'commanded max speed above)')
    lines.append('=' * len(header))

    notes = []
    for k in keys:
        sk = results[k].get('meshes_skipped_in_bbox', 0)
        if sk:
            notes.append('  {}: {} mesh collision(s) excluded from the bounding '
                         'box (extent not evaluated from binary meshes).'
                         .format(results[k]['display'], sk))
    if notes:
        lines.append('Notes:')
        lines.extend(notes)
    return '\n'.join(lines)


def write_csv(results, keys, path):
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['property', 'unit'] + [results[k]['display'] for k in keys])
        for key, label, unit, _ in ROWS:
            vals = [results[k].get(key, '') for k in keys]
            if all(v in ('', None) for v in vals):
                continue
            w.writerow([label, unit] + vals)
        w.writerow(['Source file', '-'] +
                   [results[k].get('description_file', '') for k in keys])


def print_wheel_map(results, keys):
    print('\nWheel centres in base_link (zero configuration), metres')
    for k in keys:
        r = results[k]
        centres = r.get('wheel_centres') or {}
        if not centres:
            continue
        print('\n{}:'.format(r['display']))
        print('  {:<16s}{:>9s}{:>9s}{:>9s}'.format('link', 'x', 'y', 'z'))
        for name, c in sorted(centres.items(), key=lambda kv: (kv[1][0], kv[1][1])):
            print('  {:<16s}{:>9.3f}{:>9.3f}{:>9.3f}'.format(name, *c))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=os.path.abspath(os.path.join(here, '..', '..')))
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--wheel-map', action='store_true',
                    help='also print each wheel centre')
    args = ap.parse_args()

    results = {}
    for key in ORDER:
        results[key] = analyse(key, ROBOTS[key], args.src)
        if 'error' in results[key]:
            print('WARNING {}: {}'.format(key, results[key]['error']))

    usable = [k for k in ORDER if 'error' not in results[k]]
    if not usable:
        print('No model could be parsed.')
        return 1

    table = render_table(results, usable)
    print(table)

    if args.wheel_map:
        print_wheel_map(results, usable)

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, 'table_robot_specs.txt'), 'w') as f:
            f.write(table + '\n')
        write_csv(results, usable,
                  os.path.join(args.output_dir, 'robot_specs.csv'))
        print('\nWritten to {}'.format(os.path.abspath(args.output_dir)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
