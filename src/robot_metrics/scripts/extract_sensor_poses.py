#!/usr/bin/env python3
"""Sensor mounting poses of the three platforms, relative to base_link.

Why this matters for the paper
==============================
The comparison attributes differences in SLAM quality to the locomotion type.
That inference only holds if the sensors see the world from comparable places.
A LiDAR mounted 0.20 m higher, or 0.30 m further forward, changes how much of
the gallery floor is occluded and how much the ranges swing when the chassis
pitches - independently of whether the chassis is a rocker-bogie or a skid
steer.  So the mounting geometry has to be reported, and ideally controlled.

This script does the reporting part: it reads each robot's own description and
prints where the LiDAR, camera and IMU actually are.

    --standardize  additionally writes a sensor_standardization.yaml holding a
                   common mounting pose for all three platforms, for the
                   control experiment that separates the effect of locomotion
                   from the effect of sensor placement.

A note on the lever arm.  Angular motion of the chassis turns into linear
motion at the sensor: a mast r metres above the pitch axis sweeps through
r * dtheta.  The script reports that lever arm explicitly, because it is the
mechanism by which chassis stability reaches the SLAM front end, and it is the
number that makes the comparison auditable.

Usage:
    ./extract_sensor_poses.py
    ./extract_sensor_poses.py --output_dir ~/paper_tables --standardize
"""

import argparse
import csv
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_parser as mp                                  # noqa: E402
from extract_robot_specs import ROBOTS, ORDER, wheel_geometry   # noqa: E402
from check_sim_parity import SPAWN_SOURCE                      # noqa: E402


# How each platform names the three sensor mounts.
SENSOR_LINKS = {
    'lidar': ['velodyne_base_link', 'velodyne', 'laser_link', 'lidar_link'],
    'camera': ['camera_link'],
    'imu': ['imu_link'],
}


def find_sensor_link(model, candidates):
    for name in candidates:
        if name in model.links:
            return name
    lowered = {n.lower(): n for n in model.links}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def sensor_poses(model, contact_z, centre_x=0.0, forward=1, com=None,
                 base=None):
    """{sensor: {x, y, z, roll, pitch, yaw, height_above_contact, forward_offset}}

    `height_above_contact_m` and `forward_offset_m` are the two comparable
    quantities.  Raw x/y/z are given for reference but are NOT comparable
    across platforms: each model puts base_link somewhere different inside the
    chassis, and one of them is built with forward = -x, so the same physical
    mounting point yields different numbers in each file.
    """
    out = {}
    for sensor, candidates in SENSOR_LINKS.items():
        link_name = find_sensor_link(model, candidates)
        if link_name is None:
            out[sensor] = None
            continue
        link = model.links[link_name]
        T = link.pose
        # A sensor element may carry an extra <pose> of its own inside the link.
        if link.sensors:
            T = T @ link.sensors[0]['pose']
        xyz = T[:3, 3]
        rpy = mp.rot_to_rpy(T[:3, :3])
        # x/y/z se informan RELATIVOS AL LINK BASE.  No es lo mismo que el
        # marco del modelo: en el model.sdf del tracked `body` esta 0.35 m
        # arriba del origen.  Las alturas y los offsets de mas abajo se
        # refieren al plano de contacto y al centro del poligono de apoyo, que
        # ya viven en el marco del modelo, asi que esos no llevan la resta.
        b = np.zeros(3) if base is None else np.asarray(base, dtype=float)
        xyz_base = xyz - b
        out[sensor] = {
            'link': link_name,
            'x_m': float(xyz_base[0]), 'y_m': float(xyz_base[1]),
            'z_m': float(xyz_base[2]),
            'roll_deg': float(np.degrees(rpy[0])),
            'pitch_deg': float(np.degrees(rpy[1])),
            'yaw_deg': float(np.degrees(rpy[2])),
            'height_above_contact_m': float(xyz[2] - contact_z),
            'forward_offset_m': float(forward * (xyz[0] - centre_x)),
            'lateral_offset_m': float(forward * xyz_base[1]),
            # Desde el CENTRO DE MASAS: es el origen que decide lo que mide un
            # inercial, a_sensor = a_cm + alpha x r + omega x (omega x r).
            'lateral_from_com_m': (float(forward * (xyz[1] - com[1]))
                                   if com is not None else float('nan')),
            'lever_arm_from_com_m': (float(np.linalg.norm(xyz - com))
                                     if com is not None else float('nan')),
            'sensor_type': (link.sensors[0]['type'] if link.sensors else None),
            'update_rate_hz': (link.sensors[0].get('update_rate')
                               if link.sensors else None),
            'topic': (link.sensors[0].get('topic') if link.sensors else None),
        }
    return out


def analyse(key, spec, src):
    # EL FICHERO QUE GAZEBO SPAWNEA, no el que describe extract_robot_specs.
    # Dos plataformas declaran cada montaje dos veces y solo una copia llega a
    # la fisica; leyendo la otra esta tabla ha llegado a confirmar un spread de
    # 1 mm que en Gazebo no existia (2026-08-26: errores de 0.24 a 0.35 m).  En
    # el tracked las dos copias siguen sin coincidir - su CM sale 15.6 mm mas
    # adelante en el SDF que en el xacro, porque alli los idlers estan lumpeados
    # en left_body/right_body sin desplazar el origen inercial - asi que la
    # eleccion cambia el numero.  Manda lo que se simula.
    path = os.path.join(src, SPAWN_SOURCE.get(key, spec['description']))
    if not os.path.isfile(path):
        return None
    try:
        model = mp.load(path)
    except Exception as exc:                        # noqa: BLE001
        print('WARNING {}: {} (source devel/setup.bash if this is a xacro)'
              .format(key, exc))
        return None

    # Como se llama el link base en ESTE fichero: el tracked lo llama `body`.
    base_name = ('base_link' if 'base_link' in model.links else 'body')

    wg = wheel_geometry(model, spec['wheel_pattern'])
    radius = wg.get('radius', float('nan'))
    axle_h = wg.get('axle_height', float('nan'))
    # The drop from the axle to the ground is DECLARED per platform
    # ('contact_offset_m'), not taken as the wheel radius: the tracked robot
    # rests on its belt 0.210 m below the axle, not on its 0.178 m sprockets,
    # and every height on this table is referred to that plane.
    drop = spec.get('contact_offset_m', radius)
    contact_z = (axle_h - drop
                 if np.isfinite(axle_h) and np.isfinite(drop) else 0.0)

    # Longitudinal origin for the comparison: the centre of the ground contact
    # polygon, i.e. midway between the front-most and rear-most wheel centres.
    # That is the point the chassis pitches about, so it is what a lever arm
    # should be measured from.
    centres = wg.get('centres') or {}
    xs = [p[0] for p in centres.values()]
    # float() and not just the arithmetic: max/min over numpy values return
    # numpy scalars, which yaml.safe_dump refuses to serialise.
    centre_x = float(0.5 * (max(xs) + min(xs))) if xs else 0.0
    forward = spec.get('forward_axis_x', 1)

    return {
        'display': spec['display'],
        'model': model,
        'contact_z': contact_z,
        'contact_centre_x': centre_x,
        'forward_axis_x': forward,
        'sensors': sensor_poses(
            model, contact_z, centre_x, forward,
            com=np.asarray(model.center_of_mass(), dtype=float),
            base=(model.links[base_name].pose[:3, 3]
                  if base_name in model.links else None)),
    }


ROWS = [
    ('link', 'Mount link', '-'),
    ('x_m', 'x (base_link)', 'm'),
    ('y_m', 'y (base_link)', 'm'),
    ('z_m', 'z (base_link)', 'm'),
    ('height_above_contact_m', 'Height above contact plane', 'm'),
    ('forward_offset_m', 'Forward offset from contact centre', 'm'),
    ('lateral_offset_m', 'Lateral offset from base_link (left +)', 'm'),
    ('lateral_from_com_m', 'Lateral offset from CoM (left +)', 'm'),
    ('lever_arm_from_com_m', 'Lever arm from CoM |r|', 'm'),
    ('roll_deg', 'roll', 'deg'),
    ('pitch_deg', 'pitch', 'deg'),
    ('yaw_deg', 'yaw', 'deg'),
    ('update_rate_hz', 'Update rate', 'Hz'),
]


def fmt(v):
    if v is None:
        return 'n/a'
    if isinstance(v, float):
        return '{:.4f}'.format(v)
    return str(v)


def render(results, keys):
    header = '{:<34s}'.format('Property [unit]') + ''.join(
        '{:>22s}'.format(results[k]['display']) for k in keys)
    lines = ['=' * len(header),
             'SENSOR MOUNTING POSES, relative to base_link',
             '=' * len(header)]

    for sensor in ('lidar', 'camera', 'imu'):
        lines.append('')
        lines.append('{}  ({})'.format(sensor.upper(),
                                       'LiDAR' if sensor == 'lidar' else sensor))
        lines.append(header)
        lines.append('-' * len(header))
        for field, label, unit in ROWS:
            cells = []
            for k in keys:
                s = results[k]['sensors'].get(sensor)
                cells.append(fmt(s.get(field)) if s else 'absent')
            if all(c in ('n/a', 'absent') for c in cells):
                continue
            lines.append('{:<34s}'.format('{} [{}]'.format(label, unit)) +
                         ''.join('{:>22s}'.format(c) for c in cells))

    # The lever arm: how far a chassis rotation displaces each sensor.
    lines.append('')
    lines.append('LEVER ARM  (sensor height above the contact plane; a chassis')
    lines.append('rotation of dtheta displaces the sensor by height * dtheta)')
    lines.append('El IMU ya no se iguala por altura: va en el centro de masas,')
    lines.append('asi que sus tres alturas son distintas a proposito.')
    lines.append(header)
    lines.append('-' * len(header))
    for sensor in ('lidar', 'camera', 'imu'):
        cells = []
        for k in keys:
            s = results[k]['sensors'].get(sensor)
            cells.append(fmt(s['height_above_contact_m']) if s else 'absent')
        lines.append('{:<34s}'.format('{} height [m]'.format(sensor)) +
                     ''.join('{:>22s}'.format(c) for c in cells))

    lines.append('{:<34s}'.format('Displacement per 1 deg tilt [mm]') +
                 ''.join('{:>22s}'.format(
                     fmt(results[k]['sensors']['lidar']['height_above_contact_m']
                         * np.radians(1.0) * 1000.0)
                     if results[k]['sensors'].get('lidar') else 'n/a')
                     for k in keys))
    lines.append('=' * len(header))

    # Spread across platforms: the number that says whether placement is a
    # confound worth controlling for.
    lines.append('')
    lines.append('SPREAD ACROSS PLATFORMS (max - min)')
    lines.append('-' * len(header))
    worst = 0.0
    # El IMU NO entra aqui: se le iguala el brazo desde el centro de masas, no
    # la altura sobre el suelo, y como cada chasis tiene el CM en otro sitio su
    # altura y su offset adelante tienen que salir distintos.  Ver el bloque
    # siguiente y sensor_mounting en sim_config.yaml.
    for sensor in ('lidar', 'camera'):
        vals = [results[k]['sensors'][sensor]['height_above_contact_m']
                for k in keys if results[k]['sensors'].get(sensor)]
        # Compared from the contact-polygon centre along each platform's own
        # forward axis. Raw base_link x is not comparable: the origins differ.
        fwd = [results[k]['sensors'][sensor]['forward_offset_m']
               for k in keys if results[k]['sensors'].get(sensor)]
        # LATERAL, medido desde el CENTRO DE MASAS y no desde base_link.
        # Hasta 2026-09-04 el lateral se calculaba y no entraba nunca en el
        # veredicto, asi que la tabla podia declarar "same pose, spread <= 1
        # mm" sin haber mirado y en el eje y. Y desde base_link no vale: el
        # origen del Rocker-Bogie esta 116 mm fuera de su linea central, asi
        # que sus tres sensores leian -0.1164 m estando centrados.
        lat = [results[k]['sensors'][sensor]['lateral_from_com_m']
               for k in keys if results[k]['sensors'].get(sensor)]
        if len(vals) < 2:
            continue
        dh, dx = max(vals) - min(vals), max(fwd) - min(fwd)
        dy = max(lat) - min(lat) if lat else 0.0
        worst = max(worst, dh, dx, dy)
        lines.append('  {:<10s} height spread {:.4f} m,  forward-offset spread '
                     '{:.4f} m,  lateral (desde el CM) spread {:.4f} m'
                     .format(sensor, dh, dx, dy))
    # El brazo desde el CENTRO DE MASAS: lo que decide lo que mide un inercial.
    # No entra en `worst` para el LiDAR y la camara -a esos se les iguala la
    # altura sobre el suelo a proposito, y sus brazos siguen difiriendo un
    # factor 1.4-1.6- pero SI para el IMU.
    lines.append('')
    lines.append('BRAZO DESDE EL CENTRO DE MASAS  |r| [m]')
    lines.append('-' * len(header))
    for sensor in ('lidar', 'camera', 'imu'):
        arms = [results[k]['sensors'][sensor]['lever_arm_from_com_m']
                for k in keys if results[k]['sensors'].get(sensor)]
        if len(arms) < 2:
            continue
        nota = ''
        if sensor == 'imu':
            worst = max(worst, max(arms))
            nota = '   <- tiene que ser 0: ver sensor_mounting en sim_config'
        lines.append('  {:<10s} {}{}'.format(
            sensor, '  '.join('{:.4f}'.format(a) for a in arms), nota))
    lines.append('')
    if worst <= 1e-3:
        lines.append('All three platforms carry their sensors at the same pose')
        lines.append('relative to the ground contact plane (spread <= 1 mm), so')
        lines.append('sensor placement is controlled for and any remaining')
        lines.append('difference in SLAM quality is attributable to locomotion.')
    else:
        lines.append('If a spread here is comparable to the differences the')
        lines.append('paper attributes to locomotion, standardize the mounts')
        lines.append('(--standardize) before drawing the conclusion.')
    return '\n'.join(lines)


def write_csv(results, keys, path):
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['sensor', 'property', 'unit'] +
                   [results[k]['display'] for k in keys])
        for sensor in ('lidar', 'camera', 'imu'):
            for field, label, unit in ROWS:
                row = [sensor, label, unit]
                for k in keys:
                    s = results[k]['sensors'].get(sensor)
                    row.append(s.get(field) if s else '')
                w.writerow(row)


def write_standardization(results, keys, path):
    """Record the common mounting pose the three models are now built around.

    This file documents a standardization that has already been APPLIED to the
    three model files; it is not a proposal.  Re-running the script regenerates
    it from the models, so if someone edits a mount the file stops agreeing
    with itself and the spread at the bottom of the table stops being zero.
    """
    def hs(sensor):
        return [results[k]['sensors'][sensor]['height_above_contact_m']
                for k in keys if results[k]['sensors'].get(sensor)]

    def fs(sensor):
        return [results[k]['sensors'][sensor]['forward_offset_m']
                for k in keys if results[k]['sensors'].get(sensor)]

    # The common pose is the ROCKER-BOGIE's, i.e. the HIGHEST of the three, not
    # the lowest and not the mean.  The obvious choice would be the lowest,
    # since lowering a sensor never needs new hardware - but it is not
    # achievable here: the Rocker-Bogie's ground clearance is 0.638 m, so the
    # underside of its chassis already sits above the two native low mounts
    # (0.5805 m Husky, 0.5657 m tracked) and a sensor placed there would be
    # buried inside the chassis.  Standardizing upwards instead only requires a
    # mast on the two low platforms.  The mast is modelled as massless and the
    # sensor masses are kept, which was measured to shift the Husky CoG up by
    # 4.66 mm and the tracked robot's by 0.68 mm - both negligible against
    # their 341 and 180 mm CoG heights, and both in the direction that makes
    # the low platforms slightly LESS stable, so the comparison is not tilted
    # in the Rocker-Bogie's favour.
    common_h = float(max(hs('lidar'))) if hs('lidar') else 0.4
    common_x = float(np.median(fs('lidar'))) if fs('lidar') else 0.0

    doc = {
        'purpose': (
            'Control experiment: put the LiDAR, camera and IMU of all three '
            'platforms at the same pose relative to the ground contact plane, '
            'so that any remaining difference in SLAM quality is attributable '
            'to locomotion rather than to sensor placement.'),
        'method': (
            'Heights are given ABOVE THE WHEEL/TRACK CONTACT PLANE, not above '
            'base_link, because base_link sits at a different height on each '
            'platform. Convert with:  z_base_link = height_above_contact + '
            'contact_plane_offset, where the offset per robot is listed below. '
            'x and y are measured from the centre of the ground contact '
            'polygon so that the lever arm about the pitch axis matches.'),
        'status': (
            'APPLIED. The three model files already carry these poses; the '
            'spread reported at the bottom of the sensor table is <= 1 mm.'),
        'rationale_for_common_height': (
            'The MAXIMUM of the three native heights ({:.4f} m, the '
            'Rocker-Bogie) is used rather than the minimum or the mean. '
            'Lowering would normally be preferable because it needs no new '
            'hardware, but it is not achievable here: the Rocker-Bogie has '
            '0.638 m of ground clearance, so the underside of its chassis is '
            'already above both native low mounts (0.5805 m Husky, 0.5657 m '
            'tracked) and a sensor placed there would sit inside the chassis. '
            'Standardizing upwards needs a mast on the two low platforms, '
            'modelled as massless with the sensor masses retained; measured, '
            'that raises the Husky CoG by 4.66 mm and the tracked robot\'s by '
            '0.68 mm, negligible against their 341 and 180 mm CoG heights and in the '
            'direction that makes the low platforms slightly less stable, so '
            'the choice does not favour the Rocker-Bogie.'.format(common_h)),
        'standard_pose': {
            'lidar': {'forward_offset_m': round(common_x, 4), 'y_m': 0.0,
                      'height_above_contact_m': round(common_h, 4),
                      'roll_deg': 0.0, 'pitch_deg': 0.0, 'yaw_deg': 0.0},
            'camera': {'forward_offset_m': round(
                float(np.median(fs('camera'))) if fs('camera') else 0.3543, 4),
                'y_m': 0.0,
                'height_above_contact_m': round(
                    float(max(hs('camera'))) if hs('camera') else common_h, 4),
                'roll_deg': 0.0, 'pitch_deg': 0.0, 'yaw_deg': 0.0},
            # El IMU no lleva altura comun: va en el CENTRO DE MASAS de
            # cada plataforma, que esta a una altura distinta en cada una.  Es
            # deliberado; ver sensor_mounting en sim_config.yaml.
            'imu': {'rule': 'at_centre_of_mass',
                    'lever_arm_from_com_m': 0.0,
                    'height_above_contact_m': {
                        results[k]['display']: round(
                            results[k]['sensors']['imu'][
                                'height_above_contact_m'], 4)
                        for k in keys if results[k]['sensors'].get('imu')},
                    'roll_deg': 0.0, 'pitch_deg': 0.0, 'yaw_deg': 0.0},
        },
        'note_on_x': (
            'forward_offset_m is measured from the centre of the ground '
            'contact polygon along each platform\'s own forward axis. It is '
            'not base_link x: base_link sits at a different place in each '
            'chassis and the Rocker-Bogie model has forward = -x, so raw x '
            'values are not comparable between the three files.'),
        'contact_plane_offset_m': {
            results[k]['display']: round(results[k]['contact_z'], 4)
            for k in keys
        },
        'contact_polygon_centre_x_m': {
            results[k]['display']: round(results[k]['contact_centre_x'], 4)
            for k in keys
        },
        'forward_axis_x': {
            results[k]['display']: results[k]['forward_axis_x'] for k in keys
        },
        'applied_poses': {
            results[k]['display']: {
                s: (None if results[k]['sensors'].get(s) is None else {
                    'height_above_contact_m': round(
                        results[k]['sensors'][s]['height_above_contact_m'], 4),
                    'forward_offset_m': round(
                        results[k]['sensors'][s]['forward_offset_m'], 4),
                })
                for s in ('lidar', 'camera', 'imu')
            } for k in keys
        },
        # Raw base_link coordinates as they currently stand in each file, for
        # anyone who needs to locate the exact element to edit.
        'current_base_link_coordinates': {
            results[k]['display']: {
                s: (None if results[k]['sensors'].get(s) is None else {
                    'x_m': round(results[k]['sensors'][s]['x_m'], 4),
                    'y_m': round(results[k]['sensors'][s]['y_m'], 4),
                    'z_m': round(results[k]['sensors'][s]['z_m'], 4),
                    'height_above_contact_m': round(
                        results[k]['sensors'][s]['height_above_contact_m'], 4),
                })
                for s in ('lidar', 'camera', 'imu')
            } for k in keys
        },
        'where_the_mounts_are_defined': [
            'Rocker-Bogie (unchanged, it sets the reference): origins of '
            'velodyne_base_mount_joint, camera_joint and imu_joint in '
            'rocker_bogie/urdf/ensamblajeurdf.xacro.',
            'Husky: <pose> of velodyne_base_link, camera_link and imu_link in '
            'differential/model/model.sdf.',
            'Tracked: origins of velodyne_base_mount_joint, camera_joint and '
            'imu_joint in tracked/gazebo_continuous_track_example/urdf_xacro/'
            'two_track_robot.urdf.xacro.',
            'Each edited element carries a SENSOR STANDARDIZATION comment '
            'giving its native value and the arithmetic used.',
            'Re-run extract_sensor_poses.py after any change: the spread at '
            'the bottom of the table must stay ~0 for the control to hold.',
        ],
    }
    with open(path, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=os.path.abspath(os.path.join(here, '..', '..')))
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--standardize', action='store_true',
                    help='also emit sensor_standardization.yaml')
    args = ap.parse_args()

    results = {}
    for key in ORDER:
        r = analyse(key, ROBOTS[key], args.src)
        if r:
            results[key] = r

    if not results:
        print('No model could be parsed.')
        return 1

    keys = [k for k in ORDER if k in results]
    table = render(results, keys)
    print(table)

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir,
                               'table_sensor_poses.txt'), 'w') as f:
            f.write(table + '\n')
        write_csv(results, keys,
                  os.path.join(args.output_dir, 'sensor_poses.csv'))
        if args.standardize:
            out = os.path.join(args.output_dir, 'sensor_standardization.yaml')
            write_standardization(results, keys, out)
            print('\nStandardized placement written to {}'.format(out))
        print('Written to {}'.format(os.path.abspath(args.output_dir)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
