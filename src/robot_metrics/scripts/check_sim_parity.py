#!/usr/bin/env python3
"""Verify that the three platforms are simulated under identical conditions.

The reviewer's objection was that nothing guaranteed the three robots ran with
the same physics, the same terrain and the same obstacle.  This script turns
that guarantee into a test that has to pass before a campaign is considered
valid.  It checks, in order:

  1. the three lcmine.world files are byte-identical;
  2. the world actually declares a <physics> block, and its contents match
     sim_config.yaml (engine, step size, update rate, solver, gravity);
  3. every drivable surface declares its friction explicitly, with the values
     recorded in sim_config.yaml;
  4. the step obstacle has the geometry recorded in sim_config.yaml;
  5. the shared terrain models (lc_mine, grass_plane, smaze_wall) are
     byte-identical across the three packages;
  6. the spawn x/y are the same for the three robots.

Exit status is 0 on PASS and 1 on FAIL, so it can gate a campaign script.

Usage:
    rosrun robot_metrics check_sim_parity.py
    ./check_sim_parity.py --src /path/to/catkin_ws/src
"""

import argparse
import hashlib
import os
import sys
import xml.etree.ElementTree as ET

import yaml

# Package-relative locations of the world file for each robot.
WORLD_PATHS = {
    'rocker_bogie': 'rocker_bogie/world/mine/lcmine.world',
    'differential': 'differential/world/mine/lcmine.world',
    'tracked': 'tracked/gazebo_continuous_track_example/world/mine/lcmine.world',
}

# Terrain models that must be shared verbatim between packages.
TERRAIN_MODELS = ['lc_mine', 'grass_plane', 'smaze_wall', 'sun_2', 'home']

WORLD_DIRS = {
    'rocker_bogie': 'rocker_bogie/world/mine',
    'differential': 'differential/world/mine',
    'tracked': 'tracked/gazebo_continuous_track_example/world/mine',
}

# Launch files that carry the spawn pose.
LAUNCH_PATHS = {
    'rocker_bogie': 'rocker_bogie/launch/lcmine_rocker_bogie_world.launch',
    'differential': 'differential/launch/lcmine_husky_world.launch',
    'tracked': ('tracked/gazebo_continuous_track_example/launch/'
                'lcmine_two_track_world.launch'),
}


class Report(object):
    """Collects PASS/FAIL lines and remembers whether anything failed."""

    def __init__(self):
        self.failed = False
        self.lines = []

    def ok(self, msg):
        self.lines.append('  [ OK ] {}'.format(msg))

    def fail(self, msg):
        self.failed = True
        self.lines.append('  [FAIL] {}'.format(msg))

    def warn(self, msg):
        self.lines.append('  [warn] {}'.format(msg))

    def section(self, title):
        self.lines.append('')
        self.lines.append(title)
        self.lines.append('-' * len(title))

    def dump(self):
        print('\n'.join(self.lines))


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def find_text(elem, path, default=None):
    node = elem.find(path)
    return node.text.strip() if node is not None and node.text else default


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_worlds_identical(src, rep):
    rep.section('1. World files byte-identical across the three packages')
    hashes = {}
    for robot, rel in WORLD_PATHS.items():
        path = os.path.join(src, rel)
        if not os.path.isfile(path):
            rep.fail('missing world file: {}'.format(rel))
            continue
        hashes[robot] = md5(path)

    if len(set(hashes.values())) == 1 and len(hashes) == len(WORLD_PATHS):
        rep.ok('all 3 lcmine.world identical (md5 {})'.format(
            list(hashes.values())[0][:12]))
    else:
        rep.fail('lcmine.world differs between packages:')
        for robot, h in sorted(hashes.items()):
            rep.fail('    {:<14s} {}'.format(robot, h[:12]))
        rep.fail('    -> the three robots are NOT running the same world')


def check_physics(world_root, cfg, rep):
    rep.section('2. Physics block matches sim_config.yaml')
    phys = world_root.find('./world/physics')
    if phys is None:
        rep.fail('no <physics> block in the world: Gazebo compiled-in defaults '
                 'are being used and cannot be reported in the paper')
        return

    want = cfg['physics']

    engine = phys.get('type')
    if engine == want['engine']:
        rep.ok('engine = {}'.format(engine))
    else:
        rep.fail('engine = {} (config says {})'.format(engine, want['engine']))

    checks = [
        ('max_step_size', 'max_step_size_s', 's'),
        ('real_time_update_rate', 'real_time_update_rate_hz', 'Hz'),
    ]
    for tag, key, unit in checks:
        val = find_text(phys, tag)
        if val is None:
            rep.fail('<{}> not declared'.format(tag))
        elif approx(val, want[key]):
            rep.ok('{} = {} {}'.format(tag, val, unit))
        else:
            rep.fail('{} = {} (config says {})'.format(tag, val, want[key]))

    solver = phys.find('./ode/solver')
    if solver is None:
        rep.fail('no <ode><solver> block')
    else:
        for tag, key in [('type', 'type'), ('iters', 'iters'), ('sor', 'sor')]:
            val = find_text(solver, tag)
            exp = want['solver'][key]
            if val is None:
                rep.fail('solver <{}> not declared'.format(tag))
            elif str(val) == str(exp) or (
                    _is_number(val) and approx(val, exp)):
                rep.ok('solver.{} = {}'.format(tag, val))
            else:
                rep.fail('solver.{} = {} (config says {})'.format(tag, val, exp))

    grav = find_text(world_root, './world/gravity')
    if grav is None:
        rep.warn('<gravity> not declared at world level; Gazebo default '
                 '(0 0 -9.8) applies. Config says {}'.format(
                     want['gravity_mps2']))
    else:
        got = [float(v) for v in grav.split()]
        exp = [float(v) for v in want['gravity_mps2']]
        if all(approx(a, b) for a, b in zip(got, exp)):
            rep.ok('gravity = {} m/s^2'.format(got))
        else:
            rep.fail('gravity = {} (config says {})'.format(got, exp))


def _is_number(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def check_step_obstacle(world_root, cfg, rep):
    rep.section('3. Step obstacle geometry')
    want = cfg['step_obstacle']
    name = want['model_name']

    model = None
    for m in world_root.findall('./world/model'):
        if m.get('name') == name:
            model = m
            break
    if model is None:
        rep.fail('model "{}" not found in the world'.format(name))
        return

    pose = find_text(model, 'pose')
    if pose is not None:
        got = [float(v) for v in pose.split()]
        exp = [float(v) for v in want['pose_xyzrpy']]
        if all(approx(a, b, 1e-6) for a, b in zip(got, exp)):
            rep.ok('pose = {}'.format(got))
        else:
            rep.fail('pose = {} (config says {})'.format(got, exp))

    size = find_text(model, './link/collision/geometry/box/size')
    if size is None:
        rep.fail('step obstacle has no box collision geometry')
    else:
        got = [float(v) for v in size.split()]
        exp = [float(v) for v in want['size_m']]
        if all(approx(a, b, 1e-6) for a, b in zip(got, exp)):
            rep.ok('size = {} m  (step height {} m)'.format(
                got, want['height_m']))
        else:
            rep.fail('size = {} m (config says {} m) -> the platforms are '
                     'crossing different obstacles'.format(got, exp))

    mu = find_text(model, './link/collision/surface/friction/ode/mu')
    exp_mu = cfg['terrain']['step']['mu1']
    if mu is None:
        rep.fail('step obstacle declares no friction')
    elif approx(mu, exp_mu):
        rep.ok('step friction mu = {}'.format(mu))
    else:
        rep.fail('step friction mu = {} (config says {})'.format(mu, exp_mu))


def check_terrain_friction(src, cfg, rep):
    rep.section('4. Drivable surfaces declare friction explicitly')
    ref_dir = os.path.join(src, WORLD_DIRS['rocker_bogie'])
    for model in ['lc_mine', 'grass_plane']:
        path = os.path.join(ref_dir, model, 'model.sdf')
        if not os.path.isfile(path):
            rep.fail('{}: model.sdf missing'.format(model))
            continue
        root = ET.parse(path).getroot()
        mu = None
        for coll in root.iter('collision'):
            node = coll.find('./surface/friction/ode/mu')
            if node is not None and node.text:
                mu = node.text.strip()
                break
        exp = cfg['terrain'][model]['mu1']
        if mu is None:
            rep.fail('{}: no <surface><friction> -> terrain friction is an '
                     'undocumented ODE default'.format(model))
        elif approx(mu, exp):
            rep.ok('{}: mu = {}'.format(model, mu))
        else:
            rep.fail('{}: mu = {} (config says {})'.format(model, mu, exp))


def check_terrain_shared(src, rep):
    rep.section('5. Terrain models identical across packages')
    for model in TERRAIN_MODELS:
        hashes = {}
        for robot, wd in WORLD_DIRS.items():
            path = os.path.join(src, wd, model, 'model.sdf')
            if os.path.isfile(path):
                hashes[robot] = md5(path)
        if not hashes:
            continue
        if len(hashes) < len(WORLD_DIRS):
            rep.warn('{}: present in {}/{} packages'.format(
                model, len(hashes), len(WORLD_DIRS)))
        if len(set(hashes.values())) == 1:
            rep.ok('{}: identical'.format(model))
        else:
            rep.fail('{}: differs between packages {}'.format(
                model, {k: v[:8] for k, v in hashes.items()}))


# Robot descriptions, for the sensor-noise check.  The tracked robot's sensors
# live in the _gazebo variant, so the plain xacro would report none at all.
DESCRIPTION_PATHS = {
    'rocker_bogie': 'rocker_bogie/urdf/ensamblajeurdf.xacro',
    'differential': 'differential/model/model.sdf',
    'tracked': ('tracked/gazebo_continuous_track_example/urdf_xacro/'
                'two_track_robot_gazebo.urdf.xacro'),
}

# Plugins whose noise comes from the C library's rand() rather than from
# ignition::math::Rand.  Established with `nm -D` on the compiled .so: the
# gzserver seed does NOT reach them, so a non-zero value here silently makes
# the run irreproducible.
UNSEEDED_NOISE_PLUGINS = ['libgazebo_ros_velodyne_gpu_laser.so']


def _expand_xacro(path):
    """Run xacro and return the expanded document as text.

    Needed because the sensors of two of the three platforms are declared in
    xacro files whose $(find ...) references only resolve with the workspace
    sourced.
    """
    import subprocess
    for cmd in (['xacro', path], ['xacro', '--inorder', path]):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            continue
        except OSError:
            break
    raise RuntimeError('xacro failed on {}'.format(path))


def _sensor_noise(root):
    """{(sensor_name, source): stddev} for every noise declaration."""
    out = {}
    for sensor in root.iter('sensor'):
        name = sensor.get('name') or '?'
        for n in sensor.iter('noise'):
            sd = n.find('stddev')
            if sd is None or not (sd.text or '').strip():
                continue          # per-axis wrapper, the real one is nested
            out.setdefault((name, 'sensor <noise>'), []).append(sd.text.strip())
        for p in sensor.iter('plugin'):
            g = p.find('gaussianNoise')
            if g is None or not (g.text or '').strip():
                continue
            out[(name, p.get('filename') or 'plugin')] = [g.text.strip()]
    return {k: sorted(v) for k, v in out.items()}


def check_sensor_noise(src, cfg, rep):
    rep.section('7. Sensor noise identical across platforms and seeded')
    per_robot = {}
    for robot, rel in DESCRIPTION_PATHS.items():
        path = os.path.join(src, rel)
        if not os.path.isfile(path):
            rep.fail('{}: description not found at {}'.format(robot, rel))
            continue
        try:
            if path.endswith('.xacro'):
                root = ET.fromstring(_expand_xacro(path))
            else:
                root = ET.parse(path).getroot()
        except Exception as exc:                             # noqa: BLE001
            rep.fail('{}: cannot parse ({}); source devel/setup.bash if this '
                     'is a xacro'.format(robot, exc))
            continue
        per_robot[robot] = _sensor_noise(root)

    if len(per_robot) < 2:
        rep.fail('fewer than two descriptions parsed, cannot compare')
        return

    # (a) the three must declare the same noise
    keys = sorted(per_robot)
    ref = per_robot[keys[0]]
    for robot in keys[1:]:
        if per_robot[robot] == ref:
            rep.ok('{}: noise declarations match {}'.format(robot, keys[0]))
        else:
            only_ref = {k: v for k, v in ref.items()
                        if per_robot[robot].get(k) != v}
            rep.fail('{}: noise differs from {} at {}'
                     .format(robot, keys[0], sorted(only_ref)))

    # (b) no non-zero noise may come from an unseeded generator
    for robot, decls in per_robot.items():
        for (sensor, source), vals in sorted(decls.items()):
            if not any(p in source for p in UNSEEDED_NOISE_PLUGINS):
                continue
            if any(float(v) != 0.0 for v in vals):
                rep.fail('{}/{}: {} declares stddev {} but its generator is '
                         'not covered by the gzserver seed - runs would not be '
                         'reproducible'.format(robot, sensor, source, vals))
            else:
                rep.ok('{}/{}: unseeded plugin noise is 0.0'
                       .format(robot, sensor))

    # (c) the declared LiDAR figure must be the one sim_config reports
    exp = cfg.get('sensor_noise', {}).get('lidar_range_stddev_m')
    if exp is None:
        rep.warn('sim_config.yaml has no sensor_noise block to check against')
        return
    for robot, decls in sorted(per_robot.items()):
        got = [v for (s, src_), vals in decls.items()
               if 'velodyne' in s.lower() and src_ == 'sensor <noise>'
               for v in vals]
        if not got:
            rep.warn('{}: no LiDAR <noise> block found'.format(robot))
        elif all(approx(v, exp) for v in got):
            rep.ok('{}: LiDAR stddev = {} m'.format(robot, got[0]))
        else:
            rep.fail('{}: LiDAR stddev {} but sim_config says {}'
                     .format(robot, got, exp))


def check_spawn(src, cfg, rep):
    rep.section('6. Spawn pose consistent with sim_config.yaml')
    common = cfg['spawn']['common']
    for robot, rel in LAUNCH_PATHS.items():
        path = os.path.join(src, rel)
        if not os.path.isfile(path):
            rep.fail('{}: launch file missing ({})'.format(robot, rel))
            continue
        text = open(path).read()
        want = cfg['spawn'][robot]
        found_x = ('-x {}'.format(common['x']) in text
                   or 'default="{}"'.format(common['x']) in text)
        found_y = ('-y {}'.format(common['y']) in text
                   or 'default="{}"'.format(common['y']) in text)
        if found_x and found_y:
            rep.ok('{}: spawns at x={} y={} z={}'.format(
                robot, common['x'], common['y'], want['z']))
        else:
            rep.fail('{}: could not confirm spawn x={} y={} in {}'.format(
                robot, common['x'], common['y'], os.path.basename(rel)))

    if 'model://' in open(os.path.join(src, WORLD_PATHS['rocker_bogie'])).read():
        # Only terrain includes should remain; flag robot includes.
        world_text = open(os.path.join(src, WORLD_PATHS['rocker_bogie'])).read()
        for robot in ['differential', 'two_track_robot', 'rocker_bogie']:
            if 'model://{}<'.format(robot) in world_text.replace(' ', ''):
                rep.fail('the world still instantiates robot "{}" - it will '
                         'differ between packages again'.format(robot))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_src = os.path.abspath(os.path.join(here, '..', '..'))
    default_cfg = os.path.join(here, '..', 'config', 'sim_config.yaml')

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', default=default_src,
                        help='catkin src directory (default: inferred)')
    parser.add_argument('--config', default=default_cfg,
                        help='path to sim_config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rep = Report()
    print('=' * 72)
    print('SIMULATION PARITY CHECK')
    print('=' * 72)
    print('src    : {}'.format(args.src))
    print('config : {}'.format(os.path.abspath(args.config)))

    check_worlds_identical(args.src, rep)

    ref_world = os.path.join(args.src, WORLD_PATHS['rocker_bogie'])
    if os.path.isfile(ref_world):
        root = ET.parse(ref_world).getroot()
        check_physics(root, cfg, rep)
        check_step_obstacle(root, cfg, rep)
    else:
        rep.fail('reference world not found, skipping content checks')

    check_terrain_friction(args.src, cfg, rep)
    check_terrain_shared(args.src, rep)
    check_spawn(args.src, cfg, rep)
    check_sensor_noise(args.src, cfg, rep)

    rep.dump()
    print('')
    print('=' * 72)
    if rep.failed:
        print('RESULT: FAIL - do not use this configuration for paper results')
        print('=' * 72)
        return 1
    print('RESULT: PASS - the three platforms share world, physics and terrain')
    print('=' * 72)
    return 0


if __name__ == '__main__':
    sys.exit(main())
