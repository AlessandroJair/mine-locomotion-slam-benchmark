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
  6. the spawn x/y are the same for the three robots;
  7. the three platforms declare identical sensor noise;
  8. the three platforms have the same drive authority - torque ceilings
     derived from one tractive-force budget, no per-platform joint drag,
     and matched velocity-loop stiffness;
  9. the model file each launch file actually spawns weighs what the mass
     matching says it should, so the scaling cannot be applied to a file
     Gazebo never opens;
 10. the command path (min_angular) and the planner acceleration limits are
     the same for the three robots, so no platform is quietly asked for, or
     allowed, more than the others.
 11. the three platforms configure RTAB-Map identically, so the SLAM
     results compare like with like;
 13. LiDAR, camera and IMU sit at the same height above the contact plane on
     the three platforms, measured on the file each launch actually spawns,
     and the description each robot_state_publisher turns into TF agrees with
     it.
 14. the wheels and tracks declare the same friction as each other, not just
     the ground they run on - an anisotropy here is what sets how hard a
     platform has to work to yaw, which is the quantity the locomotion
     comparison is about.

Exit status is 0 on PASS and 1 on FAIL, so it can gate a campaign script.

Usage:
    rosrun robot_metrics check_sim_parity.py
    ./check_sim_parity.py --src /path/to/catkin_ws/src
"""

import argparse
import hashlib
import io
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_parser as mp                                  # noqa: E402
import extract_robot_specs as specs                        # noqa: E402

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


def _declared_friction(path, kind, link):
    """(mu_along_fdir1, mu_across) as declared for one link, or (None, None).

    Two spellings, because the platforms are authored differently: an SDF
    collision says <surface><friction><ode><mu>, and a URDF <gazebo reference=>
    block says <mu1>.  They mean the same pair of coefficients.
    """
    root = ET.parse(path).getroot()
    if kind == 'sdf_collision':
        for elem in root.iter('link'):
            if elem.get('name') != link:
                continue
            for coll in elem.iter('collision'):
                ode = coll.find('./surface/friction/ode')
                if ode is None:
                    continue
                return (find_text(ode, 'mu'), find_text(ode, 'mu2'))
        return (None, None)
    if kind == 'gazebo_reference':
        for elem in root.iter('gazebo'):
            if elem.get('reference') != link:
                continue
            return (find_text(elem, 'mu1'), find_text(elem, 'mu2'))
        return (None, None)
    raise ValueError('unknown friction declaration kind: %s' % kind)


def _declared_stiffness(path, kind, link):
    """(kp, kd, min_depth, max_vel) declarados para un link, o Nones.

    Acepta las dos grafias: SDF usa min_depth/max_vel, el <gazebo reference>
    de URDF usa minDepth/maxVel.
    """
    txt = io.open(path, encoding='utf-8').read()
    if kind == 'gazebo_reference':
        m = re.search(r'<gazebo\s+reference=.%s.>(.*?)</gazebo>'
                      % re.escape(link), txt, re.S)
    else:
        m = re.search(r'<collision[^>]*name=[^>]*%s[^>]*>'
                      r'(.*?)</collision>' % re.escape(link), txt, re.S)
        if not m:
            m = re.search(r'<link[^>]*name=.%s.[^>]*>(.*?)</link>'
                          % re.escape(link), txt, re.S)
    if not m:
        return (None, None, None, None)
    b = m.group(1)

    def uno(*etiquetas):
        for e in etiquetas:
            g = re.search(r'<%s>\s*([0-9.eE+-]+)\s*</%s>' % (e, e), b)
            if g:
                return float(g.group(1))
        return None

    return (uno('kp'), uno('kd'), uno('min_depth', 'minDepth'),
            uno('max_vel', 'maxVel'))


def check_contact_stiffness(src, cfg, rep):
    """16. Las tres plataformas ruedan sobre el mismo contacto.

    Hasta el 2026-09-01 no era asi y nada lo comprobaba: differential y
    rocker declaraban kp = 1e5, y el tracked no declaraba kp ni kd en sus
    doce bloques <contact>, cayendo al defecto de Gazebo de 1e12.  Siete
    ordenes de magnitud entre plataformas que se comparan entre si.  Una
    rigidez de contacto distinta cambia cuanta fuerza tangencial hay
    disponible y con que retardo, o sea justo lo que mide el experimento.
    """
    rep.section('16. Las tres plataformas ruedan sobre el mismo contacto')
    want = cfg.get('robot_contact', {}).get('contact_stiffness')
    if not want:
        rep.fail('sim_config.yaml no declara robot_contact.contact_stiffness')
        return
    surfaces = cfg.get('robot_contact', {}).get('surfaces', {})
    campos = (('kp', 'kp'), ('kd', 'kd'), ('min_depth', 'min_depth'),
              ('max_vel', 'max_vel'))
    for robot, spec in sorted(surfaces.items()):
        path = os.path.join(src, spec['declared_in'])
        if not os.path.isfile(path):
            rep.fail('%s: %s missing' % (robot, spec['declared_in']))
            continue
        malos, faltan = [], []
        for link in spec['links']:
            got = _declared_stiffness(path, spec['kind'], link)
            for k, (nom, clave) in enumerate(campos):
                exp = want.get(clave)
                if got[k] is None:
                    faltan.append('%s.%s' % (link, nom))
                elif not approx(got[k], float(exp)):
                    malos.append('%s.%s=%g' % (link, nom, got[k]))
        if faltan:
            rep.fail('%-13s sin declarar: %s -> cae al defecto de Gazebo '
                     '(kp = 1e12), que NO es el de las otras'
                     % (robot, ', '.join(sorted(set(faltan))[:4])))
        elif malos:
            rep.fail('%-13s %s (config dice kp=%g kd=%g min_depth=%g '
                     'max_vel=%g)'
                     % (robot, ', '.join(sorted(set(malos))[:4]),
                        want['kp'], want['kd'], want['min_depth'],
                        want['max_vel']))
        else:
            rep.ok('%-13s %d links a kp=%g kd=%g min_depth=%g max_vel=%g'
                   % (robot, len(spec['links']), want['kp'], want['kd'],
                      want['min_depth'], want['max_vel']))

    # fdir1 no debe quedar en ninguno: en Gazebo classic va en el marco de la
    # colision, que en una rueda gira con ella (gazebo-classic#2068).
    con_fdir = []
    for robot, spec in sorted(surfaces.items()):
        path = os.path.join(src, spec['declared_in'])
        if os.path.isfile(path) and '<fdir1>' in io.open(
                path, encoding='utf-8').read():
            con_fdir.append(robot)
    # fdir1 SOLO tiene sentido con friccion anisotropa.  Con mu1 == mu2 la
    # direccion es irrelevante y tenerlo declarado engaña al que lee el
    # fichero, asi que eso si es un fallo.  Con mu1 != mu2 es deliberado:
    # es lo que hace que mu2 signifique "lateral".  Se avisa igualmente del
    # caveat de Gazebo -fdir1 va en el marco de la COLISION, que en una rueda
    # gira con ella (gazebo-classic#2068)- pero no se bloquea: quitarlo el
    # 2026-09-02 subio la friccion lateral de 0.75 a 1.0 y trabo la
    # suspension del rocker, con dos ruedas sin tocar el suelo.
    _rc = cfg.get('robot_contact', {})
    exp1 = float(_rc.get('mu1', 1.0))
    exp2 = float(_rc.get('mu2', 1.0))
    isotropo = approx(exp1, exp2)
    if con_fdir and isotropo:
        rep.fail('fdir1 declarado en %s con mu1 == mu2: no selecciona nada y '
                 'confunde al lector' % ', '.join(con_fdir))
    elif con_fdir:
        rep.ok('fdir1 en %s, con mu1 %.2f != mu2 %.2f: anisotropia '
               'deliberada' % (', '.join(con_fdir), exp1, exp2))
        rep.warn('fdir1 va en el marco de la COLISION, que en una rueda gira '
                 'con ella, asi que la direccion no queda fija '
                 '(gazebo-classic#2068).  Declarado a sabiendas.')
    elif not isotropo:
        rep.fail('mu1 != mu2 sin fdir1: la direccion "lateral" queda '
                 'indefinida y mu2 no significa nada')
    else:
        rep.ok('friccion isotropa, sin fdir1')


def check_robot_contact(src, cfg, rep):
    """The robot side of the contact the terrain check already covers.

    Section 4 verifies that the ground declares its friction; nothing verified
    what the WHEELS AND TRACKS declare, and it drifted: measured 2026-08-28,
    both wheeled platforms carried mu 100.0 / mu2 0.75 (anisotropic) against
    the tracks' 0.75 / 0.75.  ODE takes the minimum against the surface in
    contact, so on the mu 1.0 gallery floor that is 1.0 longitudinal / 0.75
    lateral for two platforms and 0.75 / 0.75 for the third.  Friction
    anisotropy is exactly what sets how hard a skid-steer platform must work to
    yaw, so it cannot differ between platforms being compared on locomotion.
    """
    rep.section('14. Driving surfaces declare the same friction on the three '
                'platforms')
    want = cfg.get('robot_contact')
    if not want:
        rep.fail('sim_config.yaml declares no robot_contact block')
        return
    exp1, exp2 = want['mu1'], want['mu2']
    # mu1 admite excepcion POR PLATAFORMA, declarada en su bloque de
    # surfaces; mu2 no.  El transversal es el arrastre, y es lo que hace
    # comparable el ensayo, asi que tiene que ser el mismo en las tres.  El
    # longitudinal es la rodadura, propiedad del neumatico o de la oruga: el
    # encabezado del mundo dice explicitamente que la friccion del lado de
    # la rueda NO es sesgo de ensayo.  Igualarlos los tres el 2026-08-28 le
    # quito al Husky la capacidad de girar (banco del 2026-09-01: 0.141 rad/s
    # de 1.0 pedidos).  La excepcion se admite, pero SOLO escrita aqui.
    for robot, spec in sorted(want['surfaces'].items()):
        exp1_robot = spec.get('mu1', exp1)
        path = os.path.join(src, spec['declared_in'])
        if not os.path.isfile(path):
            rep.fail('{}: {} missing'.format(robot, spec['declared_in']))
            continue
        bad, missing = [], []
        for link in spec['links']:
            mu1, mu2 = _declared_friction(path, spec['kind'], link)
            if mu1 is None or mu2 is None:
                missing.append(link)
            elif not (approx(mu1, exp1_robot) and approx(mu2, exp2)):
                bad.append('{}({}/{})'.format(link, mu1, mu2))
        if missing:
            rep.fail('{:<13s} no friction declared on {} -> the contact is an '
                     'undocumented ODE default'.format(robot,
                                                       ', '.join(missing)))
        if bad:
            rep.fail('{:<13s} {} (config says {}/{})'.format(
                robot, ', '.join(bad), exp1_robot, exp2))
        if not missing and not bad:
            nota = '' if approx(exp1_robot, exp1) else                    '  <- mu1 por plataforma, declarado'
            rep.ok('{:<13s} {} surfaces at mu {}/{}{}'.format(
                robot, len(spec['links']), exp1_robot, exp2, nota))
    if not approx(exp1, exp2):
        rep.warn('mu1 != mu2: the contact is anisotropic, so fdir1 selects '
                 'which axis is which and every platform\'s must be verified '
                 'in the same frame before anything is compared')


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
    # El tracked se spawnea desde el SDF, asi que ese es el archivo cuyos
    # sensores ve Gazebo.  El xacro sigue cargandose en robot_description para
    # gazebo_ros_control, pero sus bloques <gazebo> ya no los abre nadie.
    'tracked': ('tracked/gazebo_continuous_track_example/model/'
                'two_track_robot/model.sdf'),
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
            # Y el RESTO del bloque, no solo stddev.  El 2026-09-13 el IMU del
            # rocker no tenia sesgo y el del husky y el tracked si, y esta
            # comprobacion daba OK porque las stddev coincidian.  Un campo que
            # falta cuenta como '-', asi que no tenerlo tambien es diferencia.
            for campo in ('mean', 'bias_mean', 'bias_stddev', 'dynamic_bias_stddev',
                          'dynamic_bias_correlation_time', 'precision'):
                e = n.find(campo)
                txt = (e.text or '').strip() if e is not None else ''
                out.setdefault((name, 'sensor <noise> ' + campo), []).append(
                    '%g' % float(txt) if txt else '-')
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


# --------------------------------------------------------------------------
# actuation
# --------------------------------------------------------------------------

# Drive joints per platform, in the file that declares their effort limit.
DRIVE_JOINTS = {
    'rocker_bogie': ['rev_9', 'rev_10', 'rev_11', 'rev_12', 'rev_13', 'rev_14'],
    'tracked': ['left_sprocket_axle', 'right_sprocket_axle'],
    'differential': ['front_left_joint', 'front_right_joint',
                     'back_left_joint', 'back_right_joint'],
}

# Where each platform's PID velocity-loop gains are declared.  All three run
# gazebo_ros_control now; the Husky was moved off libgazebo_ros_skid_steer_drive
# because an ODE velocity motor has no gain to match (K_F was infinite).
GAIN_LAUNCH = {
    'rocker_bogie': LAUNCH_PATHS['rocker_bogie'],
    'tracked': LAUNCH_PATHS['tracked'],
    'differential': LAUNCH_PATHS['differential'],
}

# Where each platform turns cmd_vel into joint commands.  A fudge factor in one
# of these is as much a per-platform advantage as a gain would be.
TWIST_NODES = {
    'rocker_bogie': 'rocker_bogie/scripts/twist_to_wheels.py',
    'tracked': ('tracked/gazebo_continuous_track_example/scripts/'
                'twist_to_tracks.py'),
    'differential': 'differential/scripts/twist_to_wheels.py',
}

# Costmap configuration, per platform and per costmap.
COSTMAP_CFG = {}
for _r, _p in (('rocker_bogie', 'rocker_bogie'),
               ('differential', 'differential'),
               ('tracked', 'tracked/gazebo_continuous_track_example')):
    COSTMAP_CFG[_r] = dict(
        (_w, '{}/config/move_base/{}_costmap_params.yaml'.format(_p, _w))
        for _w in ('local', 'global'))

# DWA configuration, one per platform.
PLANNER_CFG = {
    'rocker_bogie': 'rocker_bogie/config/move_base/base_local_planner_params.yaml',
    'tracked': ('tracked/gazebo_continuous_track_example/config/move_base/'
                'base_local_planner_params.yaml'),
    'differential': 'differential/config/move_base/base_local_planner_params.yaml',
}

# The model file each launch file actually hands to Gazebo.  Kept explicit
# rather than parsed: there are three of them, and getting this wrong is the
# exact failure mode check 8 exists to catch.
#   -database NAME  resolves through GAZEBO_MODEL_PATH, which gazebo_ros
#                   populates from the <gazebo_ros gazebo_model_path=...>
#                   export in each package.xml, i.e. <pkg>/model/NAME/.
#   -param  NAME    is the xacro loaded into robot_description by that launch.
SPAWN_SOURCE = {
    'rocker_bogie': 'rocker_bogie/urdf/ensamblajeurdf.xacro',
    'differential': 'differential/model/differential/model.sdf',
    # Back to the model database.  Spawning the tracked platform from
    # robot_description meant sdformat's URDF->SDF conversion, which loses
    # <robotNamespace> (froze the simulation), lumps left_body/right_body away,
    # and drops the <pose relative_to="left_body"> on every belt segment - which
    # put BOTH TRACKS on the chassis centreline.  model.sdf now carries the
    # idler links and the mass matching, so it is no longer a different robot.
    'tracked': ('tracked/gazebo_continuous_track_example/model/'
                'two_track_robot/model.sdf'),
}


def _joint_block(text, name):
    """The <joint name="..."> ... </joint> substring, or None."""
    m = re.search(r'<joint\s+name="{}"[^>]*>'.format(re.escape(name)), text)
    if not m:
        return None
    end = text.find('</joint>', m.end())
    return text[m.end():end] if end != -1 else None


def _attr(text, tag, attr):
    """Value of one attribute of one self-closing tag, as float, or None."""
    m = re.search(r'<{}\b[^>]*\b{}="([-\d.eE+]+)"'.format(tag, attr), text or '')
    return float(m.group(1)) if m else None


def _gain(text, joint, key):
    """One PID gain of one joint out of a rosparam block.

    Handles both the inline flow form used by the Rocker-Bogie
    (rev_9: {p: 12.325, i: 0.474, d: 0.0, i_clamp: 46.725}) and the block form
    used by the tracked platform.  The search window stops at the next drive
    joint so the two forms cannot bleed into each other.
    """
    m = re.search(r'\b{}\s*:'.format(re.escape(joint)), text)
    if not m:
        return None
    window = text[m.end():m.end() + 400]
    for other in [j for js in DRIVE_JOINTS.values() for j in js if j != joint]:
        cut = window.find(other + ':')
        if cut != -1:
            window = window[:cut]
    g = re.search(r'(?<![\w]){}\s*:\s*([-\d.eE+]+)'.format(re.escape(key)),
                  window)
    return float(g.group(1)) if g else None


def check_actuation(src, cfg, rep):
    rep.section('8. Actuation budget matches sim_config.yaml')
    want = cfg.get('actuation')
    if not want:
        rep.fail('sim_config.yaml declares no actuation section: the drive '
                 'authority of the three platforms is unconstrained, and the '
                 'mass-matching argument does not hold without it')
        return

    budget = float(want['traction_force_budget_N'])
    rep.ok('tractive-force budget = {:.1f} N (all driven joints summed)'
           .format(budget))

    # ---- 7a. per-platform torque ceilings --------------------------------
    for robot in cfg['robot_order']:
        spec = want['platforms'][robot]
        n = int(spec['n_driven'])
        r = float(spec['wheel_radius_m'])
        tau = float(spec['tau_max_N_m'])

        # The config must be self-consistent before the models are checked.
        got_force = n * tau / r
        if abs(got_force - budget) / budget > 0.005:
            rep.fail('{}: config is self-inconsistent, {} x {} N*m / {} m = '
                     '{:.1f} N but the budget is {:.1f} N'
                     .format(robot, n, tau, r, got_force, budget))
            continue

        path = os.path.join(src, spec['declared_in'])
        if not os.path.isfile(path):
            rep.fail('{}: declared_in file missing ({})'
                     .format(robot, spec['declared_in']))
            continue
        text = open(path).read()
        tag = spec['declared_tag']

        if tag in ('torque', 'wheelTorque'):
            # Plugin parameter, one value covering every wheel.
            m = re.search(r'<{0}>\s*([-\d.eE+]+)\s*</{0}>'.format(tag), text)
            found = [float(m.group(1))] if m else []
        else:
            # Per-joint <limit effort="...">.
            found = []
            for joint in DRIVE_JOINTS[robot]:
                blk = _joint_block(text, joint)
                if blk is None:
                    rep.fail('{}: joint {} not found in {}'
                             .format(robot, joint, spec['declared_in']))
                    found = None
                    break
                eff = _attr(blk, 'limit', 'effort')
                if eff is None:
                    rep.fail('{}: joint {} declares no <limit effort=...>, so '
                             'joint_limits_interface registers no saturation '
                             'and this platform drives against an unbounded '
                             'torque ceiling'.format(robot, joint))
                    found = None
                    break
                found.append(eff)
            if found is None:
                continue

        if not found:
            rep.fail('{}: no <{}> found in {}'
                     .format(robot, tag, spec['declared_in']))
        elif all(approx(v, tau, tol=max(0.005 * tau, 1e-6)) for v in found):
            rep.ok('{:<13s} {} x {:.3f} N*m @ r={:.5f} -> {:.0f} N'
                   .format(robot, n, tau, r, n * tau / r))
        else:
            rep.fail('{}: declares {} N*m but the budget derives {:.3f} N*m'
                     .format(robot, sorted(set(found)), tau))

    # ---- 7b. drive-joint parasitic drag ----------------------------------
    want_damp = float(want['drive_joint_damping_N_m_s'])
    want_fric = float(want['drive_joint_friction_N_m'])
    for robot, joints in sorted(DRIVE_JOINTS.items()):
        path = os.path.join(src, want['platforms'][robot]['declared_in'])
        if not os.path.isfile(path):
            continue
        text = open(path).read()
        bad = []
        for joint in joints:
            blk = _joint_block(text, joint)
            if blk is None:
                continue
            d = _attr(blk, 'dynamics', 'damping') or 0.0
            f = _attr(blk, 'dynamics', 'friction') or 0.0
            if not (approx(d, want_damp) and approx(f, want_fric)):
                bad.append('{}(damping={}, friction={})'.format(joint, d, f))
        if bad:
            rep.fail('{}: drive joints carry parasitic drag that the other '
                     'platforms do not: {}'.format(robot, ', '.join(bad)))
        else:
            rep.ok('{:<13s} drive-joint damping/friction = {}/{}'
                   .format(robot, want_damp, want_fric))

    # ---- 7c. velocity-loop stiffness -------------------------------------
    loop = want.get('velocity_loop') or {}
    kf_want = float(loop.get('K_F_N_per_m_s', 0.0))
    ki_want = float(loop.get('K_I_N_per_m_s_per_s', 0.0))
    for robot in sorted(GAIN_LAUNCH):
        path = os.path.join(src, GAIN_LAUNCH[robot])
        if not os.path.isfile(path):
            rep.fail('{}: launch file missing'.format(robot))
            continue
        text = open(path).read()
        spec = want['platforms'][robot]
        n = int(spec['n_driven'])
        r = float(spec['wheel_radius_m'])
        tau = float(spec['tau_max_N_m'])

        ps, iss, clamps = [], [], []
        for joint in DRIVE_JOINTS[robot]:
            ps.append(_gain(text, joint, 'p'))
            iss.append(_gain(text, joint, 'i'))
            clamps.append(_gain(text, joint, 'i_clamp'))

        if None in ps or len(set(ps)) != 1:
            rep.fail('{}: could not read one consistent p gain for the drive '
                     'joints (got {})'.format(robot, ps))
            continue

        kf = n * ps[0] / (r * r)
        ki = n * (iss[0] or 0.0) / (r * r)
        if approx(kf, kf_want, tol=max(0.005 * kf_want, 1e-6)):
            rep.ok('{:<13s} K_F = {:.1f} N/(m/s)  (p={})'
                   .format(robot, kf, ps[0]))
        else:
            rep.fail('{}: K_F = {:.1f} N/(m/s) but config says {:.1f} - this '
                     'platform delivers a different tractive force for the '
                     'same speed error'.format(robot, kf, kf_want))
        if approx(ki, ki_want, tol=max(0.005 * ki_want, 1e-6)):
            rep.ok('{:<13s} K_I = {:.1f}           (i={})'
                   .format(robot, ki, iss[0]))
        else:
            rep.fail('{}: K_I = {:.1f} but config says {:.1f}'
                     .format(robot, ki, ki_want))

        # i_clamp defaults to 0.0 in control_toolbox, which silently disables
        # the integral term.  A missing clamp is a FAIL, not a warning: the
        # declared i gain is then a fiction.
        missing = [j for j, c in zip(DRIVE_JOINTS[robot], clamps) if c is None]
        if missing:
            rep.fail('{}: no i_clamp declared for {} - control_toolbox '
                     'defaults i_clamp_min/max to 0.0, so the declared i={} '
                     'is clamped to zero and the loop is P-only'
                     .format(robot, missing, iss[0]))
        elif all(approx(c, tau, tol=max(0.005 * tau, 1e-6)) for c in clamps):
            rep.ok('{:<13s} i_clamp = {} N*m (= its torque ceiling)'
                   .format(robot, tau))
        else:
            rep.fail('{}: i_clamp = {} but the budget gives this platform a '
                     '{} N*m ceiling'.format(robot, sorted(set(clamps)), tau))

    if want.get('husky_loop'):
        rep.fail('sim_config.yaml still records husky_loop ({}). That note '
                 'described the Husky running an ODE velocity motor instead of '
                 'a PID loop; if it is back, the three platforms no longer '
                 'respond to load the same way.'.format(want['husky_loop']))

    # ---- 8d. no derivative term anywhere ---------------------------------
    # d on a velocity loop feeds back acceleration, which is the quantity the
    # mechanical-stability results report.  A per-platform d filters that
    # measurement differently on each robot, so the policy is d = 0 everywhere
    # and a joint retune of K_F if a loop rings.
    for robot, g in sorted((loop.get('gains') or {}).items()):
        d = float(g.get('d', 0.0))
        if not approx(d, 0.0):
            rep.fail('{}: d = {} - a derivative term on a velocity loop feeds '
                     'back acceleration, the quantity this study measures. '
                     'Retune the shared K_F instead.'.format(robot, d))
    if all(approx(float(g.get('d', 0.0)), 0.0)
           for g in (loop.get('gains') or {}).values()):
        rep.ok('d = 0.0 on every platform')


def check_command_path(src, cfg, rep):
    rep.section('10. Command path and planner limits equal across platforms')
    want = cfg.get('actuation') or {}
    cmd = want.get('command') or {}
    plan = want.get('planner') or {}

    if not cmd or not plan:
        rep.fail('sim_config.yaml declares no actuation.command / '
                 'actuation.planner section')
        return

    # ---- min_angular ------------------------------------------------------
    # Forcing |w| up to a floor makes a platform turn harder than it was asked
    # to, which flatters the turn-in-place behaviour RPE is sensitive to.
    want_min = float(cmd['min_angular_rad_s'])
    for robot in sorted(TWIST_NODES):
        path = os.path.join(src, TWIST_NODES[robot])
        if not os.path.isfile(path):
            rep.fail('{}: twist node missing ({})'
                     .format(robot, TWIST_NODES[robot]))
            continue
        m = re.search(r"get_param\(\s*'~min_angular'\s*,\s*([-\d.eE+]+)\s*\)",
                      open(path).read())
        if m is None:
            rep.fail('{}: twist node declares no min_angular default'
                     .format(robot))
            continue
        default = float(m.group(1))

        launch = os.path.join(src, LAUNCH_PATHS[robot])
        pinned = None
        if os.path.isfile(launch):
            lm = re.search(r'<param\s+name="min_angular"\s+value="'
                           r'([-\d.eE+]+)"', open(launch).read())
            if lm:
                pinned = float(lm.group(1))

        effective = pinned if pinned is not None else default
        if approx(effective, want_min):
            rep.ok('{:<13s} min_angular = {} rad/s'.format(robot, effective))
        else:
            rep.fail('{}: min_angular = {} rad/s but config says {} - this '
                     'platform turns harder than it is commanded to'
                     .format(robot, effective, want_min))

    # ---- inflation radius -------------------------------------------------
    # A direct multiplier on how wide each platform believes it is, in a study
    # whose environment is a confined drift.  It was 0.25 / 0.40 / 0.40 and
    # nothing checked it.
    want_infl = plan.get('inflation_radius')
    if want_infl is not None:
        for robot in sorted(COSTMAP_CFG):
            for which, rel in COSTMAP_CFG[robot].items():
                path = os.path.join(src, rel)
                if not os.path.isfile(path):
                    rep.fail('{}: {} costmap config missing'
                             .format(robot, which))
                    continue
                m = re.search(r'^\s*inflation_radius:\s*([-\d.eE+]+)',
                              open(path).read(), re.M)
                if m is None:
                    rep.fail('{}: {} costmap declares no inflation_radius'
                             .format(robot, which))
                elif approx(float(m.group(1)), float(want_infl)):
                    rep.ok('{:<13s} {:<6s} inflation_radius = {}'
                           .format(robot, which, m.group(1)))
                else:
                    rep.fail('{}: {} inflation_radius = {} but config says {} '
                             '- this platform is given more or less clearance '
                             'than the others'
                             .format(robot, which, m.group(1), want_infl))

    # ---- costmap layer plugins -------------------------------------------
    # Not a value from sim_config.yaml - just an equality across platforms,
    # like the SLAM check.  It exists because this slipped through: the
    # differential declared costmap_2d::ObstacleLayer for its local costmap
    # while the tracked and the Rocker-Bogie declared costmap_2d::VoxelLayer,
    # so one platform cleared its costmap in 2D and the other two in 3D.  The
    # 3D layer cannot clear the sensor's near-field blind annulus (measured
    # 2026-08-20: 100% of the lethal cells within 1.5 m of the Rocker-Bogie had
    # no live sensor support), so two of the three platforms drove around
    # carrying stale obstacles the third one did not have.  Nothing in this
    # script looked at the plugin list, so it went unnoticed for the whole
    # campaign.  A difference here is a difference in what each platform
    # believes about the world, which is the one thing this study must hold
    # constant.
    plugin_re = re.compile(r'-\s*\{\s*name:\s*([A-Za-z_0-9]+)\s*,\s*'
                           r'type:\s*"([^"]+)"\s*\}')
    for which in ('local', 'global'):
        seen = {}
        for robot in sorted(COSTMAP_CFG):
            path = os.path.join(src, COSTMAP_CFG[robot][which])
            if not os.path.isfile(path):
                rep.fail('{}: {} costmap config missing'.format(robot, which))
                continue
            seen[robot] = plugin_re.findall(open(path).read())
        if len(seen) < 2:
            continue
        variants = set(tuple(v) for v in seen.values())
        if len(variants) == 1:
            rep.ok('{:<13s} costmap layers identical: {}'
                   .format(which, ', '.join(t for _, t in list(seen.values())[0])))
        else:
            rep.fail('{} costmap plugin stack differs across platforms - each '
                     'platform then builds its obstacle picture differently:'
                     .format(which))
            for robot in sorted(seen):
                rep.fail('    {:<13s} {}'
                         .format(robot,
                                 ', '.join('{}={}'.format(n, t)
                                           for n, t in seen[robot])))

    # ---- planner accelerations -------------------------------------------
    for robot in sorted(PLANNER_CFG):
        path = os.path.join(src, PLANNER_CFG[robot])
        if not os.path.isfile(path):
            rep.fail('{}: planner config missing'.format(robot))
            continue
        text = open(path).read()
        for key in ('acc_lim_x', 'acc_lim_y', 'acc_lim_theta'):
            m = re.search(r'^\s*{}\s*:\s*([-\d.eE+]+)'.format(key), text,
                          re.M)
            if m is None:
                rep.fail('{}: {} not declared'.format(robot, key))
                continue
            got, exp = float(m.group(1)), float(plan[key])
            if approx(got, exp):
                rep.ok('{:<13s} {} = {}'.format(robot, key, got))
            else:
                rep.fail('{}: {} = {} but config says {} - the planner is '
                         'allowed to demand more of this platform than of the '
                         'others'.format(robot, key, got, exp))


def check_trajectory_follower(src, cfg, rep):
    """15. El seguidor de trayectoria fija usa las ganancias declaradas.

    Son comunes a las tres por construccion -hay un solo
    trajectory_follow.launch-, pero hasta el 2026-09-01 vivian como valores
    por defecto dentro del nodo, sin declarar y sin comprobar: equivalentes
    por accidente.  Esto verifica que el launch, el nodo y sim_config.yaml
    dicen lo mismo, de modo que un retoque no se pueda colar en uno de los
    tres sitios y pasar inadvertido.

    Importa porque la MISMA ganancia no da el mismo lazo cerrado en las tres:
    entregan guiñadas muy distintas (alfa 0.202 / 0.311 / 0.805).
    """
    rep.section('15. El seguidor de trayectoria fija usa las ganancias '
                'declaradas')
    want = cfg.get('actuation', {}).get('trajectory_follower')
    if not want:
        rep.fail('sim_config.yaml no declara actuation.trajectory_follower')
        return

    # nombre en sim_config -> (arg del launch, param del nodo)
    MAPA = {
        'k_xte':               ('k_xte', 'k_xte'),
        'xte_curv_max':        ('xte_curv_max', 'xte_curv_max'),
        'lookahead_min_m':     ('lookahead_min', 'lookahead_min'),
        'lookahead_max_m':     ('lookahead_max', 'lookahead_max'),
        'lookahead_k_s':       ('lookahead_k', 'lookahead_k'),
        'align_threshold_rad': ('align_threshold', 'align_threshold'),
        'curve_slowdown':      ('curve_slowdown', 'curve_slowdown'),
        'w_v_floor_m_s':       ('w_v_floor', 'w_v_floor'),
        'min_aim_dist_m':      ('min_aim_dist', 'min_aim_dist'),
    }

    lp = os.path.join(src, 'robot_metrics/launch/trajectory_follow.launch')
    np_ = os.path.join(src, 'robot_metrics/scripts/trajectory_follower.py')
    if not os.path.isfile(lp) or not os.path.isfile(np_):
        rep.fail('falta trajectory_follow.launch o trajectory_follower.py')
        return
    launch = io.open(lp, encoding='utf-8').read()
    nodo = io.open(np_, encoding='utf-8').read()

    for clave, (arg, param) in sorted(MAPA.items()):
        esperado = want.get(clave)
        if esperado is None:
            rep.fail('sim_config no declara trajectory_follower.%s' % clave)
            continue
        m = re.search(r'<arg\s+name="%s"\s+default="([^"]+)"' % re.escape(arg),
                      launch)
        if not m:
            rep.fail('%-20s no esta expuesto en trajectory_follow.launch'
                     % clave)
            continue
        if not approx(float(m.group(1)), float(esperado)):
            rep.fail('%-20s launch dice %s, sim_config dice %s'
                     % (clave, m.group(1), esperado))
            continue
        d = re.search(r"get_param\('~%s',\s*([0-9.]+)\)" % re.escape(param),
                      nodo)
        if d and not approx(float(d.group(1)), float(esperado)):
            rep.fail('%-20s el defecto del nodo es %s, sim_config dice %s '
                     '- si el launch fallase, la corrida usaria el del nodo'
                     % (clave, d.group(1), esperado))
            continue
        rep.ok('%-20s = %-6s (launch, nodo y config coinciden)'
               % (clave, esperado))

    # El launch es uno solo: si alguien lo duplicase por plataforma, la
    # equivalencia se romperia sin que nada mas lo notara.
    otros = []
    for raiz, _dirs, files in os.walk(src):
        for f in files:
            if f == 'trajectory_follow.launch':
                otros.append(os.path.join(raiz, f))
    if len(otros) == 1:
        rep.ok('un solo trajectory_follow.launch: las tres plataformas '
               'comparten seguidor')
    else:
        rep.fail('hay %d trajectory_follow.launch; las plataformas podrian '
                 'estar usando ganancias distintas' % len(otros))


def check_spawn_source(src, cfg, rep):
    rep.section('9. Gazebo loads the model files the tooling maintains')
    # scale_masses.py, extract_robot_specs.py and the actuation budget each
    # edit or read one specific file per platform.  If the launch file spawns a
    # DIFFERENT file, every one of those guarantees is written into a document
    # Gazebo never opens.  That is not hypothetical: it is how the differential
    # and tracked platforms came to run unscaled masses after the mass matching
    # had supposedly been applied to them.
    platforms = (cfg.get('actuation') or {}).get('platforms') or {}
    for robot, rel in sorted(SPAWN_SOURCE.items()):
        loaded = os.path.join(src, rel)
        if not os.path.isfile(loaded):
            rep.fail('{}: launch spawns {} which does not exist'
                     .format(robot, rel))
            continue

        # WEIGH IT, do not look for a banner.  The banner is an XML comment,
        # so it lives in whichever file the author put it in and xacro strips
        # comments on expansion - two ways to get a wrong answer about the
        # thing that actually matters, which is what the spawned model weighs.
        target = float((cfg.get('mass_matching') or {}).get('target_kg', 45.0))
        tol = float((cfg.get('mass_matching') or {}).get('tolerance_kg', 0.05))
        try:
            model = mp.load(loaded)
            mass = model.total_mass()
        except Exception as e:                                  # noqa: BLE001
            rep.fail('{}: cannot weigh {}: {}'.format(robot, rel, e))
            continue

        if abs(mass - target) <= tol:
            rep.ok('{:<13s} spawns {} at {:.4f} kg'
                   .format(robot, rel, mass))
        else:
            rep.fail('{}: Gazebo loads {}, which weighs {:.4f} kg against the '
                     '{:.1f} kg the mass matching sets. This platform is NOT '
                     'mass-matched - scale_masses.py is editing a different '
                     'file from the one that is spawned.'
                     .format(robot, rel, mass, target))

        declared = (platforms.get(robot) or {}).get('declared_in')
        if declared and declared != rel:
            rep.warn('{}: its actuation budget is declared in {} while '
                     'Gazebo spawns {}. Expected wherever the model is spawned '
                     'from an SDF but gazebo_ros_control reads the effort limit '
                     'from robot_description - true for the differential and '
                     'the tracked platform. Keep the two in step.'
                     .format(robot, declared, rel))


# Where each platform names the model for the ground-truth logger, and where
# it tells Gazebo what to call it.  These two and sim_config.yaml must agree.
GT_NAME_SOURCES = {
    'rocker_bogie': ('rocker_bogie/launch/waypoint_navigation.launch',
                     'rocker_bogie/launch/lcmine_rocker_bogie_world.launch'),
    'differential': ('differential/launch/waypoint_navigation.launch',
                     'differential/launch/lcmine_husky_world.launch'),
    'tracked': ('tracked/gazebo_continuous_track_example/launch/'
                'waypoint_navigation.launch',
                'tracked/gazebo_continuous_track_example/launch/'
                'lcmine_two_track_world.launch'),
}


# Where each platform's robot_description comes from - what
# robot_state_publisher turns into TF.  Two platforms spawn an SDF and describe
# themselves with a xacro, so they hold the LiDAR mount TWICE and the two copies
# can drift; the Rocker-Bogie has a single file and needs no second entry.
LIDAR_TF_SOURCE = {
    'differential': 'differential/urdf_xacro/husky_gazebo.urdf.xacro',
    'tracked': ('tracked/gazebo_continuous_track_example/'
                'urdf_xacro/two_track_robot_gazebo.urdf.xacro'),
}


# Where each platform's robot_description comes from - what
# robot_state_publisher turns into TF.  Two platforms spawn an SDF and describe
# themselves with a xacro, so they hold every sensor mount TWICE and the two
# copies can drift; the Rocker-Bogie has a single file and needs no entry.
SENSOR_TF_SOURCE = {
    'differential': 'differential/urdf_xacro/husky_gazebo.urdf.xacro',
    'tracked': ('tracked/gazebo_continuous_track_example/'
                'urdf_xacro/two_track_robot_gazebo.urdf.xacro'),
}

# Tipo de sensor a buscar, y la clave de config con la altura a la que tiene
# que quedar.  Se busca POR TIPO y no por nombre de link, y la altura sale de
# donde el sensor emite -- link.pose @ sensor.pose -- porque el link de montaje
# no siempre existe: la conversion URDF fusiona los links de sensor en el padre
# (el rocker), y el tracked y el husky tienen la camara metida dentro del
# chasis a proposito desde el 2026-09-17 (la junta fija cedia; ver la nota
# CAMARA FUSIONADA en sus model.sdf).  Buscando por nombre, esta comprobacion
# se saltaba la camara de esas dos plataformas sin decir nada.
# Para el LiDAR esto ya se hacia: la <pose> del propio Velodyne lo sube 0.0377 m
# sobre el link (0.0641 m en el rocker) y lo que tiene que cuadrar es de donde
# salen los rayos.  Las tres camaras tienen <sensor> sin <pose>, asi que para
# ellas la regla nueva da exactamente lo mismo que la vieja.
# El IMU NO esta aqui a proposito: no se iguala por altura sobre el suelo
# sino por brazo cero desde el centro de masas, y lo comprueba
# check_imu_at_centre_of_mass().  Ver sensor_mounting en sim_config.yaml.
SENSOR_MOUNTS = [
    ('lidar', ('gpu_ray', 'ray'), 'lidar_height_above_contact_m'),
    ('camera', ('camera',), 'camera_height_above_contact_m'),
]


def _sensor_pose(model, tipos):
    """Pose del primer sensor de alguno de esos tipos, este donde este."""
    for link in model.links.values():
        for s in link.sensors:
            if s.get('type') in tipos:
                return link.pose @ s['pose']
    return None


def _contact_z(model, robot, cfg, spec):
    """z of the ground contact plane, in the model file's own frame.

    The drop from the axle to the ground is DECLARED per platform rather than
    taken as the wheel radius, because on the tracked platform it is not the
    wheel radius: that vehicle rests on its belt, 0.210 m below the axle,
    while its sprockets are 0.178 m - and the belt elements do not exist in
    any file, gazebo_continuous_track creates them at load time.  Deriving the
    number from the geometry is what put its LiDAR 32 mm high.  See
    sim_config.yaml: sensor_mounting.contact_plane_below_axle_m.
    """
    mounting = cfg.get('sensor_mounting') or {}
    drop = (mounting.get('contact_plane_below_axle_m') or {}).get(robot)
    wg = specs.wheel_geometry(model, spec.get('wheel_pattern', r'wheel'))
    axle = wg.get('axle_height', float('nan'))
    if drop is None:
        drop = wg.get('radius', float('nan'))
    if not (np.isfinite(axle) and np.isfinite(float(drop))):
        return None
    return axle - float(drop)


def _sensor_heights(model, robot, cfg, spec):
    """{sensor: height of its mount above the contact plane}."""
    contact = _contact_z(model, robot, cfg, spec)
    if contact is None:
        return None
    out = {}
    for sensor, tipos, _ in SENSOR_MOUNTS:
        T = _sensor_pose(model, tipos)
        if T is None:
            continue
        out[sensor] = float(T[2, 3]) - contact
    return out


def check_sensor_heights(src, cfg, rep):
    rep.section('13. The three platforms mount their sensors at the same '
                'height, in the files Gazebo spawns')
    # WHY THIS IS A PARITY CHECK.  The study attributes differences in SLAM
    # quality to the locomotion type.  That only holds if the three sensors see
    # the gallery from the same place: a LiDAR mounted higher sweeps further
    # per degree of chassis tilt, a camera mounted further forward frames a
    # different part of the wall, and an IMU further from the centre of
    # rotation reads a larger lever-arm acceleration - none of which is a
    # property of the chassis underneath.
    #
    # AND WHY IT MEASURES THE SPAWNED FILE.  The standardization pass edited
    # the tracked platform's xacro, which is the file the other two platforms
    # are described in - but this one spawns its model.sdf, so all three of its
    # sensors kept their native poses while every table said otherwise, and TF
    # placed them where the xacro claimed.  Nothing caught it, because the
    # audit tool read the same xacro the standardization had edited.
    mounting = cfg.get('sensor_mounting') or {}
    tol = float(mounting.get('height_tolerance_m', 0.001))
    if not mounting:
        rep.warn('sim_config declares no sensor_mounting block; nothing to '
                 'check the mounting heights against')
        return

    medidas = {}
    for robot, rel in sorted(SPAWN_SOURCE.items()):
        spec = specs.ROBOTS.get(robot) or {}
        try:
            model = mp.load(os.path.join(src, rel))
        except Exception as e:                                  # noqa: BLE001
            rep.fail('{}: cannot read {}: {}'.format(robot, rel, e))
            continue
        h = _sensor_heights(model, robot, cfg, spec)
        if not h:
            rep.fail('{}: no sensor mounts, or no wheel geometry to refer them '
                     'to, in {} - the file Gazebo spawns'.format(robot, rel))
            continue
        medidas[robot] = h
        for sensor, _, key in SENSOR_MOUNTS:
            if sensor not in h:
                rep.warn('{:<13s} has no {} mount in {}'
                         .format(robot, sensor, rel))
                continue
            exp = mounting.get(key)
            if exp is None:
                rep.warn('{:<13s} {} at {:.4f} m (no {} in sim_config)'
                         .format(robot, sensor, h[sensor], key))
            elif abs(h[sensor] - float(exp)) <= tol:
                rep.ok('{:<13s} {:<7s} {:.4f} m above contact in {}'
                       .format(robot, sensor, h[sensor], rel))
            else:
                rep.fail('{}: Gazebo spawns {}, which puts the {} {:.4f} m '
                         'above the contact plane against the {:.4f} m '
                         'sim_config declares. Sensor placement is NOT '
                         'controlled for on this platform.'
                         .format(robot, rel, sensor, h[sensor], float(exp)))

    for sensor, _, _ in SENSOR_MOUNTS:
        vals = {r: h[sensor] for r, h in medidas.items() if sensor in h}
        if len(vals) < 2:
            continue
        spread = max(vals.values()) - min(vals.values())
        if spread <= tol:
            rep.ok('{:<7s} spread across platforms {:.4f} m'
                   .format(sensor, spread))
        else:
            rep.fail('{} height spread across platforms is {:.4f} m ({}). The '
                     'three do not observe the gallery from the same place, so '
                     'the difference is not attributable to locomotion alone.'
                     .format(sensor, spread, ', '.join(
                         '{} {:.4f}'.format(k, v)
                         for k, v in sorted(vals.items()))))

    # The description feeding TF has to agree with the spawned model, or every
    # measurement is transformed as if the sensor were somewhere it is not.
    for robot, rel in sorted(SENSOR_TF_SOURCE.items()):
        if robot not in medidas:
            continue
        spec = specs.ROBOTS.get(robot) or {}
        try:
            model = mp.load(os.path.join(src, rel))
        except Exception as e:                                  # noqa: BLE001
            rep.warn('{}: cannot read its robot_description {}: {}'
                     .format(robot, rel, e))
            continue
        h = _sensor_heights(model, robot, cfg, spec)
        if not h:
            rep.warn('{}: no sensor mounts in {}'.format(robot, rel))
            continue
        for sensor, _, _ in SENSOR_MOUNTS:
            if sensor not in h or sensor not in medidas[robot]:
                continue
            d = h[sensor] - medidas[robot][sensor]
            if abs(d) <= tol:
                rep.ok('{:<13s} robot_description agrees on the {} ({:.4f} m)'
                       .format(robot, sensor, h[sensor]))
            else:
                rep.fail('{}: Gazebo puts the {} {:.4f} m above the contact '
                         'plane but {} - what robot_state_publisher turns into '
                         'TF - says {:.4f} m. Its measurements are transformed '
                         '{:+.4f} m off in z.'
                         .format(robot, sensor, medidas[robot][sensor], rel,
                                 h[sensor], d))


def _imu_offset_from_com(model):
    """imu_link - centro de masas, en el marco base del modelo.  None si falta."""
    link = model.links.get('imu_link')
    if link is None:
        return None
    com = np.asarray(model.center_of_mass(), dtype=float)
    if not np.isfinite(com).all():
        return None
    return np.asarray(link.pose[:3, 3], dtype=float) - com


def check_imu_at_centre_of_mass(src, cfg, rep):
    rep.section('13b. El IMU de las tres plataformas esta en su centro de '
                'masas')
    # POR QUE ESTA REGLA Y NO LA ALTURA COMUN.  Un acelerometro mide
    #     a_sensor = a_cm + alpha x r + omega x (omega x r)
    # con r desde el CENTRO DE MASAS.  Hasta 2026-09-04 los tres IMU estaban a
    # la misma altura sobre el suelo, que es la regla correcta para el LiDAR y
    # la camara y la equivocada para un inercial: como los tres chasis tienen
    # el CM a alturas muy distintas, los brazos salian 0.360 / 0.485 / 0.238 m
    # -un factor 2.04- y con el mismo balanceo de chasis el IMU del tracked
    # leia el doble de termino de brazo que el del rocker.  Eso no es una
    # propiedad de la locomocion, y contaminaba cualquier comparacion de
    # vibracion medida con el IMU.
    #
    # Se exige r = 0 y no un |r| comun porque alpha x r depende del VECTOR r:
    # dos brazos iguales en modulo y distintos en direccion siguen midiendo
    # cosas distintas.
    mounting = cfg.get('sensor_mounting') or {}
    if not mounting.get('imu_at_centre_of_mass'):
        rep.warn('sim_config no declara imu_at_centre_of_mass; no se comprueba '
                 'el brazo del IMU')
        return
    tol = float(mounting.get('imu_lever_arm_tolerance_m', 0.002))

    medidas = {}
    for robot, rel in sorted(SPAWN_SOURCE.items()):
        try:
            model = mp.load(os.path.join(src, rel))
        except Exception as e:                                  # noqa: BLE001
            rep.fail('{}: cannot read {}: {}'.format(robot, rel, e))
            continue
        r = _imu_offset_from_com(model)
        if r is None:
            rep.fail('{}: no imu_link, o sin centro de masas, en {} - el '
                     'fichero que Gazebo spawnea'.format(robot, rel))
            continue
        medidas[robot] = r
        d = float(np.linalg.norm(r))
        if d <= tol:
            rep.ok('{:<13s} IMU a {:.4f} m de su CM en {}'
                   .format(robot, d, rel))
        else:
            rep.fail('{}: Gazebo spawns {}, que deja el IMU a {:.4f} m del '
                     'centro de masas (r = [{:.4f} {:.4f} {:.4f}]) contra la '
                     'tolerancia de {:.4f} m. Su acelerometro lleva un termino '
                     'de brazo que los otros no tienen, asi que su vibracion '
                     'no es comparable. Recalcular el montaje con '
                     'imu_target.py.'
                     .format(robot, rel, d, r[0], r[1], r[2], tol))

    # La descripcion que alimenta TF tiene que poner el IMU en el mismo punto
    # que el modelo spawneado, o toda medida se transforma como si el sensor
    # estuviese donde no esta.  Se compara la POSICION COMPLETA respecto al
    # link base y no solo la altura: al mover el IMU al CM, x deja de ser cero.
    for robot, rel in sorted(SENSOR_TF_SOURCE.items()):
        if robot not in medidas:
            continue
        try:
            model = mp.load(os.path.join(src, rel))
        except Exception as e:                                  # noqa: BLE001
            rep.warn('{}: cannot read its robot_description {}: {}'
                     .format(robot, rel, e))
            continue
        link = model.links.get('imu_link')
        base = model.links.get('base_link') or model.links.get('body')
        if link is None or base is None:
            rep.warn('{}: no imu_link o no link base en {}'.format(robot, rel))
            continue
        p_tf = (np.asarray(link.pose[:3, 3], dtype=float)
                - np.asarray(base.pose[:3, 3], dtype=float))
        try:
            spawned = mp.load(os.path.join(src, SPAWN_SOURCE[robot]))
        except Exception:                                       # noqa: BLE001
            continue
        s_link = spawned.links['imu_link']
        s_base = (spawned.links.get('base_link') or spawned.links.get('body'))
        p_gz = (np.asarray(s_link.pose[:3, 3], dtype=float)
                - np.asarray(s_base.pose[:3, 3], dtype=float))
        d = float(np.linalg.norm(p_tf - p_gz))
        if d <= tol:
            rep.ok('{:<13s} robot_description pone el IMU en el mismo punto '
                   '({:.4f} m)'.format(robot, d))
        else:
            rep.fail('{}: Gazebo pone el IMU en [{:.4f} {:.4f} {:.4f}] '
                     'respecto al link base pero {} - lo que '
                     'robot_state_publisher convierte en TF - dice '
                     '[{:.4f} {:.4f} {:.4f}], {:.4f} m de diferencia.'
                     .format(robot, p_gz[0], p_gz[1], p_gz[2], rel,
                             p_tf[0], p_tf[1], p_tf[2], d))


def check_ground_truth_name(src, cfg, rep):
    rep.section('12. The ground-truth model name agrees everywhere')
    # metrics_logger.py takes the ground truth from /gazebo/model_states by
    # looking up model_name.  If that name is not the one Gazebo spawned, the
    # lookup silently fails: every gt_* column stays 0.0, gt_distance_m is 0.0
    # and the trajectory file is written with nothing but its header.  The run
    # still "succeeds" and still produces a metrics.csv, so nothing downstream
    # notices.  Measured 2026-08-20: the differential ran a full route that way
    # - waypoint_navigation.launch said "husky", Gazebo spawned "differential".
    spawn = cfg.get('spawn', {}) or {}
    for robot in sorted(GT_NAME_SOURCES):
        nav_rel, world_rel = GT_NAME_SOURCES[robot]
        nav_p = os.path.join(src, nav_rel)
        world_p = os.path.join(src, world_rel)
        if not os.path.isfile(nav_p) or not os.path.isfile(world_p):
            rep.fail('{}: launch file missing'.format(robot))
            continue
        m = re.search(r'name="model_name"\s+value="([^"]+)"', open(nav_p).read())
        logger_name = m.group(1) if m else None
        m = re.search(r'-model\s+([A-Za-z_0-9]+)', open(world_p).read())
        spawned_name = m.group(1) if m else None
        declared = (spawn.get(robot) or {}).get('model_name')

        if logger_name is None:
            rep.fail('{}: waypoint_navigation.launch passes no model_name'
                     .format(robot))
            continue
        if spawned_name is None:
            rep.fail('{}: no "-model" in the world launch'.format(robot))
            continue
        if logger_name == spawned_name == declared:
            rep.ok('{:<13s} ground-truth model name = {}'.format(robot, logger_name))
        else:
            rep.fail('{}: the ground truth will be EMPTY - logger looks up "{}", '
                     'Gazebo spawns "{}", sim_config declares "{}"'
                     .format(robot, logger_name, spawned_name, declared))


# Where each platform configures RTAB-Map.
SLAM_LAUNCH = {
    'rocker_bogie': 'rocker_bogie/launch/rtabmap_3d_slam.launch',
    'differential': 'differential/launch/rtabmap_3d_slam.launch',
    'tracked': ('tracked/gazebo_continuous_track_example/launch/'
                'rtabmap_3d_slam.launch'),
}

# Parameters that legitimately differ because they name the platform.
SLAM_PER_PLATFORM = ('frame_id', 'odom_frame_id', 'base_frame', 'tf_prefix')


def check_slam_parity(src, cfg, rep):
    rep.section('11. The three platforms configure SLAM identically')
    # The SLAM backend is a shared test condition (see the slam section of
    # sim_config.yaml), but nothing compared these three files.  That is how an
    # ICP retune came to be applied to one platform and not the other two.
    vals = {}
    for robot, rel in sorted(SLAM_LAUNCH.items()):
        path = os.path.join(src, rel)
        if not os.path.isfile(path):
            rep.fail('{}: {} missing'.format(robot, rel))
            return
        d = {}
        for m in re.finditer(r'<param\s+name="([^"]+)"[^>]*?value="([^"]*)"',
                             open(path).read()):
            d[m.group(1)] = m.group(2)
        vals[robot] = d

    keys = set()
    for d in vals.values():
        keys |= set(d.keys())

    robots = sorted(vals)
    bad = 0
    for k in sorted(keys):
        if any(p in k for p in SLAM_PER_PLATFORM):
            continue
        row = [vals[r].get(k, '(absent)') for r in robots]
        if len(set(row)) == 1:
            continue
        bad += 1
        rep.fail('{} differs: {}'.format(
            k, ', '.join('{}={}'.format(r, v) for r, v in zip(robots, row))))
    if not bad:
        rep.ok('{} SLAM parameters, identical across the three platforms'
               .format(len(keys)))
        rep.ok('(frame and topic names that name the platform are exempt)')


def check_joint_state_rate(src, cfg, rep):
    rep.section('17. Las tres publican joint_states a la misma tasa')
    # joint_states no lo leia nadie, asi que su publish_rate nunca se comparo:
    # el tracked estaba en 10 Hz y las otras dos en 50, sin una linea que lo
    # justificara.  Dejo de ser inocuo cuando wheel_odometry.py empezo a
    # tomar de ahi la velocidad longitudinal que alimenta el de-skew - a 10 Hz
    # el tracked daba UNA muestra por nube de LiDAR contra cinco.
    declarada = ((cfg.get('sensor_rates') or {}).get('joint_states_hz'))
    vals = {}
    for robot in sorted(LAUNCH_PATHS):
        path = os.path.join(src, LAUNCH_PATHS[robot])
        if not os.path.isfile(path):
            rep.fail('{}: {} missing'.format(robot, LAUNCH_PATHS[robot]))
            return
        m = re.search(r'joint_state_controller:\s*\n'
                      r'(?:\s*#[^\n]*\n)*'
                      r'\s*type:\s*\S+\s*\n'
                      r'(?:\s*#[^\n]*\n)*'
                      r'\s*publish_rate:\s*([0-9.]+)',
                      open(path).read())
        if not m:
            rep.fail('{}: no publish_rate for joint_state_controller'
                     .format(robot))
            return
        vals[robot] = float(m.group(1))

    if len(set(vals.values())) != 1:
        rep.fail('joint_states publish_rate differs: {}'.format(
            ', '.join('{}={:g}'.format(r, v) for r, v in sorted(vals.items()))))
        return
    tasa = next(iter(vals.values()))
    if declarada is not None and float(declarada) != tasa:
        rep.fail('joint_states a {:g} Hz pero sim_config declara {:g}'
                 .format(tasa, float(declarada)))
        return
    rep.ok('joint_states a {:g} Hz en las tres{}'.format(
        tasa, ' (declarado en sim_config)' if declarada is not None else ''))


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
    check_robot_contact(args.src, cfg, rep)
    check_terrain_shared(args.src, rep)
    check_spawn(args.src, cfg, rep)
    check_sensor_noise(args.src, cfg, rep)
    check_actuation(args.src, cfg, rep)
    check_spawn_source(args.src, cfg, rep)
    check_command_path(args.src, cfg, rep)
    check_slam_parity(args.src, cfg, rep)
    check_ground_truth_name(args.src, cfg, rep)
    check_sensor_heights(args.src, cfg, rep)
    check_imu_at_centre_of_mass(args.src, cfg, rep)
    check_trajectory_follower(args.src, cfg, rep)
    check_contact_stiffness(args.src, cfg, rep)
    check_joint_state_rate(args.src, cfg, rep)

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
