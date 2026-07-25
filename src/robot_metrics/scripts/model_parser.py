#!/usr/bin/env python3
"""Read a robot description (URDF, xacro or SDF) into a common structure.

The three platforms in this study are described in three different ways - the
Rocker-Bogie as a xacro-generated URDF, the Husky and the tracked robot as
Gazebo SDF models - so any script that wants to compare them has to normalise
them first.  That normalisation lives here, and extract_robot_specs.py and
extract_sensor_poses.py both build on it.

The important structural difference between the two formats:

  URDF  a link has no pose of its own.  Its placement follows from the chain of
        joint origins between it and the root, so the pose has to be composed
        by forward kinematics.
  SDF   every <link> carries a <pose> that is already expressed in the model
        frame, so no composition is needed.  Joints only declare connectivity.

Both are reduced to the same thing: for each link, a 4x4 pose in the base
frame, its inertial properties, and its collision geometry.

Conventions
-----------
* Everything is returned in metres, kilograms and radians.
* Poses are 4x4 homogeneous matrices in the BASE frame.  The base is the link
  named base_link where one exists, otherwise the kinematic root (URDF) or the
  model origin (SDF).
* Angles in <pose>/<origin> are fixed-axis roll-pitch-yaw, applied Rz*Ry*Rx,
  which is what both formats specify.
"""

import os
import subprocess
import xml.etree.ElementTree as ET

import numpy as np


# ---------------------------------------------------------------------------
# small transform helpers
# ---------------------------------------------------------------------------

def rpy_to_rot(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def rot_to_rpy(R):
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(sy) < 0.999999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:                      # gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def make_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_rot(*rpy)
    T[:3, 3] = xyz
    return T


def parse_xyz(text, default=(0.0, 0.0, 0.0)):
    if not text:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


def parse_pose(text):
    """SDF <pose>x y z roll pitch yaw</pose> -> 4x4."""
    if not text:
        return np.eye(4)
    v = [float(x) for x in text.split()]
    while len(v) < 6:
        v.append(0.0)
    return make_T(v[:3], v[3:6])


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

class Link(object):
    def __init__(self, name):
        self.name = name
        self.mass = 0.0
        self.inertial_origin = np.eye(4)   # in the link frame
        self.inertia = None
        self.collisions = []               # [(T_in_link, geometry dict)]
        self.sensors = []                  # [dict]
        self.pose = np.eye(4)              # in the base frame (filled by FK)


class Joint(object):
    def __init__(self, name, jtype):
        self.name = name
        self.type = jtype
        self.parent = None
        self.child = None
        self.origin = np.eye(4)
        self.axis = np.array([1.0, 0.0, 0.0])
        self.limit = {}


class RobotModel(object):
    def __init__(self, name, source):
        self.name = name
        self.source = source
        self.links = {}
        self.joints = {}
        self.base = 'base_link'
        self.plugins = []
        # Bodies that Gazebo simulates but that are not URDF links: SDF
        # fragments embedded in <gazebo> blocks, which plugins instantiate at
        # load time.  The tracked robot keeps 3 kg of track segments there, so
        # ignoring them under-reports its mass by 6.4%.  Each entry is
        # (mass_kg, note).
        self.embedded_masses = []

    # -- derived quantities ------------------------------------------------
    def link_mass(self):
        """Mass carried by URDF links, i.e. the mass whose position is known."""
        return sum(l.mass for l in self.links.values())

    def embedded_mass(self):
        return sum(m for m, _ in self.embedded_masses)

    def total_mass(self):
        """Everything Gazebo will actually simulate, links plus embedded SDF."""
        return self.link_mass() + self.embedded_mass()

    def center_of_mass(self):
        """Mass-weighted mean of the link inertial origins, in the base frame.

        Each link contributes its own centre of mass, which is the origin of
        its <inertial> block expressed in the link frame, pushed through the
        link's pose.  Summing the link masses alone (ignoring where they are)
        would give the mass but not the balance point, and the balance point is
        what determines tip-over behaviour on the ramp.

        Embedded SDF bodies are excluded because their pose lives inside a
        plugin's own frame rather than the kinematic tree; use
        located_mass_fraction() to see how much mass that leaves out.
        """
        m_total = self.link_mass()
        if m_total <= 0:
            return np.array([np.nan] * 3)
        acc = np.zeros(3)
        for l in self.links.values():
            if l.mass <= 0:
                continue
            p = (l.pose @ l.inertial_origin)[:3, 3]
            acc += l.mass * p
        return acc / m_total

    def located_mass_fraction(self):
        """Share of the simulated mass whose position the CoG accounts for."""
        t = self.total_mass()
        return self.link_mass() / t if t > 0 else float('nan')

    def collision_bbox(self):
        """Axis-aligned bounding box of all collision geometry, base frame.

        Returns (min_xyz, max_xyz, skipped_meshes).  Primitive geometry is
        evaluated exactly by transforming its corner set; mesh collisions are
        counted and skipped, because their extent lives in a binary file and
        including a wrong guess would be worse than reporting the omission.
        """
        pts = []
        skipped = 0
        for l in self.links.values():
            for T_c, geom in l.collisions:
                corners = _geometry_corners(geom)
                if corners is None:
                    skipped += 1
                    continue
                T = l.pose @ T_c
                for c in corners:
                    pts.append((T @ np.append(c, 1.0))[:3])
        if not pts:
            return None, None, skipped
        P = np.array(pts)
        return P.min(axis=0), P.max(axis=0), skipped


def _geometry_corners(geom):
    """Corner samples of a primitive, in its own frame."""
    kind = geom.get('type')
    if kind == 'box':
        sx, sy, sz = np.array(geom['size']) / 2.0
        return np.array([[x, y, z]
                         for x in (-sx, sx)
                         for y in (-sy, sy)
                         for z in (-sz, sz)])
    if kind == 'cylinder':
        r = geom['radius']
        h = geom['length'] / 2.0
        # The cylinder axis is +z in both URDF and SDF; sampling the rim
        # rather than a box keeps the bound tight after rotation.
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        pts = []
        for z in (-h, h):
            for a in ang:
                pts.append([r * np.cos(a), r * np.sin(a), z])
        return np.array(pts)
    if kind == 'sphere':
        r = geom['radius']
        return np.array([[x, y, z]
                         for x in (-r, r) for y in (-r, r) for z in (-r, r)])
    return None       # mesh, plane, polyline: not evaluated


def _parse_geometry(node):
    if node is None:
        return None
    box = node.find('box')
    if box is not None:
        size = box.find('size')
        txt = size.text if size is not None else box.get('size')
        return {'type': 'box', 'size': parse_xyz(txt, (0, 0, 0))}
    cyl = node.find('cylinder')
    if cyl is not None:
        r = cyl.find('radius')
        l = cyl.find('length')
        return {'type': 'cylinder',
                'radius': float(r.text) if r is not None else float(cyl.get('radius', 0)),
                'length': float(l.text) if l is not None else float(cyl.get('length', 0))}
    sph = node.find('sphere')
    if sph is not None:
        r = sph.find('radius')
        return {'type': 'sphere',
                'radius': float(r.text) if r is not None else float(sph.get('radius', 0))}
    mesh = node.find('mesh')
    if mesh is not None:
        uri = mesh.find('uri')
        return {'type': 'mesh',
                'uri': uri.text if uri is not None else mesh.get('filename', '')}
    return None


# ---------------------------------------------------------------------------
# URDF
# ---------------------------------------------------------------------------

def _run_xacro(path):
    for cmd in (['xacro', '--inorder', path], ['xacro', path]):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError):
            continue
    raise RuntimeError('could not run xacro on {}'.format(path))


def load_urdf(path):
    if path.endswith('.xacro'):
        root = ET.fromstring(_run_xacro(path))
    else:
        root = ET.parse(path).getroot()

    model = RobotModel(root.get('name', os.path.basename(path)), path)

    for ln in root.findall('link'):
        link = Link(ln.get('name'))
        inertial = ln.find('inertial')
        if inertial is not None:
            m = inertial.find('mass')
            if m is not None:
                link.mass = float(m.get('value', 0.0))
            o = inertial.find('origin')
            if o is not None:
                link.inertial_origin = make_T(parse_xyz(o.get('xyz')),
                                              parse_xyz(o.get('rpy')))
            i = inertial.find('inertia')
            if i is not None:
                link.inertia = {k: float(i.get(k, 0.0))
                                for k in ('ixx', 'iyy', 'izz',
                                          'ixy', 'ixz', 'iyz')}
        for coll in ln.findall('collision'):
            o = coll.find('origin')
            T = make_T(parse_xyz(o.get('xyz') if o is not None else None),
                       parse_xyz(o.get('rpy') if o is not None else None))
            geom = _parse_geometry(coll.find('geometry'))
            if geom:
                link.collisions.append((T, geom))
        model.links[link.name] = link

    for jn in root.findall('joint'):
        j = Joint(jn.get('name'), jn.get('type'))
        j.parent = jn.find('parent').get('link')
        j.child = jn.find('child').get('link')
        o = jn.find('origin')
        if o is not None:
            j.origin = make_T(parse_xyz(o.get('xyz')), parse_xyz(o.get('rpy')))
        a = jn.find('axis')
        if a is not None:
            j.axis = parse_xyz(a.get('xyz'), (1, 0, 0))
        lim = jn.find('limit')
        if lim is not None:
            j.limit = {k: float(lim.get(k)) for k in
                       ('lower', 'upper', 'effort', 'velocity')
                       if lim.get(k) is not None}
        model.joints[j.name] = j

    # Sensors declared in <gazebo reference="link"> blocks.
    for gz in root.findall('gazebo'):
        ref = gz.get('reference')
        for s in gz.findall('sensor'):
            info = _sensor_info(s)
            info['link'] = ref
            if ref in model.links:
                model.links[ref].sensors.append(info)
        for p in gz.findall('plugin'):
            model.plugins.append(_plugin_info(p, ref))
        # Bodies that only exist as embedded SDF: gazebo_continuous_track
        # builds the track segments this way, and they are as real to the
        # solver as any URDF link.  Anything under an <inertial> here is
        # simulated mass and belongs in the total.
        for inertial in gz.iter('inertial'):
            m = inertial.find('mass')
            if m is None or not (m.text or '').strip():
                continue
            model.embedded_masses.append(
                (float(m.text), 'embedded SDF under <gazebo reference="%s">'
                 % (ref or '-')))

    _urdf_forward_kinematics(model)
    return model


def _urdf_forward_kinematics(model):
    children = {}
    parent_of = {}
    for j in model.joints.values():
        children.setdefault(j.parent, []).append(j)
        parent_of[j.child] = j

    roots = [n for n in model.links if n not in parent_of]
    base = 'base_link' if 'base_link' in model.links else (
        roots[0] if roots else None)
    model.base = base
    if base is None:
        return

    model.links[base].pose = np.eye(4)
    stack = [base]
    seen = {base}
    while stack:
        cur = stack.pop()
        for j in children.get(cur, []):
            if j.child in seen or j.child not in model.links:
                continue
            model.links[j.child].pose = model.links[cur].pose @ j.origin
            seen.add(j.child)
            stack.append(j.child)


# ---------------------------------------------------------------------------
# SDF
# ---------------------------------------------------------------------------

def load_sdf(path):
    root = ET.parse(path).getroot()
    mnode = root.find('model')
    if mnode is None:
        mnode = root
    model = RobotModel(mnode.get('name', os.path.basename(path)), path)
    model_pose = parse_pose(_text(mnode.find('pose')))

    for ln in mnode.findall('link'):
        link = Link(ln.get('name'))
        link.pose = parse_pose(_text(ln.find('pose')))

        inertial = ln.find('inertial')
        if inertial is not None:
            m = inertial.find('mass')
            if m is not None and m.text:
                link.mass = float(m.text)
            link.inertial_origin = parse_pose(_text(inertial.find('pose')))
            i = inertial.find('inertia')
            if i is not None:
                link.inertia = {k: float(_text(i.find(k)) or 0.0)
                                for k in ('ixx', 'iyy', 'izz',
                                          'ixy', 'ixz', 'iyz')}

        for coll in ln.findall('collision'):
            T = parse_pose(_text(coll.find('pose')))
            geom = _parse_geometry(coll.find('geometry'))
            if geom:
                link.collisions.append((T, geom))

        for s in ln.findall('sensor'):
            info = _sensor_info(s)
            info['link'] = link.name
            link.sensors.append(info)

        model.links[link.name] = link

    for jn in mnode.findall('joint'):
        j = Joint(jn.get('name'), jn.get('type'))
        p = _text(jn.find('parent'))
        c = _text(jn.find('child'))
        j.parent, j.child = p, c
        j.origin = parse_pose(_text(jn.find('pose')))
        ax = jn.find('axis')
        if ax is not None:
            j.axis = parse_xyz(_text(ax.find('xyz')), (1, 0, 0))
            lim = ax.find('limit')
            if lim is not None:
                for k, tag in (('lower', 'lower'), ('upper', 'upper'),
                               ('effort', 'effort'), ('velocity', 'velocity')):
                    v = _text(lim.find(tag))
                    if v:
                        j.limit[k] = float(v)
        model.joints[j.name] = j

    for p in mnode.findall('plugin'):
        model.plugins.append(_plugin_info(p, None))

    # SDF link poses are already model-relative.  Re-express them relative to
    # base_link so the three platforms are compared in the same frame.
    base = 'base_link' if 'base_link' in model.links else None
    model.base = base or model.name
    if base:
        inv = np.linalg.inv(model.links[base].pose)
        for l in model.links.values():
            l.pose = inv @ l.pose
    elif not np.allclose(model_pose, np.eye(4)):
        for l in model.links.values():
            l.pose = model_pose @ l.pose

    return model


def _text(node):
    return node.text.strip() if node is not None and node.text else None


def _sensor_info(s):
    info = {
        'name': s.get('name'),
        'type': s.get('type'),
        'pose': parse_pose(_text(s.find('pose'))),
        'update_rate': _text(s.find('update_rate')),
        'topic': None,
        'frame': None,
    }
    for p in s.findall('plugin'):
        for tag in ('topicName', 'bumperTopicName', 'imageTopicName',
                    'robotNamespace'):
            v = _text(p.find(tag))
            if v and not info['topic']:
                info['topic'] = v
        v = _text(p.find('frameName'))
        if v:
            info['frame'] = v
    return info


def _plugin_info(p, ref):
    d = {'name': p.get('name'), 'filename': p.get('filename'),
         'reference': ref, 'params': {}}
    for child in p:
        if child.text and child.text.strip():
            d['params'][child.tag] = child.text.strip()
    return d


# ---------------------------------------------------------------------------

def load(path):
    """Dispatch on file extension."""
    if path.endswith(('.sdf', '.world')):
        return load_sdf(path)
    return load_urdf(path)
