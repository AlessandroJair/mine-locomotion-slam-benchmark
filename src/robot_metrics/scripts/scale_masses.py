#!/usr/bin/env python3
"""Uniformly scale every link mass and inertia tensor of each platform so the
three compared robots have the same total mass.

Why uniform scaling: multiplying every <mass> and every <inertia> component of
a model by the same factor k leaves the centre of gravity *exactly* unchanged
and scales the inertia tensor consistently with the new mass at unchanged
geometry (it is the same body at density * k).  Nothing about the shape, the
joint frames, the contact geometry or the sensor placement moves.  The only
thing that changes is how heavy the platform is - precisely the confound being
removed, and nothing else.

The native totals were 94.929 / 46.967 / 45.254 kg (Rocker-Bogie / tracked /
Husky), a 2.1x spread that made the locomotion comparison unfair: at equal
motor torque the light platforms accelerate harder, and at equal wheel friction
coefficient the heavy platform gets proportionally more traction force.

Usage:
    python3 scale_masses.py --src <workspace>/src           # report only
    python3 scale_masses.py --src <workspace>/src --apply   # write

Idempotent: a file that already carries the MASS MATCHING banner is skipped.
"""
import argparse
import os
import re
import sys

TARGET_MASS = 45.0

RB = 'rocker_bogie/urdf/ensamblajeurdf.xacro'
TRACKED = 'tracked/gazebo_continuous_track_example/urdf_xacro/two_track_robot.urdf.xacro'
# The continuous-track plugin builds the track bodies from <gazebo> blocks, so
# 3.0 kg of the tracked robot lives in a second file and never appears as a
# URDF link.  It has to be scaled together with the rest of the platform.
TRACKED_GAZEBO = ('tracked/gazebo_continuous_track_example/urdf_xacro/'
                  'two_track_robot_gazebo.urdf.xacro')
HUSKY = 'differential/model/model.sdf'

MARKER = 'MASS MATCHING'
INERTIA_ATTRS = ('ixx', 'iyy', 'izz', 'ixy', 'iyz', 'ixz')
# Macros that derive a whole inertia tensor from a mass argument: scaling the
# argument scales the tensor, so their call sites are rewritten but their
# bodies must not be (that would scale twice).
INERTIA_MACROS = ('make_box_inertia', 'make_wheel_inertia', 'make_track')


def banner(factor, original_total, extra='', indent='  '):
    text = (
        '<!-- ==================== %s ====================\n'
        '     Every link mass and every inertia component below is multiplied by\n'
        '     mass_scale so that the three compared platforms have the same total\n'
        '     mass (%.1f kg).  This model natively weighed %.4f kg, hence\n'
        '         mass_scale = %.1f / %.4f = %.6f\n'
        '     Uniform scaling leaves the centre of gravity and every geometric\n'
        '     property untouched; only the inertial properties change.  Set\n'
        '     mass_scale to 1.0 to recover the native model.%s\n'
        '     ============================================================ -->\n'
        % (MARKER, TARGET_MASS, original_total, TARGET_MASS, original_total,
           factor, extra))
    return ''.join(indent + ln if ln.strip() else ln
                   for ln in text.splitlines(True))


def wrap(value, expr='mass_scale'):
    return '${%s*%s}' % (value.strip(), expr)


def scale_urdf_inertials(text, expr='mass_scale'):
    """Rewrite <mass value="N"/> and <inertia .../> numeric literals as xacro
    expressions multiplied by `expr`.  Literals already containing ${} are left
    alone."""
    counts = {'mass': 0, 'inertia': 0}

    def mass_sub(m):
        counts['mass'] += 1
        return '<mass value="%s"' % wrap(m.group(1), expr)

    text = re.sub(r'<mass\s+value="([^"${]+)"', mass_sub, text)

    def inertia_sub(m):
        body = m.group(1)
        touched = [False]

        def attr_sub(a):
            touched[0] = True
            return '%s="%s"' % (a.group(1), wrap(a.group(2), expr))

        body = re.sub(r'\b(%s)="([^"${]+)"' % '|'.join(INERTIA_ATTRS),
                      attr_sub, body)
        if touched[0]:
            counts['inertia'] += 1
        return '<inertia%s/>' % body

    text = re.sub(r'<inertia([^>]*)/>', inertia_sub, text)
    return text, counts


def scale_macro_calls(text, expr='mass_scale'):
    """Rewrite the mass= argument of every inertia-generating macro call."""
    n = [0]

    def sub(m):
        n[0] += 1
        return '%s%smass="%s"' % (m.group(1), m.group(2), wrap(m.group(3), expr))

    pat = r'(<xacro:(?:%s))(\s+[^>]*?)?mass="([^"${]+)"' % '|'.join(INERTIA_MACROS)
    return re.sub(pat, sub, text), n[0]


def sum_macro_masses(text):
    pat = r'<xacro:(?:%s)\s+[^>]*?mass="([^"${]+)"' % '|'.join(INERTIA_MACROS)
    return sum(float(v) for v in re.findall(pat, text))


def split_macro_defs(text):
    """Return (definitions, rest) so macro *bodies* are never rewritten."""
    end = text.rfind('</xacro:macro>')
    if end < 0:
        return '', text
    end = text.index('\n', end) + 1
    return text[:end], text[end:]


# --------------------------------------------------------------------------
def do_rocker_bogie(path, apply_):
    text = open(path).read()
    if MARKER in text:
        print('  already scaled - skipping')
        return
    total = sum(float(m) for m in re.findall(r'<mass\s+value="([^"${]+)"', text))
    factor = TARGET_MASS / total
    new, c = scale_urdf_inertials(text)
    anchor = '<xacro:include filename="$(find rocker_bogie)/urdf/ensamblajeurdf.gazebo" />\n'
    assert anchor in new, 'anchor not found'
    prop = '<xacro:property name="mass_scale" value="%.6f"/>\n\n' % factor
    new = new.replace(anchor, anchor + '\n' + banner(factor, total, indent='')
                      + prop, 1)
    print('  %.4f kg -> %.1f kg (factor %.6f); %d masses, %d inertia tags'
          % (total, TARGET_MASS, factor, c['mass'], c['inertia']))
    if apply_:
        open(path, 'w').write(new)


def do_tracked(path_urdf, path_gazebo, apply_):
    t_urdf = open(path_urdf).read()
    t_gz = open(path_gazebo).read()
    if MARKER in t_urdf:
        print('  already scaled - skipping')
        return

    defs, body = split_macro_defs(t_urdf)
    total = (sum_macro_masses(body)
             + sum(float(m) for m in re.findall(r'<mass\s+value="([^"${]+)"', body))
             + sum_macro_masses(t_gz))

    factor = TARGET_MASS / total
    body, n_macro = scale_macro_calls(body)
    body, c = scale_urdf_inertials(body)
    t_gz, n_gz = scale_macro_calls(t_gz)

    anchor = '<robot name="two_track_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">\n'
    assert anchor in defs, 'anchor not found'
    extra = ('\n     make_box_inertia / make_wheel_inertia / make_track derive the\n'
             '     tensor from their mass argument, so scaling that argument scales\n'
             '     the whole tensor.  mass_scale is defined here and reused by\n'
             '     two_track_robot_gazebo.urdf.xacro, which includes this file and\n'
             '     holds the 3.0 kg of track bodies.')
    defs = defs.replace(anchor, anchor + '\n' + banner(factor, total, extra)
                        + '    <xacro:property name="mass_scale" value="%.6f"/>\n\n'
                        % factor, 1)
    print('  %.4f kg -> %.1f kg (factor %.6f); %d+%d macro calls, %d masses, '
          '%d inertia tags' % (total, TARGET_MASS, factor, n_macro, n_gz,
                               c['mass'], c['inertia']))
    if apply_:
        open(path_urdf, 'w').write(defs + body)
        open(path_gazebo, 'w').write(t_gz)


def do_husky(path, apply_):
    """SDF has no expression support, so the literals are scaled in place and
    the factor is recorded in a comment for reversibility."""
    text = open(path).read()
    if MARKER in text:
        print('  already scaled - skipping')
        return
    total = sum(float(m) for m in re.findall(r'<mass>([^<${]+)</mass>', text))
    factor = TARGET_MASS / total

    n = {'m': 0, 'i': 0}

    def mass_sub(m):
        n['m'] += 1
        return '<mass>%.6g</mass>' % (float(m.group(1)) * factor)

    new = re.sub(r'<mass>([^<${]+)</mass>', mass_sub, text)

    def inertia_sub(m):
        n['i'] += 1
        return '<%s>%.6g</%s>' % (m.group(1), float(m.group(2)) * factor,
                                  m.group(1))

    new = re.sub(r'<(%s)>([^<${]+)</\1>' % '|'.join(INERTIA_ATTRS),
                 inertia_sub, new)

    comment = (
        '  <!-- ==================== %s ====================\n'
        '       Every <mass> and every inertia component in this file has been\n'
        '       multiplied by %.6f = %.1f / %.4f so that the three compared\n'
        '       platforms have the same total mass (%.1f kg); this model natively\n'
        '       weighed %.4f kg.  Uniform scaling leaves the centre of gravity and\n'
        '       all geometry untouched.  SDF has no expression support, so the\n'
        '       literals are pre-multiplied - divide by %.6f to recover the native\n'
        '       model.\n'
        '       ============================================================ -->\n'
        % (MARKER, factor, TARGET_MASS, total, TARGET_MASS, total, factor))
    anchor = re.search(r'<model name=[\'"][^\'"]*[\'"]\s*>\s*\n', new)
    assert anchor, 'anchor not found'
    new = new[:anchor.end()] + comment + new[anchor.end():]
    print('  %.4f kg -> %.1f kg (factor %.6f); %d masses, %d inertia components'
          % (total, TARGET_MASS, factor, n['m'], n['i']))
    if apply_:
        open(path, 'w').write(new)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true',
                    help='write the files (default: report only)')
    ap.add_argument('--src', default='.', help='path to the catkin src/ folder')
    a = ap.parse_args()
    j = lambda p: os.path.join(a.src, p)

    print('rocker_bogie')
    do_rocker_bogie(j(RB), a.apply)
    print('tracked')
    do_tracked(j(TRACKED), j(TRACKED_GAZEBO), a.apply)
    print('husky (differential)')
    do_husky(j(HUSKY), a.apply)
    if not a.apply:
        print('\n(dry run - pass --apply to write)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
