#!/usr/bin/env python3
"""Quasi-static climb of a vertical step by one side of the rocker-bogie.

Reproduces the model of the paper (Sect. "Quasi-static step-climbing model")
and Fig. rb_step_snapshot (Imagenes/rocker_bogie_step_20cm.eps).

The geometry is read from the expanded URDF, left side, in the XZ plane of
base_link.  The differential holds the rocker rigid to the chassis on a step
that spans the full track, so the pose is q = [x_c, z_c, theta, eta]
(chassis position, chassis pitch, bogie angle relative to the chassis).
Three closed contacts plus a progress constraint make the system square; it
is SOLVED (Powell hybrid), not optimised, station by station along the sum of
the wheel-centre arc lengths, with a linear predictor.

Phases:
  F_G  front, middle and rear wheels on the terrain          (g_fm >= 0)
  F_W  middle wheel resting on the front wheel, off terrain  (g_m  >= 0)
A station is also rejected if the bogie leaves its joint limits.

Usage (in WSL, same matplotlib as the published figure):
  python3 rocker_step_quasistatic.py [--urdf expanded.urdf] [--h 0.20]
                                     [--out fig.eps]
Without --urdf the xacro is expanded with the campaign launch arguments.
"""
import argparse
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import root

# Campaign arguments of lcmine_rocker_bogie_world.launch.
XACRO_ARGS = ['differential_linkage:=true', 'bogie_upper:=0.6',
              'bogie_drop:=0.0', 'rocker_drop:=0.299477']
L_TERRAIN = 3.0        # half-length of each terrain level [m]
DS = 0.006             # progress step [m]
TOL = 1e-8             # accepted residual [m]
START_CX = -1.3        # chassis x at the first station [m]
END_CX = 0.8495        # sweep stops past this chassis x [m]
# Stations of the published figure.  They reproduce the wheel centres stored
# in the published EPS to within 0.05 mm.
SNAP_STATIONS = (0, 434, 541, 651, 839, -1)
SNAP_TITLES = ['(a) on the lower level', '(b) front wheel climbing',
               '(c) front wheel up', '(d) middle wheel climbing',
               '(e) rear wheel climbing', '(f) on the upper level']


# ----------------------------------------------------------------- geometry
def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p), np.sin(p),
                              np.cos(y), np.sin(y))
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp, cp*sr, cp*cr]])


def _tf(origin):
    m = np.eye(4)
    if origin is not None:
        m[:3, 3] = [float(v) for v in origin.get('xyz', '0 0 0').split()]
        m[:3, :3] = _rpy(*[float(v) for v in origin.get('rpy', '0 0 0').split()])
    return m


def load_geometry(urdf_text):
    """Zero-configuration XZ geometry of the left side, in base_link."""
    root_el = ET.fromstring(urdf_text)
    parent, joint = {}, {}
    for j in root_el.findall('joint'):
        child = j.find('child').get('link')
        parent[child] = (j.find('parent').get('link'), _tf(j.find('origin')))
        joint[j.get('name')] = j
    links = {l.get('name'): l for l in root_el.findall('link')}
    cache = {}

    def world(name):
        if name not in cache:
            cache[name] = (np.eye(4) if name not in parent
                           else world(parent[name][0]) @ parent[name][1])
        return cache[name]

    def xz(m):
        return np.array([m[0, 3], m[2, 3]])

    def collision(name):
        c = links[name].find('collision')
        return world(name) @ _tf(c.find('origin')), c.find('geometry')[0]

    def child_of(jname):
        return joint[jname].find('child').get('link')

    wheel = {}
    for k, jn in (('f', 'rev_11'), ('m', 'rev_10'), ('r', 'rev_9')):
        m, g = collision(child_of(jn))
        wheel[k] = xz(m)
        radius = float(g.get('radius'))
    bogie_lim = joint['rev_3'].find('limit')
    m, g = collision('chassis_link')
    size = np.array([float(v) for v in g.get('size').split()])
    corners = np.array([m @ np.r_[size * np.array(s) / 2, 1]
                        for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)])
    lidar_m, lidar_g = collision('velodyne_base_link')
    beam = xz(world('differential_beam'))
    pivot = xz(world(child_of('rocker_pivot_left')))
    return dict(
        R=radius, r_f=wheel['f'], r_m=wheel['m'], r_r=wheel['r'],
        r_b=xz(world(child_of('rev_3'))), pivot=pivot,
        eta_lim=(float(bogie_lim.get('lower')), float(bogie_lim.get('upper'))),
        chassis=(corners[:, 0].min(), corners[:, 0].max(),
                 corners[:, 2].min(), corners[:, 2].max()),
        lidar=xz(lidar_m), lidar_r=float(lidar_g.get('radius')),
        camera=xz(world('camera_link')), beam=beam,
        arm_tip=np.array([pivot[0], beam[1]]))


# ------------------------------------------------------------------ terrain
class Step:
    """Polyline (-L,0) (0,0) (0,h) (L,h) and the arc length of a wheel centre
    along the terrain offset by R (the path of a centre while in contact)."""

    def __init__(self, h, R, L=L_TERRAIN):
        self.h, self.R, self.L = h, R, L
        self.seg = [(np.array([-L, 0.0]), np.array([0.0, 0.0])),
                    (np.array([0.0, 0.0]), np.array([0.0, h])),
                    (np.array([0.0, h]), np.array([L, h]))]
        if h >= R:
            xa, self.a0 = -R, np.pi
        else:
            self.a0 = np.pi - np.arcsin((R - h) / R)
            xa = R * np.cos(self.a0)
        ls = [L + xa, max(0.0, h - R), R * (self.a0 - np.pi / 2), L]
        self.C = np.cumsum(ls)

    def closest(self, p):
        best = None
        for i, (a, b) in enumerate(self.seg):
            d = b - a
            t = np.clip((p - a) @ d / (d @ d), 0.0, 1.0)
            q = a + t * d
            dist = np.linalg.norm(p - q)
            if best is None or dist < best[0]:
                best = (dist, i, t, q)
        return best

    def gap(self, p):
        return self.closest(p)[0] - self.R

    def s(self, p):
        _, i, t, q = self.closest(p)
        if i == 0:
            return q[0] + self.L
        if (i == 1 and t >= 1.0) or (i == 2 and t <= 0.0):
            return self.C[1] + self.R * (self.a0 - np.arctan2(p[1] - self.h, p[0]))
        if i == 1:
            return self.C[0] + q[1] - self.R
        return self.C[2] + q[0]


# -------------------------------------------------------------------- model
def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def wheels(q, G):
    c, th, eta = q[:2], q[2], q[3]
    Rt = rot(th)
    p_b = c + Rt @ G['r_b']
    Rb = Rt @ rot(eta)
    return dict(f=c + Rt @ G['r_f'], b=p_b,
                m=p_b + Rb @ (G['r_m'] - G['r_b']),
                r=p_b + Rb @ (G['r_r'] - G['r_b']))


def residual(q, S, phase, G, T):
    p = wheels(q, G)
    g_fm = np.linalg.norm(p['m'] - p['f']) - 2 * G['R']
    prog = T.s(p['f']) + T.s(p['m']) + T.s(p['r']) - S
    mid = T.gap(p['m']) if phase == 'G' else g_fm
    return np.array([T.gap(p['f']), mid, T.gap(p['r']), prog])


def gaps(q, G, T):
    p = wheels(q, G)
    return dict(f=T.gap(p['f']), m=T.gap(p['m']), r=T.gap(p['r']),
                fm=np.linalg.norm(p['m'] - p['f']) - 2 * G['R'])


def solve_station(q0, S, G, T):
    """First admissible phase, trying ground contact before wheel-wheel."""
    lo, hi = G['eta_lim']
    for phase, other in (('G', 'fm'), ('W', 'm')):
        sol = root(residual, q0, args=(S, phase, G, T), method='hybr')
        q = sol.x
        if (np.max(np.abs(residual(q, S, phase, G, T))) < TOL
                and gaps(q, G, T)[other] >= -TOL and lo <= q[3] <= hi):
            return phase, q
    return None, q0


def sweep(G, h):
    T = Step(h, G['R'])
    q = np.array([START_CX, G['R'] - G['r_m'][1], 0.0, 0.0])
    S0 = sum(T.s(p) for k, p in wheels(q, G).items() if k != 'b')
    out, prev, n = [], None, 0
    while q[0] <= END_CX:
        S = S0 + n * DS
        q0 = q if prev is None else 2 * q - prev
        phase, qn = solve_station(q0, S, G, T)
        if phase is None:
            print('JAMMED at S = %.3f m, x_c = %.3f m' % (S, q[0]))
            return T, out, False
        out.append((S, phase, qn, gaps(qn, G, T)))
        prev, q, n = q, qn, n + 1
    return T, out, True


# --------------------------------------------------------------------- plot
def draw(ax, G, T, q, title, first, dim):
    import matplotlib.patches as mp
    R, h, L = G['R'], T.h, T.L
    Rt = rot(q[2])
    X = lambda v: q[:2] + Rt @ v                       # chassis -> world
    p = wheels(q, G)
    cx = q[0]
    x0, x1 = cx - 0.75, cx + 0.80
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.12, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Two layers around the text (zorder 3): terrain and body below, the
    # mechanism above.  Within a layer, insertion order is drawing order.
    lo, hi = dict(zorder=1), dict(zorder=4)
    ax.add_patch(mp.Polygon([(-L, 0), (0, 0), (0, h), (L, h), (L, -0.5), (-L, -0.5)],
                            fill=False, hatch='/////', edgecolor='0.55', lw=0, **lo))
    ax.plot([-L, 0, 0, L], [0, 0, h, h], color='k', lw=1.0, **lo)
    ax.text(x1 - 0.02, 0.98, 'pitch %+.1f$^\\circ$\nbogie %+.1f$^\\circ$'
            % (np.degrees(q[2]), np.degrees(q[3])),
            ha='right', va='top', fontsize=7, color='0.25')
    if dim:
        xd = x1 - 0.08
        for z in (0.0, h):
            ax.plot([0.013, xd + 0.016], [z, z], color='k', lw=0.4, **lo)
        ax.annotate('', xy=(xd, 0), xytext=(xd, h),
                    arrowprops=dict(arrowstyle='<->', lw=0.6, shrinkA=0,
                                    shrinkB=0, mutation_scale=9))
        ax.text(xd + 0.012, h / 2, '%.1f' % (h * 1000), rotation=90,
                ha='left', va='center', fontsize=7.5)

    xa, xb, za, zb = G['chassis']
    ax.add_patch(mp.Polygon([X(np.array(v)) for v in ((xa, za), (xb, za), (xb, zb), (xa, zb))],
                            fc='0.94', ec='k', lw=1.3, **lo))
    top = X(np.array([G['lidar'][0], zb]))
    lid = X(G['lidar'])
    ax.plot([lid[0], top[0]], [lid[1], top[1]], color='k', lw=1.1, **lo)
    if first:
        y = 0.9
        ax.plot([x0 + 0.03, x0 + 0.53], [y, y], color='k', lw=1.0, **lo)
        for xe in (x0 + 0.03, x0 + 0.53):
            ax.plot([xe, xe], [y - 0.025, y + 0.025], color='k', lw=1.0, **lo)
        ax.text(x0 + 0.28, y - 0.04, '0.5 m', ha='center', va='top', fontsize=7)
    ax.set_title(title, fontsize=7.5)

    ax.add_patch(mp.Circle(lid, G['lidar_r'], fc='0.88', ec='k', lw=1.1, **hi))
    cam = X(G['camera'])
    ax.add_patch(mp.Rectangle(cam - 0.019, 0.038, 0.038, fc='0.88', ec='k', lw=1.1, **hi))

    piv, bog, tip, beam = X(G['pivot']), p['b'], X(G['arm_tip']), X(G['beam'])
    link = dict(color='k', lw=2.2, solid_capstyle='round', solid_joinstyle='round', **hi)
    for a, b in ((piv, p['f']), (piv, bog), (bog, p['m']), (bog, p['r']), (piv, tip)):
        ax.plot([a[0], b[0]], [a[1], b[1]], **link)
    ax.plot([beam[0], tip[0]], [beam[1], tip[1]], **dict(link, lw=1.1))
    ax.add_patch(mp.Circle(beam, 0.01, fc='w', ec='k', lw=1.1, **hi))
    for k in 'fmr':
        ax.add_patch(mp.Circle(p[k], R, fill=False, ec='k', lw=1.3, **hi))
    ax.plot(*beam, marker='+', ls='none', ms=4, mew=1.0, color='k', **hi)
    centre = dict(color='k', lw=0.5, linestyle=(0, (7, 2, 1, 2)), **hi)
    for k in 'fmr':
        c = p[k]
        ax.plot([c[0] - 0.045, c[0] + 0.045], [c[1], c[1]], **centre)
        ax.plot([c[0], c[0]], [c[1] - 0.045, c[1] + 0.045], **centre)
    for c in (piv, bog):
        ax.add_patch(mp.Circle(c, 0.017, fc='w', ec='k', lw=1.1, **hi))
    ax.add_patch(mp.Circle(tip, 0.013, fc='k', ec='k', lw=1.0, **hi))
    for c in (piv, bog):
        ax.plot(*c, marker='o', ls='none', ms=1.25, mew=1.0, color='k', **hi)


def figure(G, T, snaps, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'cm'})
    W, H = 367.02, 227.667069          # published figure, in points
    fig = plt.figure(figsize=(W / 72, H / 72))
    for k, ((_, _, q, _), title) in enumerate(zip(snaps, SNAP_TITLES)):
        x = (2.16, 125.01, 247.86)[k % 3]
        y = (128.887, 2.16)[k // 3]
        ax = fig.add_axes([x / W, y / H, 117 / W, 84.542 / H])
        draw(ax, G, T, q, title, first=(k == 0), dim=(k == 1))
    fig.savefig(path)
    if path.endswith('.eps'):
        fig.savefig(path[:-4] + '.png', dpi=200)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--urdf', help='expanded URDF (default: run xacro)')
    ap.add_argument('--h', type=float, default=0.20, help='step height [m]')
    ap.add_argument('--out', help='write the snapshot figure here (.eps)')
    a = ap.parse_args()

    if a.urdf:
        text = open(a.urdf).read()
    else:
        xacro = subprocess.check_output(['rospack', 'find', 'rocker_bogie'],
                                        text=True).strip() + '/urdf/ensamblajeurdf.xacro'
        text = subprocess.check_output(['xacro', xacro] + XACRO_ARGS, text=True)
    G = load_geometry(text)
    T, st, done = sweep(G, a.h)

    S0 = st[0][0]
    ph = np.array([s[1] for s in st])
    fm = np.array([s[3]['fm'] for s in st])
    eta = np.array([s[2][3] for s in st])
    th = np.array([s[2][2] for s in st])
    lo, hi = G['eta_lim']
    print('h = %.3f m, R = %.3f m, %d stations, climb %s'
          % (a.h, G['R'], len(st), 'completed' if done else 'NOT completed'))
    print('stations in F_W (middle wheel on front wheel): %d' % np.sum(ph == 'W'))
    print('min front-middle gap g_fm = %.1f mm at progress %.3f m (phase %s)'
          % (1e3 * fm.min(), st[fm.argmin()][0] - S0, ph[fm.argmin()]))
    print('pitch   %+.1f .. %+.1f deg' % (np.degrees(th.min()), np.degrees(th.max())))
    print('bogie   %+.1f .. %+.1f deg, limits %+.1f / %+.1f deg, min margin %.1f deg'
          % (np.degrees(eta.min()), np.degrees(eta.max()), np.degrees(lo),
             np.degrees(hi), np.degrees(min(eta.min() - lo, hi - eta.max()))))

    if a.out:
        snaps = [st[i] for i in SNAP_STATIONS]
        for (S, p, q, g), t in zip(snaps, SNAP_TITLES):
            print('%-28s progress %.3f  phase %s  pitch %+.1f  bogie %+.1f  g_fm %.1f mm'
                  % (t, S - S0, p, np.degrees(q[2]), np.degrees(q[3]), 1e3 * g['fm']))
        figure(G, T, snaps, a.out)


if __name__ == '__main__':
    main()
