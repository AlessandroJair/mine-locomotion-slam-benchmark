#!/usr/bin/env python3
"""Project the shared world-frame route into each robot's map frame.

The route lives in robot_metrics/config/route_mine.yaml in Gazebo world
coordinates.  Each robot's SLAM map frame is created at that robot's spawn
pose, so a world point has to be transformed before it can be used as a goal:

    p_map = R(-yaw_spawn) * (p_world - p_spawn)
    yaw_map = yaw_world - yaw_spawn

The three platforms spawn at the same x, y but the Rocker-Bogie spawns at
yaw = pi (its base_link is authored facing -x), which is exactly why the
hand-maintained files needed that "x*-1, y*-1, yaw+180" transform - and exactly
why doing it by hand was fragile.  Here it falls out of the spawn pose
recorded in sim_config.yaml.

The generated files carry a header saying they are generated.  Edit
route_mine.yaml, not the outputs.

Usage:
    ./generate_waypoints.py            # write all three, and report the route
    ./generate_waypoints.py --check    # verify existing files are up to date
    ./generate_waypoints.py --plot     # also render the route over the map
"""

import argparse
import math
import os
import sys

import yaml


OUTPUTS = {
    'rocker_bogie': 'rocker_bogie/config/waypoints_mine.yaml',
    'differential': 'differential/config/waypoints_mine.yaml',
    'tracked': ('tracked/gazebo_continuous_track_example/'
                'config/waypoints_mine.yaml'),
}


def wrap_rad(a):
    return math.atan2(math.sin(a), math.cos(a))


def wrap_deg(a):
    while a > 180.0:
        a -= 360.0
    while a <= -180.0:
        a += 360.0
    return a


def to_map_frame(wp, spawn_x, spawn_y, map_yaw_rad):
    """Mundo -> el frame `map` en el que move_base recibe los goals.

    `map_yaw_rad` es el rumbo del frame al que se ancla el SLAM, que NO es
    siempre yaw_spawn: el Rocker-Bogie ancla rtabmap y los costmaps a
    base_link_nav -- base_link girado 180 deg -- asi que su `map` nace en
    yaw_spawn + base_yaw_offset.  Con yaw_spawn a secas sus waypoints salian
    espejados (start_i1 en x = -3.167 donde las otras dos lo tienen en +3.167)
    y move_base lo mandaba a recorrer la ruta al reves.
    """
    dx = wp['x'] - spawn_x
    dy = wp['y'] - spawn_y
    c, s = math.cos(-map_yaw_rad), math.sin(-map_yaw_rad)
    return {
        'name': wp['name'],
        # Se redondea ANTES de envolver, y se suma 0.0 para no arrastrar
        # -0.0.  yaw_rad vale 3.14159, no pi, asi que map_yaw_rad se queda a
        # ~1e-4 deg de cero en el Rocker-Bogie; justo en el borde +-180 ese
        # pelo decidia el signo y su fichero salia con -180.0/-0.0 donde los
        # otros dos llevan 180.0/0.0.  Con esto los tres salen IDENTICOS, que
        # es la propiedad que se comprueba de un vistazo: misma ruta, mismo
        # spawn, mismo rumbo de `map`, mismo fichero.
        'x': round(c * dx - s * dy, 3) + 0.0,
        'y': round(s * dx + c * dy, 3) + 0.0,
        'yaw': wrap_deg(round(wp['yaw'] - math.degrees(map_yaw_rad), 2)) + 0.0,
    }


def in_no_go(x, y, zones):
    for z in zones:
        if (z['x_min'] <= x <= z['x_max']) and (z['y_min'] <= y <= z['y_max']):
            return z.get('name', 'zone')
    return None


def densify(waypoints, max_spacing, zones):
    """Subdivide the route so no gap exceeds max_spacing, without moving any of
    the original points.

    Done in WORLD coordinates and before projection, so every platform gets the
    same points.  Interpolated goals that fall inside a no-waypoint zone are
    dropped rather than moved: the zones exist because something is there that
    a goal must not sit on, and nudging a point off a step obstacle would just
    put it somewhere else nobody chose.
    """
    if not max_spacing or max_spacing <= 0:
        return list(waypoints), 0, 0

    out = []
    added = skipped = 0
    for a, b in zip(waypoints, waypoints[1:]):
        out.append(a)
        d = math.hypot(b['x'] - a['x'], b['y'] - a['y'])
        n = int(math.ceil(d / max_spacing)) - 1
        if n <= 0:
            continue
        # shortest angular path for the heading
        da = wrap_deg(b['yaw'] - a['yaw'])
        for k in range(1, n + 1):
            f = float(k) / (n + 1)
            x = a['x'] + f * (b['x'] - a['x'])
            y = a['y'] + f * (b['y'] - a['y'])
            zone = in_no_go(x, y, zones)
            if zone:
                skipped += 1
                continue
            out.append({
                'name': '%s_i%d' % (a['name'], k),
                'x': round(x, 3),
                'y': round(y, 3),
                'yaw': round(wrap_deg(a['yaw'] + f * da), 2),
            })
            added += 1
    out.append(waypoints[-1])
    return out, added, skipped

def route_length(waypoints):
    total = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        total += math.hypot(b['x'] - a['x'], b['y'] - a['y'])
    return total


def segment_report(waypoints):
    return [(a['name'], b['name'],
             math.hypot(b['x'] - a['x'], b['y'] - a['y']))
            for a, b in zip(waypoints, waypoints[1:])]


HEADER = """# =============================================================================
# GENERATED FILE - DO NOT EDIT
# =============================================================================
# Produced by robot_metrics/scripts/generate_waypoints.py from
#   robot_metrics/config/route_mine.yaml   (route, in Gazebo world coordinates)
#   robot_metrics/config/sim_config.yaml   (spawn pose of each platform)
#
# Robot          : {robot}
# Spawn pose     : x={sx} y={sy} yaw={syaw_deg:.1f} deg (world frame)
# Map frame yaw  : {myaw_deg:.1f} deg = yaw_spawn + base_yaw_offset, porque el
#                  SLAM se ancla al frame de navegacion, no siempre a base_link
# Transform      : p_map = R(-yaw_map) * (p_world - p_spawn)
# Route          : {route_name}, {n} waypoints, {length:.1f} m
#
# To change the route, edit route_mine.yaml and re-run generate_waypoints.py.
# Editing this file by hand puts this platform on a different route from the
# other two and invalidates the comparison.
# =============================================================================

"""


def build(robot, route, cfg):
    spawn = cfg['spawn']
    common = spawn['common']
    sx = common['x']
    sy = common['y']
    syaw = spawn[robot]['yaw_rad']
    # Donde nace `map`: ver base_yaw_offset_rad en sim_config.yaml.
    myaw = wrap_rad(syaw + spawn[robot].get('base_yaw_offset_rad', 0.0))

    # Densify in world coordinates first, so the three platforms are given the
    # same points and not three different interpolations of the same route.
    dense, _added, _skipped = densify(
        route['route']['waypoints'],
        route['route'].get('max_spacing_m'),
        route['route'].get('no_waypoint_zones') or [])

    waypoints = [to_map_frame(w, sx, sy, myaw) for w in dense]

    nav = dict(route['navigation_params'])
    limits = route.get('platform_limits', {}).get(robot, {})
    nav.update({k: v for k, v in limits.items()})

    doc = {'waypoints': waypoints, 'navigation_params': nav}
    header = HEADER.format(
        robot=robot, sx=sx, sy=sy, syaw_deg=math.degrees(syaw),
        myaw_deg=math.degrees(myaw),
        route_name=route['route']['name'], n=len(waypoints),
        length=route_length(waypoints))
    return header + yaml.safe_dump(doc, default_flow_style=False,
                                   sort_keys=False)


def plot_route(route, cfg, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    wps = route['route']['waypoints']
    xs = [w['x'] for w in wps]
    ys = [w['y'] for w in wps]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(xs, ys, '-', color='#1f77b4', linewidth=1.2, zorder=2)
    ax.scatter(xs, ys, s=28, color='#1f77b4', zorder=3, edgecolors='black',
               linewidths=0.4)
    for i, w in enumerate(wps):
        ax.annotate('{}'.format(i + 1), (w['x'], w['y']),
                    textcoords='offset points', xytext=(4, 4), fontsize=6)

    step = cfg['step_obstacle']
    cx, cy = step['pose_xyzrpy'][0], step['pose_xyzrpy'][1]
    sx, sy = step['size_m'][0], step['size_m'][1]
    ax.add_patch(Rectangle((cx - sy / 2, cy - sx / 2), sy, sx,
                           facecolor='#d62728', alpha=0.30, zorder=1,
                           label='step (0.1 m)'))
    # The ramp is drawn as a marker, not a footprint: its <pose> combines a
    # roll of 1.57 with a yaw of 4.71 on an extruded polyline, so its ground
    # footprint is not obvious from the world file and a guessed rectangle
    # would be worse than none. It is inside the loop and not traversed.
    ax.plot([25.5], [-9.5], 'x', color='#2ca02c', markersize=10,
            markeredgewidth=2, zorder=4, label='ramp (not on route)')
    ax.scatter([cfg['spawn']['common']['x']], [cfg['spawn']['common']['y']],
               marker='*', s=160, color='black', zorder=4, label='spawn')

    ax.set_xlabel('world $x$ (m)')
    ax.set_ylabel('world $y$ (m)')
    ax.set_aspect('equal')
    ax.grid(True, color='0.9', linewidth=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'eps'):
        fig.savefig('{}.{}'.format(out_path, ext), dpi=200)
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=os.path.abspath(os.path.join(here, '..', '..')))
    ap.add_argument('--route', default=os.path.join(here, '..', 'config',
                                                    'route_mine.yaml'))
    ap.add_argument('--config', default=os.path.join(here, '..', 'config',
                                                     'sim_config.yaml'))
    ap.add_argument('--check', action='store_true',
                    help='exit non-zero if a generated file is stale')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--output_dir', default=None,
                    help='where to put the route figure (default: alongside)')
    args = ap.parse_args()

    with open(args.route) as f:
        route = yaml.safe_load(f)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    corners = route['route']['waypoints']
    wps, added, skipped = densify(
        corners, route['route'].get('max_spacing_m'),
        route['route'].get('no_waypoint_zones') or [])
    print('Route "{}": {} corner points, {:.1f} m in world coordinates'
          .format(route['route']['name'], len(corners), route_length(corners)))
    if added or skipped:
        print('  densified to {} waypoints at max_spacing_m = {}: {} inserted, '
              '{} dropped inside a no-waypoint zone'
              .format(len(wps), route['route'].get('max_spacing_m'),
                      added, skipped))
    print('')
    # Report the spacing of what is actually WRITTEN, not of the corner points:
    # printing the corner spacing after densifying describes a route that no
    # platform drives.
    print('Segment spacing of the generated route:')
    segs = segment_report(wps)
    for a, b, d in segs:
        flag = ''
        if d > 2.0 * (route['route'].get('max_spacing_m') or 1e9):
            flag = '  <- long: interpolation was blocked by a no-waypoint zone'
        print('  {:<22s} -> {:<22s} {:6.2f} m{}'.format(a, b, d, flag))
    lengths = [d for _, _, d in segs]
    print('')
    print('  min {:.2f} m   mean {:.2f} m   max {:.2f} m'
          .format(min(lengths), sum(lengths) / len(lengths), max(lengths)))

    stale = []
    for robot, rel in OUTPUTS.items():
        text = build(robot, route, cfg)
        path = os.path.join(args.src, rel)
        if args.check:
            current = open(path).read() if os.path.isfile(path) else None
            if current != text:
                stale.append(rel)
            continue
        d = os.path.dirname(path)
        if not os.path.isdir(d):
            os.makedirs(d)
        with open(path, 'w') as f:
            f.write(text)
        print('\nwrote {}'.format(rel))

    if args.check:
        if stale:
            print('\nSTALE (re-run generate_waypoints.py):')
            for s in stale:
                print('  - {}'.format(s))
            return 1
        print('\nAll generated waypoint files are up to date.')
        return 0

    if args.plot:
        out = os.path.join(args.output_dir or os.path.dirname(args.route),
                           'route_mine')
        plot_route(route, cfg, out)
        print('\nroute figure: {}.png / .eps'.format(out))

    return 0


if __name__ == '__main__':
    sys.exit(main())
