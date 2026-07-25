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


def wrap_deg(a):
    while a > 180.0:
        a -= 360.0
    while a <= -180.0:
        a += 360.0
    return a


def to_map_frame(wp, spawn_x, spawn_y, spawn_yaw_rad):
    dx = wp['x'] - spawn_x
    dy = wp['y'] - spawn_y
    c, s = math.cos(-spawn_yaw_rad), math.sin(-spawn_yaw_rad)
    return {
        'name': wp['name'],
        'x': round(c * dx - s * dy, 3),
        'y': round(s * dx + c * dy, 3),
        'yaw': round(wrap_deg(wp['yaw'] - math.degrees(spawn_yaw_rad)), 2),
    }


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
# Transform      : p_map = R(-yaw_spawn) * (p_world - p_spawn)
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

    waypoints = [to_map_frame(w, sx, sy, syaw) for w in route['route']['waypoints']]

    nav = dict(route['navigation_params'])
    limits = route.get('platform_limits', {}).get(robot, {})
    nav.update({k: v for k, v in limits.items()})

    doc = {'waypoints': waypoints, 'navigation_params': nav}
    header = HEADER.format(
        robot=robot, sx=sx, sy=sy, syaw_deg=math.degrees(syaw),
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

    wps = route['route']['waypoints']
    print('Route "{}": {} waypoints, {:.1f} m in world coordinates'
          .format(route['route']['name'], len(wps), route_length(wps)))
    print('')
    print('Segment spacing (target 8-12 m, no waypoint on an obstacle):')
    segs = segment_report(wps)
    for a, b, d in segs:
        flag = ''
        if d < 4.0:
            flag = '  <- short, the follower may not settle between them'
        elif d > 16.0:
            flag = '  <- long, a curved corridor may not be represented'
        print('  {:<18s} -> {:<18s} {:6.2f} m{}'.format(a, b, d, flag))
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
