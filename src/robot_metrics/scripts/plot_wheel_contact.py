#!/usr/bin/env python3
"""Per-wheel ground contact during the step crossing - the figure that
replaces (or backs up) Fig. 2.

Fig. 2 as submitted came from an idealised MATLAB model of the suspension.  It
describes what the mechanism is supposed to do, not what the simulated robot
actually did, and the reviewer was right that the two need not agree.  This
script plots the measured quantity instead: the contact state and normal force
reported by a Gazebo contact sensor on each of the six wheels, taken from the
same run that produced the SLAM numbers.

WHAT IS PLOTTED
===============
Top panel   contact raster - a filled band per wheel wherever that wheel is
            touching the ground.  Gaps are real losses of contact.
Middle      normal force per wheel [N].  A wheel that is nominally in contact
            but carrying almost no load is visible here and not in the raster,
            which matters: the rocker-bogie's advantage is not just keeping six
            wheels down, it is keeping load on them.
Bottom      chassis pitch and roll over the same interval, so the contact
            pattern can be read against what the body did.

The shaded span marks the step obstacle, located from sim_config.yaml.

WHY THE INTERVAL IS CHOSEN THE WAY IT IS
========================================
The window is derived from ground truth: the samples where the robot is within
`--margin` metres of the step slab.  Falling back on "where the vertical
acceleration was biggest" would beg the question, since that is part of what is
being measured.

Usage:
    ./plot_wheel_contact.py --run ~/metrics_output/rocker_bogie/run01
    ./plot_wheel_contact.py --run ... --output_dir ~/paper_tables
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_io import load_metrics                       # noqa: E402

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.patches import Rectangle                  # noqa: E402

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
    'axes.grid': True, 'grid.color': '0.88', 'grid.linewidth': 0.4,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

# Row order top-to-bottom, and the colour of each side.
WHEEL_ORDER = [
    ('front_left', '#1f77b4'), ('middle_left', '#1f77b4'), ('rear_left', '#1f77b4'),
    ('front_right', '#d62728'), ('middle_right', '#d62728'), ('rear_right', '#d62728'),
]


def load_layout(config_path, robot):
    """link name -> physical wheel position, from sim_config.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    entry = cfg.get('robots', {}).get(robot, {})
    layout = entry.get('wheel_layout', {})
    step = cfg.get('step_obstacle', {})
    return layout, step


def find_step_window(df, step, margin):
    """Indices where the robot is near the step slab, from ground truth."""
    if not step or 'gt_x' not in df.columns:
        return None
    cx, cy = step['pose_xyzrpy'][0], step['pose_xyzrpy'][1]
    sx, sy = step['size_m'][0], step['size_m'][1]
    # The slab is rotated 1.57 rad about z, so its world footprint is
    # sy along x and sx along y.
    half_x, half_y = sy / 2.0, sx / 2.0

    near = ((np.abs(df['gt_x'].values - cx) <= half_x + margin) &
            (np.abs(df['gt_y'].values - cy) <= half_y + margin))
    if not near.any():
        return None
    idx = np.where(near)[0]
    return idx[0], idx[-1]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='a run<NN> directory')
    ap.add_argument('--robot', default=None,
                    help='robot key in sim_config.yaml (default: inferred)')
    ap.add_argument('--config',
                    default=os.path.join(here, '..', 'config', 'sim_config.yaml'))
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--margin', type=float, default=3.0,
                    help='metres of approach/exit to include around the step')
    ap.add_argument('--force_threshold', type=float, default=1.0,
                    help='N below which a wheel counts as unloaded')
    ap.add_argument('--full', action='store_true',
                    help='plot the whole run instead of the step window')
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run)
    robot = args.robot or os.path.basename(os.path.dirname(run_dir))
    out_dir = args.output_dir or run_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = load_metrics(os.path.join(run_dir, 'metrics.csv'))
    layout, step = load_layout(args.config, robot)

    present = [(c[len('contact_'):]) for c in df.columns
               if c.startswith('contact_')]
    if not present:
        print('No contact_* columns in {}.'.format(run_dir))
        print('This platform has no wheel contact sensors, or the run predates '
              'them. For the Rocker-Bogie, check that the contact sensors in '
              'ensamblajeurdf.gazebo are present and that waypoint_navigation '
              'passed wheel_links to the logger.')
        return 1

    # Order the wheels by physical position when the layout is known.
    pos_of = {link: layout.get(link, link) for link in present}
    ordered = []
    for name, colour in WHEEL_ORDER:
        for link, pos in pos_of.items():
            if pos == name:
                ordered.append((link, name, colour))
    for link in present:                       # anything unmapped, appended
        if not any(link == l for l, _, _ in ordered):
            ordered.append((link, pos_of[link], '0.4'))

    t = df['timestamp'].values
    t = t - t[0]

    window = None if args.full else find_step_window(df, step, args.margin)
    if window is None:
        if not args.full:
            print('Robot never came within {} m of the step slab; plotting the '
                  'whole run instead.'.format(args.margin))
        lo, hi = 0, len(df) - 1
    else:
        lo, hi = window
        print('Step window: samples {}..{}  (t = {:.2f}..{:.2f} s)'
              .format(lo, hi, t[lo], t[hi]))

    sl = slice(lo, hi + 1)
    tt = t[sl]

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.2), sharex=True,
                             gridspec_kw={'height_ratios': [1.5, 1.4, 1.0]})

    # ---- raster ----
    ax = axes[0]
    for row, (link, name, colour) in enumerate(ordered):
        c = df['contact_' + link].values[sl].astype(float)
        f = df['normal_force_' + link].values[sl]
        y = len(ordered) - 1 - row
        # A wheel touching but unloaded is drawn hollow: contact alone can be
        # a grazing touch that carries no weight.
        loaded = (c > 0.5) & (f >= args.force_threshold)
        touching = (c > 0.5) & (f < args.force_threshold)
        ax.fill_between(tt, y + 0.12, y + 0.88, where=loaded,
                        color=colour, linewidth=0, step='mid')
        ax.fill_between(tt, y + 0.12, y + 0.88, where=touching,
                        color=colour, alpha=0.25, linewidth=0, step='mid')
    ax.set_yticks([len(ordered) - 1 - i + 0.5 for i in range(len(ordered))])
    ax.set_yticklabels([n.replace('_', '-') for _, n, _ in ordered])
    ax.set_ylim(0, len(ordered))
    ax.set_ylabel('Wheel contact')
    ax.grid(True, axis='x')
    ax.set_title('Solid = in contact and loaded (>= {:.0f} N); '
                 'pale = touching but unloaded'.format(args.force_threshold),
                 fontsize=7, pad=4)

    # ---- normal force ----
    ax = axes[1]
    for link, name, colour in ordered:
        f = df['normal_force_' + link].values[sl]
        ls = {'front': '-', 'middle': '--', 'rear': '-.'}.get(
            name.split('_')[0], '-')
        ax.plot(tt, f, ls=ls, color=colour, linewidth=0.9,
                label=name.replace('_', '-'))
    ax.set_ylabel('Normal force (N)')
    ax.legend(ncol=3, fontsize=6.5)

    # ---- attitude ----
    ax = axes[2]
    ax.plot(tt, np.degrees(df['pitch'].values[sl]), '-', color='#333333',
            linewidth=1.0, label='pitch')
    ax.plot(tt, np.degrees(df['roll'].values[sl]), '--', color='#888888',
            linewidth=1.0, label='roll')
    ax.set_ylabel('Attitude (deg)')
    ax.set_xlabel('Time (s)')
    ax.legend(ncol=2)

    fig.tight_layout(h_pad=0.4)
    base = os.path.join(out_dir, 'wheel_contact_step_{}'.format(robot))
    for ext in ('eps', 'png'):
        fig.savefig('{}.{}'.format(base, ext), dpi=200)
    plt.close(fig)

    # ---- numbers to quote in the text ----
    print('')
    print('Contact statistics over the plotted interval ({:.2f} s):'
          .format(tt[-1] - tt[0]))
    print('{:<14s}{:>12s}{:>14s}{:>14s}{:>12s}'.format(
        'wheel', 'contact[%]', 'loaded[%]', 'mean F[N]', 'peak F[N]'))
    rows = []
    for link, name, _ in ordered:
        c = df['contact_' + link].values[sl].astype(float)
        f = df['normal_force_' + link].values[sl]
        loaded = ((c > 0.5) & (f >= args.force_threshold)).mean() * 100.0
        rows.append((name, c.mean() * 100.0, loaded, f.mean(), f.max()))
        print('{:<14s}{:>12.1f}{:>14.1f}{:>14.2f}{:>12.2f}'.format(*rows[-1]))

    n_lost = sum(1 for _, c, _, _, _ in rows if c < 99.9)
    print('')
    print('Wheels that lost contact at least once: {} of {}'
          .format(n_lost, len(rows)))
    print('Figure written to {}.eps / .png'.format(base))
    return 0


if __name__ == '__main__':
    sys.exit(main())
