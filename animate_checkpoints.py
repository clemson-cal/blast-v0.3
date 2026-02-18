#!/usr/bin/env python3
"""
Animation script for blast checkpoint files.

Creates an animation with three subplots showing density, Lorentz factor,
and pressure evolution over time from checkpoint files.
"""

import sys
sys.path.insert(0, 'mist/mist')

import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mist_archive as ma
import argparse


# Physical constants
GAMMA_LAW_INDEX = 4.0 / 3.0


def cons_to_prim(D, S, tau):
    """
    Convert conserved variables to primitive variables using Newton-Raphson.

    Args:
        D: conserved mass density
        S: conserved momentum density
        tau: conserved energy density (excluding rest mass)

    Returns:
        (rho, u, p): density, four-velocity, pressure
    """
    newton_iter_max = 50
    error_tolerance = 1e-12 * (np.abs(D) + np.abs(tau))
    gm = GAMMA_LAW_INDEX

    # Initial guess for pressure
    p = np.maximum(0.1 * tau, 1e-10)
    ss = S * S  # momentum squared

    for _ in range(newton_iter_max):
        et = tau + p + D
        b2 = np.minimum(ss / (et * et), 1.0 - 1e-10)
        w2 = 1.0 / (1.0 - b2)
        w = np.sqrt(w2)
        d = D / w
        de = (tau + D * (1.0 - w) + p * (1.0 - w2)) / w2
        dh = d + de + p
        a2 = gm * p / dh
        g = b2 * a2 - 1.0
        f = de * (gm - 1.0) - p

        # Check convergence
        if np.all(np.abs(f) < error_tolerance):
            break

        # Newton step
        p = p - f / g
        p = np.maximum(p, 1e-15)

    # Compute primitives
    et = tau + p + D
    b2 = np.minimum(ss / (et * et), 1.0 - 1e-10)
    w = np.sqrt(1.0 / (1.0 - b2))

    rho = D / w
    u = w * S / (tau + D + p)  # four-velocity

    return rho, u, p


def lorentz_factor(u):
    """Compute Lorentz factor from four-velocity."""
    return np.sqrt(1.0 + u * u)


def load_checkpoint(filename):
    """
    Load a checkpoint file and extract primitive variables.

    Returns:
        dict with keys: 'time', 'r', 'density', 'lorentz_factor', 'pressure',
                        'r_min', 'r_max' (domain edges)
    """
    chkpt = ma.load(filename)
    ps = chkpt['physics_state']
    initial = chkpt['initial']

    time = ps['time']
    geom = initial.get('geom', 'spherical')

    all_r = []
    all_rho = []
    all_gamma = []
    all_p = []

    # Track domain edges
    domain_r_min = float('inf')
    domain_r_max = float('-inf')

    for patch in ps['patches']:
        domain_r_min = min(domain_r_min, patch['r0'])
        domain_r_max = max(domain_r_max, patch['r1'])
        r0 = patch['r0']
        r1 = patch['r1']
        cons_data = patch['cons']

        start = cons_data['start'][0]
        num_zones = cons_data['shape'][0]
        data = cons_data['data']

        # Reshape interleaved data: [D0, S0, tau0, D1, S1, tau1, ...]
        cons = data.reshape(num_zones, 3)
        D = cons[:, 0]
        S = cons[:, 1]
        tau = cons[:, 2]

        # Compute cell radii
        face_radii = np.linspace(r0, r1, num_zones + 1)
        cell_radii = 0.5 * (face_radii[:-1] + face_radii[1:])

        # Compute cell volumes
        if geom == 'spherical':
            rl = face_radii[:-1]
            rr = face_radii[1:]
            cell_volumes = (4.0 * np.pi / 3.0) * (rr**3 - rl**3)
        else:  # planar
            cell_volumes = face_radii[1:] - face_radii[:-1]

        # Convert conserved (volume-integrated) to conserved (density)
        D_density = D / cell_volumes
        S_density = S / cell_volumes
        tau_density = tau / cell_volumes

        # Convert to primitives
        rho, u, p = cons_to_prim(D_density, S_density, tau_density)
        gamma = lorentz_factor(u)

        all_r.append(cell_radii)
        all_rho.append(rho)
        all_gamma.append(gamma)
        all_p.append(p)

    return {
        'time': time,
        'r': np.concatenate(all_r),
        'density': np.concatenate(all_rho),
        'lorentz_factor': np.concatenate(all_gamma),
        'pressure': np.concatenate(all_p),
        'r_min': domain_r_min,
        'r_max': domain_r_max
    }


def find_checkpoint_files(pattern='chkpt.*.dat'):
    """Find and sort checkpoint files by number."""
    files = glob.glob(pattern)
    # Sort by the numeric part
    def sort_key(f):
        # Extract number from filename like 'chkpt.0042.dat'
        parts = f.split('.')
        for p in parts:
            if p.isdigit():
                return int(p)
        return 0
    return sorted(files, key=sort_key)


def create_animation(checkpoint_files, output_file=None, interval=100, figsize=(6, 8)):
    """
    Create an animation from checkpoint files.

    Args:
        checkpoint_files: list of checkpoint file paths
        output_file: if provided, save animation to this file (e.g., 'animation.mp4')
        interval: milliseconds between frames
        figsize: figure size tuple
    """
    if not checkpoint_files:
        print("No checkpoint files found!")
        return

    print(f"Loading {len(checkpoint_files)} checkpoint files...")

    # Load all checkpoints
    checkpoints = []
    for f in checkpoint_files:
        try:
            checkpoints.append(load_checkpoint(f))
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}")

    if not checkpoints:
        print("No valid checkpoints loaded!")
        return

    print(f"Loaded {len(checkpoints)} checkpoints")

    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(hspace=0)

    # Initialize plots
    line_rho, = axes[0].plot([], [], 'b-', lw=1)
    line_gamma, = axes[1].plot([], [], 'r-', lw=1)
    line_p, = axes[2].plot([], [], 'g-', lw=1)

    # Configure axes (limits will be set dynamically)
    axes[0].set_ylabel(r'Density $\rho$')
    axes[0].set_yscale('log')

    axes[1].set_ylabel(r'Lorentz factor $\Gamma$')

    axes[2].set_ylabel(r'Pressure $p$')
    axes[2].set_xlabel(r'Radius $r$')

    # Configure ticks: inward direction, on all sides
    for ax in axes:
        ax.tick_params(axis='both', direction='in', top=True, right=True)
        ax.tick_params(axis='x', which='both', direction='in', top=True)

    # Title showing time
    title = axes[0].set_title('', fontsize=12)

    def init():
        line_rho.set_data([], [])
        line_gamma.set_data([], [])
        line_p.set_data([], [])
        title.set_text('')
        return line_rho, line_gamma, line_p, title

    def update(frame):
        c = checkpoints[frame]
        r = c['r']
        rho = c['density']
        gamma = c['lorentz_factor']
        p = c['pressure']

        line_rho.set_data(r, rho)
        line_gamma.set_data(r, gamma)
        line_p.set_data(r, p)
        title.set_text(f't = {c["time"]:.4f}')

        # Update x-limits to follow the domain
        for ax in axes:
            ax.set_xlim(c['r_min'], c['r_max'])

        # Update y-limits adaptively with padding
        # For log-scale density, use multiplicative padding
        rho_pos = rho[rho > 0]
        if len(rho_pos) > 0:
            axes[0].set_ylim(rho_pos.min() * 0.5, rho_pos.max() * 2.0)

        # For linear-scale Lorentz factor, use additive padding
        gamma_min, gamma_max = gamma.min(), gamma.max()
        gamma_range = gamma_max - gamma_min
        if gamma_range > 0:
            axes[1].set_ylim(gamma_min - 0.1 * gamma_range, gamma_max + 0.1 * gamma_range)
        else:
            axes[1].set_ylim(gamma_min * 0.9, gamma_max * 1.1)

        # For linear-scale pressure, use additive padding
        p_min, p_max = p.min(), p.max()
        p_range = p_max - p_min
        if p_range > 0:
            axes[2].set_ylim(p_min - 0.1 * p_range, p_max + 0.1 * p_range)
        else:
            axes[2].set_ylim(p_min * 0.9, p_max * 1.1)

        return line_rho, line_gamma, line_p, title

    # blit=False required for dynamic axis limits
    anim = FuncAnimation(
        fig, update, frames=len(checkpoints),
        init_func=init, blit=False, interval=interval
    )

    if output_file:
        print(f"Saving animation to {output_file}...")
        if output_file.endswith('.gif'):
            anim.save(output_file, writer='pillow', fps=1000/interval)
        else:
            anim.save(output_file, writer='ffmpeg', fps=1000/interval)
        print("Done!")
    else:
        plt.show()

    return anim


def main():
    parser = argparse.ArgumentParser(
        description='Animate checkpoint files from blast simulations'
    )
    parser.add_argument(
        'pattern', nargs='?', default='chkpt.*.dat',
        help='Glob pattern for checkpoint files (default: chkpt.*.dat)'
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Output file (e.g., animation.mp4 or animation.gif)'
    )
    parser.add_argument(
        '-i', '--interval', type=int, default=100,
        help='Milliseconds between frames (default: 100)'
    )
    parser.add_argument(
        '--figsize', type=float, nargs=2, default=[6, 8],
        help='Figure size in inches (default: 6 8)'
    )

    args = parser.parse_args()

    files = find_checkpoint_files(args.pattern)
    print(f"Found {len(files)} checkpoint files matching '{args.pattern}'")

    create_animation(
        files,
        output_file=args.output,
        interval=args.interval,
        figsize=tuple(args.figsize)
    )


if __name__ == '__main__':
    main()
