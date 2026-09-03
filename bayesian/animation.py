r"""
Dynamic Animation Module for Parallel Two-Chain MH Experiment (1D Heat Equation)
================================================================================
Generates animated GIF visualization from ACTUAL saved chain trajectory data:
- Common initial starting point \alpha_0
- Exact Chain (Blue) vs Approximate PINN Chain (Red) advancing over iterations
- Continuous distance evolution plot d_t vs t alongside trajectory
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from typing import Optional, Union, Dict, Any, Tuple

from .two_chain_sampler import TwoChainMCMCResult


def generate_two_chain_animation(
    result_or_npz: Union[TwoChainMCMCResult, str],
    true_alpha: Optional[float] = 0.5,
    output_path: str = "results/two_chain_evolution.gif",
    n_frames: int = 150,
    fps: int = 15,
    dpi: int = 120
) -> str:
    """Generates dynamic GIF animation from real chain data."""
    if isinstance(result_or_npz, TwoChainMCMCResult):
        alpha_E = result_or_npz.get_alpha_E_array()
        alpha_A = result_or_npz.get_alpha_A_array()
        distance = result_or_npz.get_distance_array()
        alpha_0 = result_or_npz.alpha_0
        n_total = len(result_or_npz)
    elif isinstance(result_or_npz, str):
        with np.load(result_or_npz) as data:
            alpha_E = np.array(data["alpha_E"], dtype=float)
            alpha_A = np.array(data["alpha_A"], dtype=float)
            distance = np.array(data["distance"], dtype=float)
            alpha_0 = float(data["alpha_0"])
            n_total = len(alpha_E)
    else:
        raise ValueError("Unsupported input format for generate_two_chain_animation.")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    frame_indices = np.linspace(0, n_total - 1, n_frames, dtype=int)

    fig, (ax_traj, ax_dist) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(r"Parallel Two-Chain MCMC Evolution: $\alpha_E$ vs $\alpha_A$", fontsize=12, fontweight='bold')

    # Domain limits
    a_min = min(np.min(alpha_E), np.min(alpha_A), alpha_0, (true_alpha if true_alpha else 0.5)) * 0.9
    a_max = max(np.max(alpha_E), np.max(alpha_A), alpha_0, (true_alpha if true_alpha else 0.5)) * 1.1
    max_dist = max(np.max(distance) * 1.15, 0.05)

    # Panel 1: Trajectories
    ax_traj.set_xlim(0, n_total)
    ax_traj.set_ylim(max(a_min, 0.0), a_max)
    ax_traj.set_xlabel("Iteration $t$", fontsize=10)
    ax_traj.set_ylabel(r"Thermal Diffusivity $\alpha$", fontsize=10)
    ax_traj.grid(True, alpha=0.3)
    if true_alpha:
        ax_traj.axhline(true_alpha, color="black", linestyle=":", label=rf"True $\alpha^*={true_alpha:.3f}$")

    line_E, = ax_traj.plot([], [], color="#1f77b4", lw=1.2, alpha=0.8, label=r"Exact Chain $\alpha_E$")
    line_A, = ax_traj.plot([], [], color="#d62728", lw=1.2, alpha=0.8, linestyle="--", label=r"PINN Chain $\alpha_A$")
    head_E = ax_traj.scatter([], [], color="#1f77b4", s=70, edgecolors="black", zorder=8)
    head_A = ax_traj.scatter([], [], color="#d62728", s=70, edgecolors="black", zorder=8)
    ax_traj.legend(loc="upper right", fontsize=8)

    # Panel 2: Distance
    ax_dist.set_xlim(0, n_total)
    ax_dist.set_ylim(0, max_dist)
    ax_dist.set_xlabel("Iteration $t$", fontsize=10)
    ax_dist.set_ylabel(r"Distance $d_t = |\alpha_E^{(t)} - \alpha_A^{(t)}|$", fontsize=10)
    ax_dist.grid(True, alpha=0.3)
    line_dist, = ax_dist.plot([], [], color="#2ca02c", lw=1.5, label=r"$d_t$ Trace")
    head_dist = ax_dist.scatter([], [], color="#2ca02c", s=60, edgecolors="black", zorder=8)
    ax_dist.legend(loc="upper right", fontsize=8)

    def init():
        line_E.set_data([], [])
        line_A.set_data([], [])
        head_E.set_offsets(np.empty((0, 2)))
        head_A.set_offsets(np.empty((0, 2)))
        line_dist.set_data([], [])
        head_dist.set_offsets(np.empty((0, 2)))
        return line_E, line_A, head_E, head_A, line_dist, head_dist

    def animate(frame_num):
        idx = frame_indices[frame_num]
        t_arr = np.arange(idx + 1)
        line_E.set_data(t_arr, alpha_E[:idx + 1])
        line_A.set_data(t_arr, alpha_A[:idx + 1])
        head_E.set_offsets([[idx, alpha_E[idx]]])
        head_A.set_offsets([[idx, alpha_A[idx]]])

        line_dist.set_data(t_arr, distance[:idx + 1])
        head_dist.set_offsets([[idx, distance[idx]]])
        return line_E, line_A, head_E, head_A, line_dist, head_dist

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1000 // fps, blit=True)
    anim.save(output_path, writer="pillow", fps=fps, dpi=dpi)
    plt.close()
    return output_path
