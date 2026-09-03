r"""
Two-Chain Visualization & Publication Figures Module (1D Heat Equation)
========================================================================
Generates publication-quality figures for the parallel two-chain Bayesian experiment:
1. Two-Chain Parameter Trajectory (\alpha_E(t) vs \alpha_A(t))
2. Observational Distance Evolution (d_t = |\alpha_E^{(t)} - \alpha_A^{(t)}|)
3. 1D Stationary Posterior Comparison (Exact vs PINN)
4. Log-Posterior and Data Misfit Trajectories
5. Multi-panel Master Diagnostic Summary Dashboard
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Optional, List, Tuple

from .two_chain_sampler import TwoChainMCMCResult


def plot_alpha_trajectories(
    result: TwoChainMCMCResult,
    burn_in: int = 1000,
    true_alpha: Optional[float] = None,
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """Plots parallel trace trajectories for alpha_E(t) and alpha_A(t)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    alpha_E = result.get_alpha_E_array()
    alpha_A = result.get_alpha_A_array()
    iters = np.arange(1, len(alpha_E) + 1)

    ax.plot(iters, alpha_E, color="#1f77b4", alpha=0.85, linewidth=1.2, label=r"Chain E (Exact Forward $\mathcal{F}_{\mathrm{exact}}$)")
    ax.plot(iters, alpha_A, color="#d62728", alpha=0.85, linewidth=1.2, label=r"Chain A (PINN Surrogate $\mathcal{F}_{\mathrm{PINN}}$)")

    if true_alpha is not None:
        ax.axhline(true_alpha, color="black", linestyle="--", linewidth=1.5, label=rf"Ground Truth $\alpha^* = {true_alpha:.3f}$")

    ax.axvline(burn_in, color="gray", linestyle=":", linewidth=1.5, label=f"Burn-In Boundary ($t={burn_in}$)")
    ax.scatter([1], [result.alpha_0], color="purple", s=80, zorder=5, label=rf"Common Start $\alpha_0 = {result.alpha_0:.3f}$")

    ax.set_xlabel("MCMC Iteration $t$", fontsize=11)
    ax.set_ylabel(r"Thermal Diffusivity $\alpha$", fontsize=11)
    ax.set_title(r"Parallel MCMC Trajectories: $\alpha_E^{(t)}$ vs $\alpha_A^{(t)}$", fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_distance_trajectory(
    result: TwoChainMCMCResult,
    burn_in: int = 1000,
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """Plots observational trajectory distance d_t = |alpha_E(t) - alpha_A(t)|."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    dist = result.get_distance_array()
    iters = np.arange(1, len(dist) + 1)

    # Linear scale
    ax1.plot(iters, dist, color="#2ca02c", linewidth=1.2, label=r"Distance $d_t = |\alpha_E^{(t)} - \alpha_A^{(t)}|$")
    ax1.axvline(burn_in, color="gray", linestyle=":", linewidth=1.5, label=f"Burn-In ($t={burn_in}$)")
    ax1.set_ylabel("Linear $d_t$", fontsize=11)
    ax1.set_title(r"Observational Trajectory Distance Evolution $d_t$", fontsize=12, pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Log scale
    dist_safe = np.maximum(dist, 1e-12)
    ax2.semilogy(iters, dist_safe, color="#9467bd", linewidth=1.2, label=r"$\log_{10}(d_t)$")
    ax2.axvline(burn_in, color="gray", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("MCMC Iteration $t$", fontsize=11)
    ax2.set_ylabel("Log Scale $d_t$", fontsize=11)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_posterior_comparison(
    result: TwoChainMCMCResult,
    burn_in: int = 1000,
    true_alpha: Optional[float] = None,
    dense_grid_data: Optional[Dict[str, np.ndarray]] = None,
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """Plots post-burn-in stationary posterior sample histograms and KDEs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    alpha_E_post, alpha_A_post = result.get_post_burnin_samples(burn_in)

    # Histograms
    bins = np.linspace(
        min(np.min(alpha_E_post), np.min(alpha_A_post)) - 0.05,
        max(np.max(alpha_E_post), np.max(alpha_A_post)) + 0.05,
        40
    )
    ax.hist(alpha_E_post, bins=bins, density=True, alpha=0.45, color="#1f77b4", label="Chain E Post-Burn-in Samples")
    ax.hist(alpha_A_post, bins=bins, density=True, alpha=0.45, color="#d62728", label="Chain A Post-Burn-in Samples")

    # KDE curves
    from scipy.stats import gaussian_kde
    kde_E = gaussian_kde(alpha_E_post)
    kde_A = gaussian_kde(alpha_A_post)
    grid_eval = np.linspace(bins[0], bins[-1], 200)
    ax.plot(grid_eval, kde_E(grid_eval), color="#1f77b4", linewidth=2.2, label=r"Exact Target KDE $\hat{\pi}_E(\alpha \mid y)$")
    ax.plot(grid_eval, kde_A(grid_eval), color="#d62728", linewidth=2.2, linestyle="--", label=r"PINN Target KDE $\hat{\pi}_A(\alpha \mid y)$")

    if dense_grid_data is not None:
        ax.plot(
            dense_grid_data["alpha_grid"],
            dense_grid_data["normalized_pdf"],
            color="black",
            linewidth=1.8,
            linestyle=":",
            label=r"Exact Target (Dense Numerical Grid)"
        )

    if true_alpha is not None:
        ax.axvline(true_alpha, color="black", linestyle="-.", linewidth=1.5, label=rf"True $\alpha^* = {true_alpha:.3f}$")

    ax.set_xlabel(r"Thermal Diffusivity $\alpha$", fontsize=11)
    ax.set_ylabel("Posterior Probability Density", fontsize=11)
    ax.set_title(r"Post-Burn-in Posterior Density Comparison: $\pi_E(\alpha \mid y)$ vs $\pi_A(\alpha \mid y)$", fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_master_dashboard(
    result: TwoChainMCMCResult,
    burn_in: int = 1000,
    true_alpha: Optional[float] = None,
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """Generates a 4-panel master summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    alpha_E = result.get_alpha_E_array()
    alpha_A = result.get_alpha_A_array()
    dist = result.get_distance_array()
    log_p_E = result.get_log_posterior_E_array()
    log_p_A = result.get_log_posterior_A_array()
    iters = np.arange(1, len(alpha_E) + 1)
    alpha_E_post, alpha_A_post = result.get_post_burnin_samples(burn_in)

    # Panel 1: Trajectories
    axes[0, 0].plot(iters, alpha_E, color="#1f77b4", alpha=0.8, linewidth=1.2, label=r"Chain E ($\mathcal{F}_{\mathrm{exact}}$)")
    axes[0, 0].plot(iters, alpha_A, color="#d62728", alpha=0.8, linewidth=1.2, label=r"Chain A ($\mathcal{F}_{\mathrm{PINN}}$)")
    if true_alpha is not None:
        axes[0, 0].axhline(true_alpha, color="black", linestyle="--", label=rf"True $\alpha^* = {true_alpha:.3f}$")
    axes[0, 0].axvline(burn_in, color="gray", linestyle=":", label="Burn-in")
    axes[0, 0].set_title(r"(a) Parallel MCMC Trajectories $\alpha^{(t)}$", fontsize=11)
    axes[0, 0].set_xlabel("Iteration $t$")
    axes[0, 0].set_ylabel(r"$\alpha$")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right")

    # Panel 2: Distance Metric
    axes[0, 1].plot(iters, dist, color="#2ca02c", linewidth=1.2, label=r"$d_t = |\alpha_E^{(t)} - \alpha_A^{(t)}|$")
    axes[0, 1].axvline(burn_in, color="gray", linestyle=":")
    axes[0, 1].set_title(r"(b) Trajectory Distance Evolution $d_t$", fontsize=11)
    axes[0, 1].set_xlabel("Iteration $t$")
    axes[0, 1].set_ylabel(r"$d_t$")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc="upper right")

    # Panel 3: Log Posterior Traces
    axes[1, 0].plot(iters, log_p_E, color="#1f77b4", alpha=0.8, linewidth=1.0, label=r"$\ln \pi_E(\alpha_E \mid y)$")
    axes[1, 0].plot(iters, log_p_A, color="#d62728", alpha=0.8, linewidth=1.0, label=r"$\ln \pi_A(\alpha_A \mid y)$")
    axes[1, 0].axvline(burn_in, color="gray", linestyle=":")
    axes[1, 0].set_title(r"(c) Unnormalized Log Posterior Traces", fontsize=11)
    axes[1, 0].set_xlabel("Iteration $t$")
    axes[1, 0].set_ylabel(r"$\ln \pi$")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="lower right")

    # Panel 4: Stationary Posterior KDEs
    from scipy.stats import gaussian_kde
    kde_E = gaussian_kde(alpha_E_post)
    kde_A = gaussian_kde(alpha_A_post)
    grid_eval = np.linspace(
        min(np.min(alpha_E_post), np.min(alpha_A_post)) - 0.05,
        max(np.max(alpha_E_post), np.max(alpha_A_post)) + 0.05,
        200
    )
    axes[1, 1].plot(grid_eval, kde_E(grid_eval), color="#1f77b4", linewidth=2.2, label=r"Exact Target $\pi_E$")
    axes[1, 1].plot(grid_eval, kde_A(grid_eval), color="#d62728", linewidth=2.2, linestyle="--", label=r"PINN Target $\pi_A$")
    if true_alpha is not None:
        axes[1, 1].axvline(true_alpha, color="black", linestyle="-.", label=rf"True $\alpha^* = {true_alpha:.3f}$")
    axes[1, 1].set_title(r"(d) Post-Burn-in Stationary Posterior Distributions", fontsize=11)
    axes[1, 1].set_xlabel(r"$\alpha$")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="upper right")

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig


def generate_all_two_chain_plots(
    result: TwoChainMCMCResult,
    output_dir: str,
    burn_in: int = 1000,
    true_alpha: Optional[float] = None,
    dense_grid_data: Optional[Dict[str, np.ndarray]] = None,
    dpi: int = 300
) -> Dict[str, str]:
    """Generates and saves all primary two-chain publication figures."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = "control_" if "control" in result.experiment_type.lower() else ""

    p1 = os.path.join(output_dir, f"{prefix}alpha_trajectories.png")
    plot_alpha_trajectories(result, burn_in=burn_in, true_alpha=true_alpha, output_path=p1, dpi=dpi)
    plt.close()

    p2 = os.path.join(output_dir, f"{prefix}distance_trajectory.png")
    plot_distance_trajectory(result, burn_in=burn_in, output_path=p2, dpi=dpi)
    plt.close()

    p3 = os.path.join(output_dir, f"{prefix}posterior_comparison.png")
    plot_posterior_comparison(result, burn_in=burn_in, true_alpha=true_alpha, dense_grid_data=dense_grid_data, output_path=p3, dpi=dpi)
    plt.close()

    p4 = os.path.join(output_dir, f"{prefix}master_dashboard.png")
    plot_master_dashboard(result, burn_in=burn_in, true_alpha=true_alpha, output_path=p4, dpi=dpi)
    plt.close()

    return {
        "alpha_trajectories": p1,
        "distance_trajectory": p2,
        "posterior_comparison": p3,
        "master_dashboard": p4
    }
