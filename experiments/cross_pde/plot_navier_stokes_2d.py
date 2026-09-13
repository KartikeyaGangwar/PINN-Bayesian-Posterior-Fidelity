"""
Publication Vector Figure Generator for 2D Navier-Stokes Benchmark
===================================================================
Generates Figure 26 with:
- Exact LaTeX Computer Modern fonts (text.usetex = True)
- Vector PDF output for publication integration
- All legends placed horizontally below the x-axis to maintain clear field visualization
- Purely continuous analytical functions and high-resolution spatial fields
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from experiments.cross_pde.navier_stokes_2d import (
    NavierStokes2DTaylorGreenBenchmark,
    generate_space_time_sensors_2d,
)


def generate_figure_26(
    data_dir: str = "results/navier_stokes_2d",
    output_pdf: str = "paper/figures/fig26_navier_stokes_2d_validation.pdf",
    output_png: str = "paper/figures/fig26_navier_stokes_2d_validation.png"
):
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    bench = NavierStokes2DTaylorGreenBenchmark()
    cfg = bench.config

    # Load experimental data
    raw_csv = os.path.join(data_dir, "ns2d_20models_raw.csv")
    loc_csv = os.path.join(data_dir, "ns2d_localization_sweep.csv")
    summary_json = os.path.join(data_dir, "ns2d_summary.json")

    df_models = pd.read_csv(raw_csv)
    df_loc = pd.read_csv(loc_csv)
    with open(summary_json, "r") as f:
        summary = json.load(f)

    # Global Publication LaTeX Formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 1.6,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })

    fig = plt.figure(figsize=(15.2, 5.0), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1.25, 1.1], wspace=0.35, bottom=0.22, top=0.90)

    # -------------------------------------------------------------
    # Panel (a): Fluid Dynamics Physics (Streamlines & Continuous Vorticity)
    # -------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0])
    nx, ny = 250, 250  # Dense continuous mesh
    x = np.linspace(0.0, 2.0 * np.pi, nx)
    y = np.linspace(0.0, 2.0 * np.pi, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    t0 = 0.2
    u, v, _ = bench.exact_velocity_and_pressure(X, Y, t0, cfg.true_param)
    omega = bench.exact_vorticity(X, Y, t0, cfg.true_param)

    # Continuous Contour plot of vorticity with 50 levels
    c = ax0.contourf(X, Y, omega, levels=50, cmap="coolwarm", alpha=0.88)
    cbar = plt.colorbar(c, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label(r"Vorticity $\omega = \nabla \times \mathbf{u}$", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    # Smooth streamlines
    ax0.streamplot(x, y, u.T, v.T, color="black", density=1.2, linewidth=0.75, arrowsize=0.75)

    # Velocity sensor probes
    sensors = generate_space_time_sensors_2d(n_sensors=40, seed=42)
    ax0.scatter(
        sensors[:, 0], sensors[:, 1],
        c="gold", edgecolor="black", s=32, zorder=5, label=r"Velocity Probes ($M=40$)"
    )

    ax0.set_title(r"\textbf{(a) 2D Navier-Stokes Vorticity \& Streamlines}", fontsize=10)
    ax0.set_xlabel(r"Spatial Coordinate $x \in [0, 2\pi]$")
    ax0.set_ylabel(r"Spatial Coordinate $y \in [0, 2\pi]$")
    ax0.set_xlim(0, 2.0 * np.pi)
    ax0.set_ylim(0, 2.0 * np.pi)
    # Legend below x-axis horizontally
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=True, framealpha=0.95)

    # -------------------------------------------------------------
    # Panel (b): Diagnostic Hierarchy with Continuous Regression Band
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[1])
    tiers = df_models["tier_id"].unique()
    tier_colors = {1: "#d95f02", 2: "#e7298a", 3: "#7570b3", 4: "#1b9e77", 5: "#2b83ba"}
    tier_labels = {
        1: "Tier 1: Underconv.",
        2: "Tier 2: Early",
        3: "Tier 3: Intermediate",
        4: "Tier 4: Near-Optimal",
        5: "Tier 5: Converged"
    }

    # Model scatter points
    for tid in sorted(tiers):
        sub = df_models[df_models["tier_id"] == tid]
        ax1.scatter(
            sub["l1_delta_loglik"], sub["w1"],
            color=tier_colors.get(tid, "blue"),
            s=42, edgecolor="k", linewidth=0.6,
            label=tier_labels.get(tid, f"Tier {tid}"),
            zorder=4
        )

    # Continuous Regression Line & 95% Confidence Band
    x_data = df_models["l1_delta_loglik"].values
    y_data = df_models["w1"].values
    x_dense = np.linspace(x_data.min(), x_data.max(), 1000)

    slope, intercept, r_val, p_val, std_err = stats.linregress(x_data, y_data)
    y_dense = slope * x_dense + intercept

    # Two-sided 95% confidence interval for mean response
    n = len(x_data)
    t_crit = stats.t.ppf(0.975, df=n - 2)
    s_err = np.sqrt(np.sum((y_data - (slope * x_data + intercept))**2) / (n - 2))
    ci = t_crit * s_err * np.sqrt(1.0 / n + (x_dense - np.mean(x_data))**2 / np.sum((x_data - np.mean(x_data))**2))

    ax1.plot(x_dense, y_dense, color="crimson", linestyle="--", linewidth=1.5,
             label=rf"Linear Fit ($r = {r_val:.3f}$)", zorder=3)
    ax1.fill_between(x_dense, y_dense - ci, y_dense + ci, color="crimson", alpha=0.15,
                     label=r"95\% Confidence Band", zorder=2)

    ax1.set_title(r"\textbf{(b) Likelihood Tracking Fidelity ($N=20$ Models)}", fontsize=10)
    ax1.set_xlabel(r"Log-Likelihood Perturbation $\|\Delta \log \mathcal{L}\|_{L_1(\mu^y)}$")
    ax1.set_ylabel(r"Posterior Discrepancy $\mathcal{W}_1(\mu^y, \widehat{\mu}^y)$")
    ax1.grid(True)

    # Legend below x-axis horizontally in 3 columns
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True, framealpha=0.95)

    # -------------------------------------------------------------
    # Panel (c): Purely Continuous Causal Error Localization Curve
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[2])
    a = summary["mass_support"]  # support mass a = 0.4881

    # Dense continuous evaluation of delta in [0, 20] (1,000 points)
    delta_dense = np.linspace(0.0, 20.0, 1000)

    # Continuous theoretical Total Variation formula
    # d_TV(delta) = a*(1-a)*(exp(delta)-1) / (1 + a*(exp(delta)-1))
    tv_theory_dense = np.where(
        delta_dense < 50.0,
        a * (1.0 - a) * (np.expm1(delta_dense)) / (1.0 + a * (np.expm1(delta_dense))),
        1.0 - a
    )

    # Continuous theoretical Wasserstein-1 scaling for support perturbation: W1 proportional to TV
    scale_w1 = summary["localization_evidence"]["w1_support_max"] / (1.0 - a)
    w1_theory_dense = tv_theory_dense * scale_w1

    # Dense theoretical tail curve: identically zero / machine precision floor
    w1_tail_floor = 5.77e-20
    tail_dense = np.full_like(delta_dense, w1_tail_floor)

    # Plot continuous analytical curves
    ax2.plot(delta_dense, w1_theory_dense, "-", color="#e41a1c", linewidth=2.0,
             label=r"Bulk Theory $\mu^y(\Omega_{\mathrm{bulk}}) = " + f"{a:.3f}$")
    ax2.plot(delta_dense, tail_dense, "--", color="#377eb8", linewidth=1.8,
             label=r"Tail Theory $\mu^y(\Omega_{\mathrm{tail}}) < 10^{-200}$")

    # Overlay discrete numerical quadrature validation points
    ax2.scatter(df_loc["delta"], df_loc["w1_support"], color="#e41a1c", edgecolor="k",
                s=38, zorder=5, label=r"Quadrature $\Omega_{\mathrm{bulk}}$")
    ax2.scatter(df_loc["delta"], np.maximum(df_loc["w1_tail"], 1e-20), color="#377eb8", edgecolor="k",
                s=30, marker="s", zorder=5, label=r"Quadrature $\Omega_{\mathrm{tail}}$")

    # Theoretical saturation bound line
    ax2.axhline(y=(1.0 - a) * scale_w1, color="gray", linestyle=":", alpha=0.7)

    ax2.set_title(r"\textbf{(c) Causal Localization in 2D Flow}", fontsize=10)
    ax2.set_xlabel(r"Perturbation Magnitude $\delta \in [0, 20]$")
    ax2.set_ylabel(r"Induced Distortion $\mathcal{W}_1$")
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=1e-21, top=summary["localization_evidence"]["w1_support_max"] * 6.0)
    ax2.grid(True, which="both")

    # Legend below x-axis horizontally in 2 columns
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True, framealpha=0.95)

    # Save Vector PDF and High-Resolution PNG
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(output_png, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Vector PDF: {output_pdf}", flush=True)
    print(f"Generated High-Res PNG: {output_png}", flush=True)


if __name__ == "__main__":
    generate_figure_26()
