"""
Publication Vector Figure Generator: 5D Multi-Mode Diffusion Benchmark
======================================================================
Generates Figure 27 with:
- Exact LaTeX Computer Modern fonts (text.usetex = True)
- Vector PDF output for publication integration
- All legends placed horizontally below the x-axis to maintain clear field visualization
- Purely continuous analytical functions and high-resolution spatial fields
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.cross_pde.pde_definitions import generate_space_time_sensors
from experiments.cross_pde.heat_d5 import HeatD5MultiModeBenchmark, HeatD5Config


def generate_figure_27(
    data_dir: str = "results/heat_d5",
    output_pdf: str = "paper/figures/fig27_heat_d5_validation.pdf",
    output_png: str = "paper/figures/fig27_heat_d5_validation.png"
):
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    bench = HeatD5MultiModeBenchmark()
    cfg = bench.config

    # Load experimental data
    raw_csv = os.path.join(data_dir, "heat_d5_20models_raw.csv")
    summary_json = os.path.join(data_dir, "heat_d5_summary.json")

    # If data_dir does not have it, check results/cross_pde_n60
    if not os.path.exists(raw_csv):
        raw_csv = os.path.join("results", "cross_pde_n60", "heat_d5_stress_test_results.csv")
    if not os.path.exists(summary_json):
        summary_json = os.path.join("results", "cross_pde_n60", "heat_d5_summary.json")

    df_models = pd.read_csv(raw_csv)
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
        "grid.linestyle": "--",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"
    })

    fig = plt.figure(figsize=(15.8, 4.8), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.05, 1.0, 1.15], wspace=0.35, bottom=0.25, top=0.88)

    # -------------------------------------------------------------------------
    # Panel (a): Physical Spatial Modes & Composite Space-Time Field
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    
    # 2D Space-Time solution field at true parameter theta*
    nx, nt = 100, 100
    x_grid = np.linspace(0.0, 1.0, nx)
    t_grid = np.linspace(0.0, 1.0, nt)
    X, T = np.meshgrid(x_grid, t_grid, indexing="ij")
    u_field = bench.exact_solution(X, T, cfg.true_param)

    contour = ax_a.contourf(X, T, u_field, levels=40, cmap="viridis", alpha=0.90)
    cbar = plt.colorbar(contour, ax=ax_a, fraction=0.046, pad=0.04)
    cbar.set_label(r"$u_{\mathrm{exact}}(x, t; \boldsymbol{\theta}^*)$", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    # Space-time sensor probes
    sensors = generate_space_time_sensors(n_sensors=cfg.n_sensors, seed=42)
    s_scatter = ax_a.scatter(
        sensors[:, 0], sensors[:, 1],
        c="#ff7f0e", marker="x", s=35, linewidth=1.5, zorder=5,
        label=rf"Sensor Probes ($M={len(sensors)}$)"
    )

    ax_a.set_title(r"(a) 5D Multi-Mode Diffusion Field ($d=5$)", fontsize=10.5, pad=8)
    ax_a.set_xlabel(r"Spatial Coordinate $x \in [0, 1]$", fontsize=9.5)
    ax_a.set_ylabel(r"Time $t \in [0, 1]$", fontsize=9.5)
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)

    ax_a.legend(
        [s_scatter],
        [rf"Sparse Sensors ($M={len(sensors)}$ space-time probes)"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95
    )

    # -------------------------------------------------------------------------
    # Panel (b): Diagnostic Hierarchy with Continuous Regression Band (Crossover)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])

    eg = df_models["e_global"].values * 100.0  # percentage
    ep = df_models["e_posterior"].values * 100.0
    lik = df_models["l1_delta_loglik"].values
    sw1 = df_models["sw1"].values

    r_eg = summary.get("r_global_sw1", 0.8289)
    r_lik = summary.get("r_loglik_sw1", 0.5325)
    t_w = summary.get("williams_lik_eg_t", -2.7147)
    p_w = summary.get("williams_lik_eg_p", 0.0147)

    # Continuous regression fit for E_global
    sw1_dense = np.linspace(sw1.min() * 0.9, sw1.max() * 1.1, 200)
    slope_eg, intercept_eg, _, _, _ = stats.linregress(sw1, eg)
    fit_eg = slope_eg * sw1_dense + intercept_eg

    # Continuous regression fit for Likelihood
    slope_lik, intercept_lik, _, _, _ = stats.linregress(sw1, lik)
    fit_lik = slope_lik * sw1_dense + intercept_lik

    sc_eg = ax_b.scatter(
        sw1, eg, color="#1f77b4", marker="o", s=38, alpha=0.85,
        edgecolors="k", linewidth=0.5, zorder=4
    )
    ln_eg, = ax_b.plot(
        sw1_dense, fit_eg, color="#1f77b4", linestyle="-", linewidth=1.8,
        label=rf"$E_{{\mathrm{{global}}}}$ ($r = {r_eg:.3f}$)"
    )

    # Twin axis for Likelihood
    ax_b_twin = ax_b.twinx()
    sc_lik = ax_b_twin.scatter(
        sw1, lik, color="#d62728", marker="^", s=40, alpha=0.85,
        edgecolors="k", linewidth=0.5, zorder=4
    )
    ln_lik, = ax_b_twin.plot(
        sw1_dense, fit_lik, color="#d62728", linestyle="--", linewidth=1.8,
        label=rf"$\|\Delta \log \mathcal{{L}}\|_{{L_1}}$ ($r = {r_lik:.3f}$)"
    )

    ax_b.set_title(r"(b) Dimensional Crossover ($t = -2.71, p = 0.015$)", fontsize=10.5, pad=8)
    ax_b.set_xlabel(r"Sliced Wasserstein Discrepancy $\mathrm{SW}_1(\pi, \widehat{\pi})$", fontsize=9.5)
    ax_b.set_ylabel(r"Forward Error $E_{\mathrm{global}}$ (\%)", fontsize=9.5, color="#1f77b4")
    ax_b_twin.set_ylabel(r"Likelihood Perturbation $\|\Delta \log \mathcal{L}\|_{L_1}$", fontsize=9.5, color="#d62728")
    ax_b.tick_params(axis="y", labelcolor="#1f77b4")
    ax_b_twin.tick_params(axis="y", labelcolor="#d62728")
    ax_b.grid(True, linestyle=":", alpha=0.5)

    ax_b.legend(
        [ln_eg, ln_lik],
        [rf"$E_{{\mathrm{{global}}}}$ ($r={r_eg:.3f}$)", rf"Likelihood ($r={r_lik:.3f}$, $p={p_w:.3f}$)"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95
    )

    # -------------------------------------------------------------------------
    # Panel (c): 5D Coordinate Marginal Posterior Contractions
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])

    dims = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$", r"$\theta_5$"]
    x_indices = np.arange(len(dims))
    w1_dims_mean = [
        float(df_models[f"w1_dim{i}"].mean()) for i in range(5)
    ]
    w1_dims_std = [
        float(df_models[f"w1_dim{i}"].std()) for i in range(5)
    ]

    bars = ax_c.bar(
        x_indices, w1_dims_mean, yerr=w1_dims_std, capsize=4,
        color=["#2ca02c", "#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"],
        alpha=0.85, edgecolor="black", linewidth=0.8,
        label=r"Mean Marginal $\mathcal{W}_1(\pi_j, \widehat{\pi}_j) \pm 1\,\mathrm{std}$"
    )

    ax_c.set_title(r"(c) 5D Coordinate Marginals $\mathcal{W}_1(\pi_j, \widehat{\pi}_j)$", fontsize=10.5, pad=8)
    ax_c.set_xticks(x_indices)
    ax_c.set_xticklabels(dims, fontsize=9.5)
    ax_c.set_ylabel(r"Marginal Discrepancy $\mathcal{W}_1$", fontsize=9.5)
    ax_c.grid(True, linestyle=":", alpha=0.5, axis="y")

    ax_c.legend(
        [bars],
        [r"5D Marginal Posterior Discrepancy ($\pm 1\,\sigma$)"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95
    )

    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(output_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Vector PDF: {output_pdf}")
    print(f"Generated High-Res PNG: {output_png}")


if __name__ == "__main__":
    generate_figure_27()
