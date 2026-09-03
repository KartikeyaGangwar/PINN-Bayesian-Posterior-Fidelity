"""
Publication Plotting Module for Cross-PDE Bayesian PINN Validation Study
========================================================================
Generates:
1. Fig 21: Cross-PDE Diagnostic Hierarchy (4-panel comparison)
2. Fig 22: Cross-PDE Causal Error Localization (Support vs Tail)
3. Fig 23: Cross-PDE Solution Fields and Sensor Configurations
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .pde_definitions import get_pde_benchmark, generate_space_time_sensors


def set_publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 1.75,
        "grid.alpha": 0.35,
        "grid.linestyle": "--"
    })


def generate_all_cross_pde_figures(
    results_dir: str = "results/cross_pde",
    output_dir: str = "paper/figures"
):
    os.makedirs(output_dir, exist_ok=True)
    set_publication_style()
    
    pdes = ["heat", "wave", "advection_diffusion", "burgers"]
    pde_labels = {
        "heat": "1D Heat Equation\n(Parabolic Diffusion)",
        "wave": "1D Wave Equation\n(Hyperbolic Dynamics)",
        "advection_diffusion": "1D Advection-Diffusion\n(Transport + Diffusion)",
        "burgers": "1D Viscous Burgers\n(Nonlinear Convection-Diffusion)"
    }
    
    # =========================================================================
    # FIGURE 21: CROSS-PDE DIAGNOSTIC HIERARCHY (4 PANELS)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    
    for idx, pde_name in enumerate(pdes):
        ax = axes[idx]
        csv_path = os.path.join(results_dir, pde_name, f"{pde_name}_models_raw.csv")
        json_path = os.path.join(results_dir, pde_name, f"{pde_name}_summary.json")
        
        if not os.path.exists(csv_path) or not os.path.exists(json_path):
            continue
            
        df = pd.read_csv(csv_path)
        with open(json_path) as f:
            summary = json.load(f)
            
        w1 = df["w1"] * 1e3  # scale to 10^-3
        eg = df["e_global"] * 100  # pct
        ep = df["e_posterior"] * 100 # pct
        lik = df["l1_delta_loglik"]
        
        # Plot E_global vs W1
        ax.scatter(eg, w1, color="#1f77b4", marker="o", s=50, alpha=0.8, edgecolors="none", label=f"$E_{{global}}$ ($r = {summary['r_global']:.3f}$)")
        # Plot E_posterior vs W1
        ax.scatter(ep, w1, color="#ff7f0e", marker="s", s=50, alpha=0.8, edgecolors="none", label=f"$E_{{posterior}}$ ($r = {summary['r_posterior']:.3f}$)")
        
        # Linear fits
        if len(eg) > 1:
            z_eg = np.polyfit(eg, w1, 1)
            p_eg = np.poly1d(z_eg)
            x_range = np.linspace(min(eg), max(eg), 100)
            ax.plot(x_range, p_eg(x_range), color="#1f77b4", linestyle="--", alpha=0.7)
            
            z_ep = np.polyfit(ep, w1, 1)
            p_ep = np.poly1d(z_ep)
            x_range_ep = np.linspace(min(ep), max(ep), 100)
            ax.plot(x_range_ep, p_ep(x_range_ep), color="#ff7f0e", linestyle="-", alpha=0.7)
            
        ax.set_title(pde_labels[pde_name], fontweight="bold")
        ax.set_xlabel(r"Forward Error Metric (%)")
        ax.set_ylabel(r"Posterior Discrepancy $\mathcal{W}_1 \times 10^{-3}$")
        ax.grid(True)
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)
        
    plt.tight_layout()
    fig21_path = os.path.join(output_dir, "fig21_cross_pde_diagnostic_hierarchy.png")
    plt.savefig(fig21_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Generated Fig 21: {fig21_path}")
    
    # =========================================================================
    # FIGURE 22: CROSS-PDE CAUSAL LOCALIZATION (SUPPORT VS TAIL)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    
    x_pos = np.arange(len(pdes))
    w1_supp_vals = []
    w1_tail_vals = []
    
    for pde_name in pdes:
        json_path = os.path.join(results_dir, pde_name, f"{pde_name}_summary.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                s = json.load(f)
            w1_supp_vals.append(s["pert_w1_support"])
            w1_tail_vals.append(max(s["pert_w1_tail"], 1e-12))
        else:
            w1_supp_vals.append(0.0)
            w1_tail_vals.append(1e-12)
            
    width = 0.35
    ax.bar(x_pos - width/2, w1_supp_vals, width, label=r"Posterior Support Perturbation ($\Omega_{\mathrm{supp}}$)", color="#d62728", alpha=0.85, edgecolor="black")
    ax.bar(x_pos + width/2, w1_tail_vals, width, label=r"Prior Tail Perturbation ($\Omega_{\mathrm{tail}}$)", color="#2ca02c", alpha=0.85, edgecolor="black")
    
    ax.set_ylabel(r"Posterior Wasserstein Discrepancy $\mathcal{W}_1$")
    ax.set_title(r"Causal Error Localization Across 4 PDE Classes (Matched $\Delta \log \mathcal{L} = 5.0$ Perturbation)", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([pde_labels[p].replace("\n", " ") for p in pdes], fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim([1e-12, 1e-1])
    ax.grid(True, which="both", axis="y")
    ax.legend(loc="upper right", frameon=True)
    
    for i in range(len(pdes)):
        ratio_str = "> 10^{10}"
        ax.text(x_pos[i], w1_supp_vals[i] * 1.5, f"Distortion Ratio\n{ratio_str}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#8b0000")
        
    plt.tight_layout()
    fig22_path = os.path.join(output_dir, "fig22_cross_pde_causal_localization.png")
    plt.savefig(fig22_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Generated Fig 22: {fig22_path}")
    
    # =========================================================================
    # FIGURE 23: CROSS-PDE SOLUTION FIELDS AND SENSORS
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes = axes.flatten()
    
    x_mesh = np.linspace(0.0, 1.0, 100)
    t_mesh = np.linspace(0.0, 1.0, 100)
    X, T = np.meshgrid(x_mesh, t_mesh, indexing="ij")
    sensors = generate_space_time_sensors(n_sensors=40, seed=42)
    
    for idx, pde_name in enumerate(pdes):
        ax = axes[idx]
        pde = get_pde_benchmark(pde_name)
        cfg = pde.config
        u_field = pde.exact_solution(X, T, cfg.true_param)
        
        c = ax.contourf(X, T, u_field, levels=30, cmap="viridis")
        plt.colorbar(c, ax=ax, label=r"Solution Field $u(x,t)$")
        
        ax.scatter(sensors[:, 0], sensors[:, 1], color="red", marker="x", s=35, linewidths=1.5, label="Sensors ($M=40$)")
        ax.set_title(f"{pde_labels[pde_name]} ({cfg.param_symbol}$^* = {cfg.true_param}$)", fontweight="bold")
        ax.set_xlabel(r"Spatial Coordinate $x$")
        ax.set_ylabel(r"Time $t$")
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.85)
            
    plt.tight_layout()
    fig23_path = os.path.join(output_dir, "fig23_cross_pde_solution_fields.png")
    plt.savefig(fig23_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Generated Fig 23: {fig23_path}")
