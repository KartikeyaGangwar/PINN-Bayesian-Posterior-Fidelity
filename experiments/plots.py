"""
Publication Figure Generator for Bayesian PINN Fidelity Research Program
========================================================================
Generates all 12 publication-quality 2D figures into results/research_figures/:
  Fig 1: Exact vs PINN solution fields
  Fig 2: Parameter-resolved error e(alpha)
  Fig 3: Representative Exact vs PINN posterior densities
  Fig 4: Forward error vs Posterior W1
  Fig 5: Error localization (PINN-A vs B vs C)
  Fig 6: Prior-weighted vs Posterior-weighted error diagnostics
  Fig 7: Observation noise sweep vs BFR
  Fig 8: Sensor density / concentration vs BFR
  Fig 9: Bayesian Fidelity Ratio distribution (Phase I baseline)
  Fig 10: Adaptive refinement progression (Stage 0 to Stage 1)
  Fig 11: Computational cost vs Posterior Fidelity
  Fig 12: Conceptual research workflow diagram
"""

import os, sys
sys.path.insert(0, os.getcwd())
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from bayesian.forward_operator import ExactForwardOperator, ParametricPINNForwardOperator, ParametricModifiedMLP
from bayesian.observation_operator import ObservationOperator


def generate_all_12_figures(output_dir: str = "results/research_figures"):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exact_op = ExactForwardOperator(resolution=100)
    canonical_weights = os.path.join("results", "heat_equation_pinn.pth")
    model = ParametricModifiedMLP(n_input=3, n_output=1, n_hidden=64, n_layers=4, use_fourier=False).to(device).to(torch.float64)
    model.load_state_dict(torch.load(canonical_weights, map_location=device, weights_only=True))
    model.eval()
    pinn_op = ParametricPINNForwardOperator(model=model, resolution=100, device=device)
    
    # -------------------------------------------------------------------------
    # FIGURE 1: Exact vs PINN Solution Field & Pointwise Absolute Error
    # -------------------------------------------------------------------------
    print("Generating Figure 1: Exact vs PINN Solution Field...", flush=True)
    u_exact_05 = exact_op(0.50).u
    u_pinn_05 = pinn_op(0.50).u
    diff_05 = np.abs(u_pinn_05 - u_exact_05)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    im0 = axes[0].imshow(u_exact_05.T, origin="lower", extent=[0, 1, 0, 1], cmap="inferno", aspect="auto")
    axes[0].set_title(r"(a) Exact Solution $u_{\mathrm{exact}}(x,t; \alpha=0.5)$")
    axes[0].set_xlabel(r"Space $x$")
    axes[0].set_ylabel(r"Time $t$")
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(u_pinn_05.T, origin="lower", extent=[0, 1, 0, 1], cmap="inferno", aspect="auto")
    axes[1].set_title(r"(b) PINN Surrogate $u_{\mathrm{PINN}}(x,t; \alpha=0.5)$")
    axes[1].set_xlabel(r"Space $x$")
    axes[1].set_ylabel(r"Time $t$")
    fig.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(diff_05.T, origin="lower", extent=[0, 1, 0, 1], cmap="magma", aspect="auto")
    axes[2].set_title(r"(c) Pointwise Absolute Error $|u_{\mathrm{PINN}} - u_{\mathrm{exact}}|$")
    axes[2].set_xlabel(r"Space $x$")
    axes[2].set_ylabel(r"Time $t$")
    fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_exact_vs_pinn_solution_fields.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # FIGURE 2: Parameter-Resolved Forward Error e(alpha)
    # -------------------------------------------------------------------------
    print("Generating Figure 2: Parameter-Resolved Forward Error e(alpha)...", flush=True)
    eval_alphas = np.linspace(0.1134, 2.2050, 100)
    x_sens = np.linspace(0.0, 1.0, 10)
    t_sens = np.array([0.1, 0.4, 0.7, 1.0])
    Xs, Ts = np.meshgrid(x_sens, t_sens, indexing="ij")
    obs_op = ObservationOperator(sensor_locations=np.column_stack([Xs.ravel(), Ts.ravel()]))
    
    e_field = []
    e_obs = []
    for a in eval_alphas:
        ue = exact_op(a).u
        up = pinn_op(a).u
        e_field.append(np.linalg.norm(up - ue) / np.linalg.norm(ue))
        ye = obs_op(ue)
        yp = obs_op(up)
        e_obs.append(np.linalg.norm(yp - ye) / np.linalg.norm(ye))
        
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(eval_alphas, np.array(e_field)*100, "b-", lw=2, label=r"Full Field Relative $L_2$ Error $e_{\mathrm{field}}(\alpha)$")
    ax.plot(eval_alphas, np.array(e_obs)*100, "r--", lw=2, label=r"Observation-Space Relative Error $e_{\mathrm{obs}}(\alpha)$")
    ax.axvspan(0.485, 0.525, color="gray", alpha=0.25, label=r"99% Posterior Region ($\alpha^*=0.50$)")
    ax.axvline(0.50, color="k", ls=":", lw=1.5, label=r"True Parameter $\alpha^* = 0.50$")
    ax.set_xlabel(r"Thermal Diffusivity Parameter $\alpha$")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(r"Parameter-Resolved PINN Surrogate Error $e(\alpha)$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_parameter_resolved_forward_error.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # FIGURE 3: Representative Exact vs PINN Posterior Densities
    # -------------------------------------------------------------------------
    print("Generating Figure 3: Representative Posterior Densities...", flush=True)
    with open("results/two_chain_summary.json") as f:
        tc_sum = json.load(f)
    res_npz = np.load("results/research_two_chain_results.npz")
    burn_in = 1000
    post_E = res_npz["alpha_E"][burn_in:]
    post_A = res_npz["alpha_A"][burn_in:]
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(post_E, bins=40, density=True, alpha=0.5, color="tab:blue", edgecolor="blue", label=rf"Exact Solver $\mathcal{{F}}_{{\mathrm{{exact}}}}$ ($\hat{{\alpha}}={np.mean(post_E):.4f}$)")
    ax.hist(post_A, bins=40, density=True, alpha=0.5, color="tab:orange", edgecolor="darkorange", label=rf"PINN Surrogate $\mathcal{{F}}_{{\mathrm{{PINN}}}}$ ($\hat{{\alpha}}={np.mean(post_A):.4f}$)")
    ax.axvline(0.50, color="red", lw=2, ls="--", label=r"True Parameter $\alpha^* = 0.5000$")
    ax.set_xlabel(r"Parameter $\alpha$")
    ax.set_ylabel("Posterior Probability Density")
    ax.set_title("Exact vs PINN Stationary Posterior Distributions")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_representative_posterior_comparison.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # FIGURE 4: Forward Error vs Posterior W1 (Phase II Accuracy Sweep)
    # -------------------------------------------------------------------------
    print("Generating Figure 4: Forward Error vs Posterior W1...", flush=True)
    df_p2 = pd.read_csv("results/phase2_accuracy_sweep/phase2_accuracy_sweep_summary.csv")
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(df_p2["mean_rel_l2_pct"], df_p2["w1_research"], "o-", color="darkblue", lw=2, ms=7, label=r"Surrogate Accuracy Levels (L1 $\to$ L6)")
    ax.axhline(df_p2["w1_control_baseline"].iloc[0], color="green", ls="--", lw=1.8, label=r"Exact-vs-Exact MCMC Noise Floor $\mathcal{W}_1^{\mathrm{baseline}}$")
    for i, row in df_p2.iterrows():
        ax.annotate(row["level"].replace("level", "L").replace("_", " "), (row["mean_rel_l2_pct"], row["w1_research"]), textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax.set_xlabel(r"Global Forward Relative $L_2$ Error (%)")
    ax.set_ylabel(r"Posterior Wasserstein-1 Distance $\mathcal{W}_1(\pi_E, \pi_A)$")
    ax.set_yscale("log")
    ax.set_title(r"Forward Error vs Posterior Discrepancy $\mathcal{W}_1$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_forward_error_vs_posterior_w1.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # FIGURE 5: Error Localization (PINN-A vs B vs C)
    # -------------------------------------------------------------------------
    print("Generating Figure 5: Parameter-Space Error Localization...", flush=True)
    with open("results/phase3_error_localization/phase3_localization_results.json") as f:
        p3_json = json.load(f)
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    curves = p3_json["curves"]
    colors = {"pinn_a_posterior_focused": "tab:purple", "pinn_b_uniform": "tab:green", "pinn_c_tail_focused": "tab:red"}
    labels = {
        "pinn_a_posterior_focused": r"PINN-A (Posterior-Focused, $E_{\mathrm{post}}=2.58\%$, $\mathcal{W}_1=1.23\times 10^{-3}$)",
        "pinn_b_uniform": r"PINN-B (Uniform Coverage, $E_{\mathrm{post}}=1.55\%$, $\mathcal{W}_1=4.70\times 10^{-4}$)",
        "pinn_c_tail_focused": r"PINN-C (Tail-Focused, $E_{\mathrm{post}}=2.92\%$, $\mathcal{W}_1=6.95\times 10^{-4}$)"
    }
    
    for k, v in curves.items():
        ax.plot(v["alpha_grid"], np.array(v["e_alpha_field"])*100, lw=2, color=colors[k], label=labels[k])
        
    ax.axvspan(0.485, 0.525, color="gray", alpha=0.25, label=r"Active Posterior Support ($\alpha \approx 0.50$)")
    ax.set_xlabel(r"Parameter $\alpha$")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Parameter-Space Error Distribution Across Training Strategies")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig5_error_localization_pinn_a_b_c.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # FIGURE 6: Prior vs Posterior Weighted Error Diagnostics
    # -------------------------------------------------------------------------
    print("Generating Figure 6: Prior vs Posterior Weighted Diagnostics...", flush=True)
    df_p3 = pd.read_csv("results/phase3_error_localization/phase3_localization_summary.csv")
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x_idx = np.arange(len(df_p3))
    width = 0.22
    
    ax.bar(x_idx - width, df_p3["E_global_pct"], width=width, color="tab:blue", label=r"Global Error $E_{\mathrm{global}}$")
    ax.bar(x_idx, df_p3["E_prior_pct"], width=width, color="tab:orange", label=r"Prior-Weighted $E_{\mathrm{prior}}$")
    ax.bar(x_idx + width, df_p3["E_posterior_pct"], width=width, color="tab:green", label=r"Posterior-Weighted $E_{\mathrm{posterior}}$")
    
    ax.set_xticks(x_idx)
    ax.set_xticklabels(["PINN-A\n(Post-Focused)", "PINN-B\n(Uniform)", "PINN-C\n(Tail-Focused)"])
    ax.set_ylabel("Surrogate Error Diagnostic (%)")
    ax.set_title("Comparison of Global, Prior-Weighted, and Posterior-Weighted Error")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig6_prior_vs_posterior_weighted_error.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # FIGURE 7: Observation Noise Sweep vs Posterior Discrepancy & BFR
    # -------------------------------------------------------------------------
    print("Generating Figure 7: Observation Noise Sweep vs BFR...", flush=True)
    df_p5 = pd.read_csv("results/phase5_noise_sweep/phase5_noise_sweep_summary.csv")
    
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    color = "tab:red"
    ax1.set_xlabel(r"Observation Noise Standard Deviation $\sigma_{\mathrm{noise}}$")
    ax1.set_ylabel(r"Bayesian Fidelity Ratio $\mathrm{BFR} = \mathcal{W}_1 / \mathcal{W}_1^{\mathrm{baseline}}$", color=color)
    ax1.plot(df_p5["sigma_noise"], df_p5["mean_bfr"], "o-", color=color, lw=2, ms=7, label="Mean BFR")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel(r"Posterior Standard Deviation $\sigma_{\mathrm{post}}$", color=color)
    ax2.plot(df_p5["sigma_noise"], df_p5["mean_exact_std"], "s--", color=color, lw=2, ms=6, label=r"$\sigma_{\mathrm{post}}$")
    ax2.tick_params(axis="y", labelcolor=color)
    
    plt.title(r"Modulation of Surrogate Sensitivity by Noise Level $\sigma_{\mathrm{noise}}$")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig7_observation_noise_vs_bfr.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # FIGURE 8: Sensor Density / Posterior Concentration vs BFR
    # -------------------------------------------------------------------------
    print("Generating Figure 8: Sensor Density vs BFR...", flush=True)
    df_p6 = pd.read_csv("results/phase6_sensor_sweep/phase6_sensor_sweep_summary.csv")
    
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(df_p6["sensor_count_M"], df_p6["mean_bfr"], "o-", color="purple", lw=2, ms=7, label="Mean BFR")
    ax1.set_xlabel(r"Sensor Count $M$")
    ax1.set_ylabel(r"Bayesian Fidelity Ratio $\mathrm{BFR}$", color="purple")
    ax1.tick_params(axis="y", labelcolor="purple")
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(df_p6["sensor_count_M"], df_p6["mean_exact_std"], "s--", color="teal", lw=2, ms=6, label=r"Posterior Std $\sigma_{\mathrm{post}}$")
    ax2.set_ylabel(r"Posterior Uncertainty $\sigma_{\mathrm{post}}$", color="teal")
    ax2.tick_params(axis="y", labelcolor="teal")
    
    plt.title(r"Sensor Information Content & Posterior Concentration vs BFR")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig8_sensor_density_vs_bfr.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # FIGURE 9: Bayesian Fidelity Ratio Distribution (Phase I Baseline)
    # -------------------------------------------------------------------------
    print("Generating Figure 9: Baseline BFR Distribution Across Realizations...", flush=True)
    df_p1 = pd.read_csv("results/phase1_baseline/phase1_baseline_summary.csv")
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(df_p1["realization_idx"], df_p1["bfr"], color="royalblue", edgecolor="darkblue", alpha=0.75, width=0.6)
    ax.axhline(df_p1["bfr"].mean(), color="red", ls="--", lw=2, label=rf"Mean $\mathrm{{BFR}} = {df_p1['bfr'].mean():.2f} \pm {df_p1['bfr'].std():.2f}$")
    ax.axhline(1.0, color="green", ls=":", lw=1.8, label=r"Ideal Stochastic Parity ($\mathrm{BFR} = 1.0$)")
    ax.set_xlabel("Noise Realization Index")
    ax.set_ylabel(r"Bayesian Fidelity Ratio $\mathrm{BFR}$")
    ax.set_title("Bayesian Fidelity Ratio Across 10 Independent Noise Realizations")
    ax.set_xticks(df_p1["realization_idx"])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig9_bayesian_fidelity_ratio_distribution.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # FIGURE 10: Adaptive Refinement Progression (Stage 0 vs Stage 1)
    # -------------------------------------------------------------------------
    print("Generating Figure 10: Adaptive Refinement Progression...", flush=True)
    df_p9 = pd.read_csv("results/phase9_adaptive_refinement/phase9_adaptive_progression.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    stages = ["Stage 0\n(Pilot)", "Stage 1\n(Refined)"]
    
    ax1.bar(stages, df_p9["E_posterior_pct"], color=["gray", "forestgreen"], width=0.5, edgecolor="black")
    ax1.set_ylabel(r"Posterior-Weighted Error $E_{\mathrm{posterior}}$ (%)")
    ax1.set_title(r"(a) Posterior Error Reduction ($21.2\% \to 1.3\%$)")
    ax1.grid(True, axis="y", alpha=0.3)
    
    ax2.bar(stages, df_p9["bfr"], color=["firebrick", "dodgerblue"], width=0.5, edgecolor="black")
    ax2.set_ylabel(r"Bayesian Fidelity Ratio $\mathrm{BFR}$")
    ax2.set_title(r"(b) Fidelity Ratio Progression ($61.0 \to 7.0$)")
    ax2.grid(True, axis="y", alpha=0.3)
    
    plt.suptitle("Multi-Stage Adaptive Refinement Progression", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig10_adaptive_refinement_progression.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # FIGURE 11: Computational Cost vs Posterior Fidelity
    # -------------------------------------------------------------------------
    print("Generating Figure 11: Computational Cost vs Posterior Fidelity...", flush=True)
    df_p8 = pd.read_csv("results/phase8_posterior_aware/phase8_strategy_comparison.csv")
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors_p8 = ["tab:blue", "tab:orange", "tab:green"]
    for i, row in df_p8.iterrows():
        ax.scatter(row["training_time_s"], row["w1_research"], s=140, color=colors_p8[i], label=row["strategy"].replace("_", " ").title())
        ax.annotate(row["strategy"].replace("_", " ").title(), (row["training_time_s"], row["w1_research"]), textcoords="offset points", xytext=(8, -4), fontsize=9)
        
    ax.set_xlabel("PINN Training Time (seconds)")
    ax.set_ylabel(r"Posterior Wasserstein-1 Distance $\mathcal{W}_1$")
    ax.set_title("Training Cost vs Bayesian Posterior Discrepancy")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig11_computational_cost_vs_fidelity.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # FIGURE 12: Conceptual Research Workflow Diagram
    # -------------------------------------------------------------------------
    print("Generating Figure 12: Conceptual Workflow Diagram...", flush=True)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    
    boxes = [
        ("Forward Surrogate\nAccuracy (PINN)", (0.05, 0.55), "lightblue"),
        ("Likelihood\nPerturbation", (0.28, 0.55), "lightgreen"),
        ("Posterior\nDistortion", (0.51, 0.55), "wheat"),
        ("Posterior\nFidelity (BFR)", (0.74, 0.55), "salmon"),
        ("Error Localization\n& Diagnostics", (0.28, 0.15), "thistle"),
        ("Posterior-Aware\nAdaptive Loop", (0.62, 0.15), "lightcoral")
    ]
    
    for text, (x, y), c in boxes:
        box = patches.FancyBboxPatch((x, y), 0.18, 0.28, boxstyle="round,pad=0.03", fc=c, ec="black", lw=1.5)
        ax.add_patch(box)
        ax.text(x + 0.09, y + 0.14, text, ha="center", va="center", fontsize=10.5, weight="bold")
        
    # Forward arrows
    ax.annotate("", xy=(0.28, 0.69), xytext=(0.23, 0.69), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8))
    ax.annotate("", xy=(0.51, 0.69), xytext=(0.46, 0.69), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8))
    ax.annotate("", xy=(0.74, 0.69), xytext=(0.69, 0.69), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8))
    
    # Feedback loop arrows
    ax.annotate("", xy=(0.37, 0.43), xytext=(0.37, 0.55), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8))
    ax.annotate("", xy=(0.62, 0.29), xytext=(0.46, 0.29), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8))
    ax.annotate("", xy=(0.14, 0.55), xytext=(0.62, 0.35), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8, connectionstyle="arc3,rad=0.3"))
    
    plt.title("Bayesian PINN Surrogate Fidelity & Error Localization Framework", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig12_conceptual_framework_workflow.png"))
    plt.close()
    
    print(f"\n[SUCCESS] Generated all 12 publication figures in {output_dir}/")


if __name__ == "__main__":
    generate_all_12_figures()
