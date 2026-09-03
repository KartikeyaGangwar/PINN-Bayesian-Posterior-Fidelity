r"""
Parallel Two-Chain Bayesian Inversion Runner (1D Heat Equation Benchmark)
==========================================================================
Executes:
1. Control Experiment: Chain E (Exact) vs Chain A (Exact) -> pi_E = pi_A
2. Research Experiment: Chain E (Exact) vs Chain A (PINN Surrogate)
3. 12 Publication Figures:
   Fig 1: Exact Heat-Equation Solution Field
   Fig 2: PINN Solution Field
   Fig 3: Exact-vs-PINN Absolute Error Field
   Fig 4: Exact-vs-PINN Relative Error vs Alpha
   Fig 5: Observation-Space Error vs Alpha
   Fig 6: Log-Posterior Discrepancy Delta ln pi(alpha) vs Alpha
   Fig 7: Dense Grid Direct Posterior Density (Exact vs PINN)
   Fig 8: Two-Chain Alpha Trajectories (alpha_E(t) vs alpha_A(t))
   Fig 9: Observational Distance Evolution (d_t = |alpha_E - alpha_A|)
   Fig 10: Post-Burn-in Posterior Histograms and KDE Overlay
   Fig 11: 4-Panel Master Summary Dashboard
   Fig 12: Control Experiment Trajectories and Posterior Comparison
4. Dynamic GIF Animation
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from bayesian.forward_operator import ExactForwardOperator, ParametricPINNForwardOperator, ParametricModifiedMLP
from bayesian.observation_operator import ObservationOperator
from bayesian.prior import Prior, LogNormalPrior
from bayesian.likelihood import GaussianLikelihood
from bayesian.posterior import Posterior
from bayesian.proposal import GaussianRandomWalkProposal
from bayesian.two_chain_sampler import TwoChainSampler, TwoChainMCMCResult
from bayesian.two_chain_analysis import analyze_two_chain_experiment
from bayesian.two_chain_plots import generate_all_two_chain_plots
from bayesian.animation import generate_two_chain_animation


def main():
    print("=" * 85)
    print("PARALLEL TWO-CHAIN BAYESIAN INVERSE PROBLEM (1D HEAT EQUATION BENCHMARK)")
    print("=" * 85)
    
    output_dir = "results"
    figures_dir = os.path.join(output_dir, "two_chain_figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. SETUP PROBLEM & FORWARD SOLVERS
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Initializing Forward Solvers & Ground Truth...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exact_op = ExactForwardOperator(resolution=100)
    
    # Load or instantiate PINN model
    model = ParametricModifiedMLP(
        n_input=3, n_output=1, n_hidden=64, n_layers=4,
        use_fourier=False, fourier_scale=1.0
    ).to(device).to(torch.float64)
    
    weights_path = os.path.join(output_dir, "heat_equation_pinn.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(output_dir, "parametric_pinn_weights.pth")
        
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"  [OK] Loaded trained PINN surrogate weights from {weights_path}")
    else:
        print(f"  [WARNING] Checkpoint {weights_path} not found. Running with un-trained weights.")
        
    pinn_op = ParametricPINNForwardOperator(model=model, resolution=100, device=device)
    
    # -------------------------------------------------------------------------
    # 2. GENERATE SYNTHETIC SENSOR OBSERVATIONS
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Generating Synthetic Observations from Ground Truth...")
    true_alpha = 0.5
    sigma_noise = 0.01
    
    # Spatial-temporal sensor grid: 10 spatial x 4 temporal = 40 sensors
    x_sens = np.linspace(0.0, 1.0, 10)
    t_sens = np.linspace(0.1, 1.0, 4)
    Xs, Ts = np.meshgrid(x_sens, t_sens, indexing="ij")
    sensor_coords = np.column_stack([Xs.ravel(), Ts.ravel()])
    obs_op = ObservationOperator(sensor_locations=sensor_coords)
    
    out_star = exact_op(true_alpha)
    y_clean = obs_op(out_star.u)
    np.random.seed(42)
    noise = sigma_noise * np.random.randn(*y_clean.shape)
    y_obs = y_clean + noise
    print(f"  True alpha*   : {true_alpha:.4f}")
    print(f"  Sensor Points : {len(sensor_coords)} ({len(x_sens)} spatial x {len(t_sens)} temporal)")
    print(f"  Noise Level   : sigma = {sigma_noise:.4f}")
    
    # -------------------------------------------------------------------------
    # 3. BAYESIAN POSTERIOR & DENSE GRID EVALUATION
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Evaluating Prior and Dense Grid Posterior Landscapes...")
    prior = Prior(LogNormalPrior(mu=float(np.log(0.5)), sigma=0.5))
    likelihood = GaussianLikelihood(noise_std=sigma_noise)
    posterior = Posterior(prior=prior, likelihood=likelihood, obs_operator=obs_op)
    
    # Dense 1D grid of alpha for direct posterior density comparison
    alpha_dense_grid = np.linspace(0.1, 1.5, 200)
    dense_grid_exact = posterior.evaluate_grid(alpha_dense_grid, y_obs, exact_op)
    dense_grid_pinn = posterior.evaluate_grid(alpha_dense_grid, y_obs, pinn_op)
    
    # -------------------------------------------------------------------------
    # 4. CONTROL EXPERIMENT (EXACT VS EXACT)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("RUNNING CONTROL EXPERIMENT: CHAIN E (EXACT) VS CHAIN A (EXACT)")
    print("=" * 85)
    proposal = GaussianRandomWalkProposal(scale=0.05)
    control_sampler = TwoChainSampler(
        posterior=posterior,
        proposal=proposal,
        forward_solver_exact=exact_op,
        forward_solver_pinn=exact_op
    )
    
    control_result = control_sampler.run(
        y_obs=y_obs,
        n_samples=5000,
        initial_alpha=None,
        seed=101,
        is_control=True,
        verbose=True,
        print_interval=1000
    )
    control_result.save_to_npz(os.path.join(output_dir, "control_two_chain_results.npz"))
    control_analysis = analyze_two_chain_experiment(control_result, burn_in=1000)
    
    # -------------------------------------------------------------------------
    # 5. RESEARCH EXPERIMENT (EXACT VS PINN)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("RUNNING RESEARCH EXPERIMENT: CHAIN E (EXACT) VS CHAIN A (PINN SURROGATE)")
    print("=" * 85)
    research_sampler = TwoChainSampler(
        posterior=posterior,
        proposal=proposal,
        forward_solver_exact=exact_op,
        forward_solver_pinn=pinn_op
    )
    
    # Common random initial parameter drawn from prior
    common_alpha_0 = float(prior.sample())
    while common_alpha_0 <= 0 or np.isneginf(prior.log_prior(common_alpha_0)):
        common_alpha_0 = float(prior.sample())
        
    research_result = research_sampler.run(
        y_obs=y_obs,
        n_samples=5000,
        initial_alpha=common_alpha_0,
        seed=101,
        is_control=False,
        verbose=True,
        print_interval=1000
    )
    
    research_result.save_to_npz(os.path.join(output_dir, "research_two_chain_results.npz"))
    research_result.save_summary_json(os.path.join(output_dir, "two_chain_summary.json"), burn_in=1000)
    
    research_analysis = analyze_two_chain_experiment(research_result, burn_in=1000)
    with open(os.path.join(output_dir, "two_chain_analysis_report.json"), "w") as f:
        json.dump(research_analysis, f, indent=4)
        
    # -------------------------------------------------------------------------
    # 6. GENERATE ALL 12 PUBLICATION FIGURES
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Generating 12 Publication-Quality Figures...")
    
    out_e_true = exact_op(true_alpha)
    out_a_true = pinn_op(true_alpha)
    
    # Fig 1: Exact Heat-Equation Solution Field
    fig, ax = plt.subplots(figsize=(6, 5))
    im1 = ax.imshow(out_e_true.u.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", aspect="auto")
    ax.set_title(rf"Fig 1: Exact Heat-Equation Solution Field $u_{{\mathrm{{exact}}}}(x,t;\alpha^*={true_alpha:.3f})$")
    ax.set_xlabel("Space $x$")
    ax.set_ylabel("Time $t$")
    fig.colorbar(im1, ax=ax)
    fig.savefig(os.path.join(figures_dir, "fig1_exact_solution_field.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Fig 2: PINN Solution Field
    fig, ax = plt.subplots(figsize=(6, 5))
    im2 = ax.imshow(out_a_true.u.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", aspect="auto")
    ax.set_title(rf"Fig 2: PINN Surrogate Solution Field $u_{{\mathrm{{PINN}}}}(x,t;\alpha^*={true_alpha:.3f})$")
    ax.set_xlabel("Space $x$")
    ax.set_ylabel("Time $t$")
    fig.colorbar(im2, ax=ax)
    fig.savefig(os.path.join(figures_dir, "fig2_pinn_solution_field.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Fig 3: Exact-vs-PINN Absolute Error Field
    err_field = np.abs(out_a_true.u - out_e_true.u)
    fig, ax = plt.subplots(figsize=(6, 5))
    im3 = ax.imshow(err_field.T, origin="lower", extent=[0, 1, 0, 1], cmap="inferno", aspect="auto")
    ax.set_title(rf"Fig 3: Solution Absolute Error $|u_{{\mathrm{{PINN}}}} - u_{{\mathrm{{exact}}}}|$")
    ax.set_xlabel("Space $x$")
    ax.set_ylabel("Time $t$")
    fig.colorbar(im3, ax=ax)
    fig.savefig(os.path.join(figures_dir, "fig3_absolute_error_field.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Fig 4: Exact-vs-PINN Relative Error vs Alpha
    val_alphas = np.linspace(0.1, 1.5, 50)
    rel_l2s = [float(np.linalg.norm(pinn_op(a).u - exact_op(a).u) / np.linalg.norm(exact_op(a).u)) for a in val_alphas]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(val_alphas, np.array(rel_l2s) * 100, color="#1f77b4", lw=2.0)
    ax.axvline(true_alpha, color="red", linestyle="--", label=rf"True $\alpha^*={true_alpha:.3f}$")
    ax.set_xlabel(r"Thermal Diffusivity $\alpha$")
    ax.set_ylabel(r"Relative $L_2$ Error (%)")
    ax.set_title(r"Fig 4: Solution Field Relative $L_2$ Error Across $\alpha$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(figures_dir, "fig4_relative_l2_error_vs_alpha.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Fig 5: Observation-Space Error vs Alpha
    obs_errs = [float(np.linalg.norm(obs_op(pinn_op(a).u) - obs_op(exact_op(a).u)) / np.linalg.norm(obs_op(exact_op(a).u))) for a in val_alphas]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(val_alphas, np.array(obs_errs) * 100, color="#2ca02c", lw=2.0)
    ax.axvline(true_alpha, color="red", linestyle="--", label=rf"True $\alpha^*={true_alpha:.3f}$")
    ax.set_xlabel(r"Thermal Diffusivity $\alpha$")
    ax.set_ylabel("Observation Relative Error (%)")
    ax.set_title(r"Fig 5: Observation-Space Error Across $\alpha$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(figures_dir, "fig5_observation_error_vs_alpha.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Fig 6: Log-Posterior Discrepancy vs Alpha
    lp_e = [posterior.log_posterior(a, y_obs, exact_op) for a in val_alphas]
    lp_a = [posterior.log_posterior(a, y_obs, pinn_op) for a in val_alphas]
    d_lp = np.array(lp_a) - np.array(lp_e)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(val_alphas, d_lp, color="#d62728", lw=2.0)
    ax.axhline(0.0, color="black", linestyle=":")
    ax.axvline(true_alpha, color="red", linestyle="--", label=rf"True $\alpha^*={true_alpha:.3f}$")
    ax.set_xlabel(r"Thermal Diffusivity $\alpha$")
    ax.set_ylabel(r"$\Delta \ln \pi(\alpha) = \ln \pi_A - \ln \pi_E$")
    ax.set_title(r"Fig 6: Log-Posterior Discrepancy $\Delta \ln \pi(\alpha)$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(figures_dir, "fig6_log_posterior_discrepancy.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Fig 7: Dense Grid Direct Posterior Density Comparison (Exact vs PINN)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(dense_grid_exact["alpha_grid"], dense_grid_exact["normalized_pdf"], color="#1f77b4", lw=2.2, label=r"Exact Target $\pi_E(\alpha \mid y)$")
    ax.plot(dense_grid_pinn["alpha_grid"], dense_grid_pinn["normalized_pdf"], color="#d62728", lw=2.2, linestyle="--", label=r"PINN Target $\pi_A(\alpha \mid y)$")
    ax.axvline(true_alpha, color="black", linestyle="-.", label=rf"True $\alpha^*={true_alpha:.3f}$")
    ax.set_xlabel(r"Thermal Diffusivity $\alpha$")
    ax.set_ylabel("Posterior Probability Density")
    ax.set_title(r"Fig 7: Exact vs PINN Posterior Density (Direct Dense Grid)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(figures_dir, "fig7_dense_grid_posterior_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Fig 8: Two-Chain Alpha Trajectories
    fig_tr = generate_all_two_chain_plots(
        result=research_result,
        output_dir=figures_dir,
        burn_in=1000,
        true_alpha=true_alpha,
        dense_grid_data=dense_grid_exact
    )
    
    # Fig 12: Control Experiment Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    alpha_E_c = control_result.get_alpha_E_array()
    alpha_A_c = control_result.get_alpha_A_array()
    dist_c = control_result.get_distance_array()
    iters_c = np.arange(1, len(alpha_E_c) + 1)
    
    ax1.plot(iters_c, alpha_E_c, color="#1f77b4", alpha=0.7, lw=1.2, label=r"Chain 1 ($\mathcal{F}_{\mathrm{exact}}$)")
    ax1.plot(iters_c, alpha_A_c, color="#ff7f0e", alpha=0.7, lw=1.2, linestyle="--", label=r"Chain 2 ($\mathcal{F}_{\mathrm{exact}}$)")
    ax1.axhline(true_alpha, color="black", linestyle=":", label=rf"True $\alpha^*={true_alpha:.3f}$")
    ax1.set_xlabel("Iteration $t$")
    ax1.set_ylabel(r"$\alpha$")
    ax1.set_title("Control MCMC Trajectories (Both Exact)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(iters_c, dist_c, color="#2ca02c", lw=1.2, label=r"$d_t = |\alpha_1^{(t)} - \alpha_2^{(t)}|$")
    ax2.set_xlabel("Iteration $t$")
    ax2.set_ylabel(r"Distance $d_t$")
    ax2.set_title("Control Distance Evolution")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig12_control_experiment_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Fig 8, 9, 10, 11 renamed to canonical paths
    os.replace(os.path.join(figures_dir, "alpha_trajectories.png"), os.path.join(figures_dir, "fig8_alpha_trajectories.png"))
    os.replace(os.path.join(figures_dir, "distance_trajectory.png"), os.path.join(figures_dir, "fig9_distance_trajectory.png"))
    os.replace(os.path.join(figures_dir, "posterior_comparison.png"), os.path.join(figures_dir, "fig10_posterior_comparison.png"))
    os.replace(os.path.join(figures_dir, "master_dashboard.png"), os.path.join(figures_dir, "fig11_master_dashboard.png"))
    
    # -------------------------------------------------------------------------
    # 7. GENERATE DYNAMIC ANIMATION GIF
    # -------------------------------------------------------------------------
    print("\n[STEP 7] Generating Dynamic GIF Animation...")
    gif_path = os.path.join(output_dir, "two_chain_evolution.gif")
    generate_two_chain_animation(
        result_or_npz=research_result,
        true_alpha=true_alpha,
        output_path=gif_path,
        n_frames=120,
        fps=15
    )
    print(f"  [OK] Animation saved to {gif_path}")
    
    # -------------------------------------------------------------------------
    # 8. PRINT SUMMARY RESULTS
    # -------------------------------------------------------------------------
    a_E_post, a_A_post = research_result.get_post_burnin_samples(burn_in=1000)
    w1_dist = float(stats.wasserstein_distance(a_E_post, a_A_post))
    ks_stat, ks_pval = stats.ks_2samp(a_E_post, a_A_post)
    
    print("\n" + "=" * 85)
    print("SCIENTIFIC EXPERIMENT SUMMARY (1D HEAT EQUATION BENCHMARK)")
    print("=" * 85)
    print(f"  True Parameter alpha*       : {true_alpha:.4f}")
    print(f"  Common Initial Guess alpha0 : {common_alpha_0:.4f}")
    print("  -------------------------------------------------------------------------")
    print(f"  Exact Chain Post-Burnin Mean: {np.mean(a_E_post):.6f} +/- {np.std(a_E_post):.6f}")
    print(f"  PINN Chain Post-Burnin Mean : {np.mean(a_A_post):.6f} +/- {np.std(a_A_post):.6f}")
    print(f"  Mean Parameter Bias         : {np.mean(a_A_post) - np.mean(a_E_post):.6e}")
    print(f"  Wasserstein-1 Distance      : {w1_dist:.6e}")
    print(f"  Kolmogorov-Smirnov Statistic: {ks_stat:.4f} (p-value = {ks_pval:.4f})")
    print(f"  Mean Observational Distance : {np.mean(research_result.get_distance_array()):.6e}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
