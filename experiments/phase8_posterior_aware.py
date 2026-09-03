"""
Phase VIII & IX: Posterior-Aware PINN Training Strategy & Adaptive Refinement Loop
================================================================================
Implements:
1. Phase VIII: Controlled comparison of Uniform vs Prior-Weighted vs Posterior-Aware Training.
2. Phase IX: Multi-stage Adaptive Refinement Loop (Pilot PINN -> Pilot MCMC -> Targeted Collocation -> Refined MCMC).
Tracks computational cost, forward accuracy, and Bayesian posterior fidelity.
"""

import os, sys
sys.path.insert(0, os.getcwd())
import time
import json
import numpy as np
import torch
import pandas as pd
from typing import Dict, Any, List, Tuple

from bayesian.forward_operator import ExactForwardOperator, ParametricPINNForwardOperator, ParametricModifiedMLP
from bayesian.observation_operator import ObservationOperator
from bayesian.prior import LogNormalPrior, Prior
from bayesian.likelihood import GaussianLikelihood
from bayesian.posterior import Posterior
from bayesian.proposal import GaussianRandomWalkProposal
from bayesian.two_chain_sampler import TwoChainSampler

from parametric_surrogate.exact_dataset_generator import generate_exact_parametric_dataset
from parametric_surrogate.dataset import ParametricPINNDataset
from parametric_surrogate.trainer import ParametricPINNTrainer

from experiments.metrics import (
    evaluate_parameter_resolved_error,
    compute_global_forward_metrics,
    compute_weighted_forward_errors,
    compute_bayesian_posterior_metrics,
    compute_bayesian_fidelity_ratio
)


def train_pinn_with_alphas(
    alphas: np.ndarray,
    adam_epochs: int = 400,
    lbfgs_steps: int = 25,
    initial_weights_path: str = None,
    device: torch.device = None
) -> Tuple[ParametricModifiedMLP, float]:
    """Trains a PINN surrogate given exact parameter collocation points."""
    t0 = time.time()
    model = ParametricModifiedMLP(n_input=3, n_output=1, n_hidden=64, n_layers=4, use_fourier=False).to(device).to(torch.float64)
    if initial_weights_path and os.path.exists(initial_weights_path):
        model.load_state_dict(torch.load(initial_weights_path, map_location=device, weights_only=True))
        
    data_dict = generate_exact_parametric_dataset(alphas, grid_resolution=50, output_path=None)
    dataset = ParametricPINNDataset(data_dict)
    
    trainer = ParametricPINNTrainer(
        model=model,
        dataset=dataset,
        learning_rate=2e-3,
        device=device
    )
    trainer.train(
        epochs=adam_epochs,
        lbfgs_iters=lbfgs_steps,
        lambda_data=1.0,
        lambda_phys=0.01,
        log_interval=100
    )
    model.eval()
    train_time = time.time() - t0
    return model, train_time


def run_phase8_and_9(
    n_samples: int = 5000,
    burn_in: int = 1000,
    alpha_true: float = 0.5000,
    noise_std: float = 0.0100,
    proposal_scale: float = 0.0500,
    output_dir_p8: str = "results/phase8_posterior_aware",
    output_dir_p9: str = "results/phase9_adaptive_refinement"
) -> Dict[str, Any]:
    os.makedirs(output_dir_p8, exist_ok=True)
    os.makedirs(output_dir_p9, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exact_op = ExactForwardOperator(resolution=100)
    x_sens = np.linspace(0.0, 1.0, 10)
    t_sens = np.array([0.1, 0.4, 0.7, 1.0])
    Xs, Ts = np.meshgrid(x_sens, t_sens, indexing="ij")
    obs_op = ObservationOperator(sensor_locations=np.column_stack([Xs.ravel(), Ts.ravel()]))
    
    u_true = exact_op(alpha_true).u
    y_clean = obs_op(u_true)
    
    rng = np.random.RandomState(101)
    y_obs = y_clean + rng.normal(0.0, noise_std, size=y_clean.shape)
    alpha_0 = 0.271404
    
    prior = Prior(LogNormalPrior(mu=float(np.log(0.5)), sigma=0.5))
    likelihood = GaussianLikelihood(noise_std=noise_std)
    posterior = Posterior(prior=prior, likelihood=likelihood, obs_operator=obs_op)
    proposal = GaussianRandomWalkProposal(scale=proposal_scale)
    
    # Baseline Control
    ctrl_sampler = TwoChainSampler(
        posterior=posterior,
        proposal=proposal,
        forward_solver_exact=exact_op,
        forward_solver_pinn=exact_op
    )
    ctrl_res = ctrl_sampler.run(y_obs=y_obs, n_samples=n_samples, initial_alpha=alpha_0, is_control=True, verbose=False, seed=101)
    s_ctrl1, s_ctrl2 = ctrl_res.get_post_burnin_samples(burn_in)
    ctrl_m = compute_bayesian_posterior_metrics(s_ctrl1, s_ctrl2, alpha_true=alpha_true)
    w1_ctrl_baseline = ctrl_m["discrepancy"]["wasserstein_1"]
    
    eval_alpha_grid = np.linspace(0.1134, 2.2050, 100)
    prior_density = np.array([np.exp(prior.log_prior(a)) for a in eval_alpha_grid])
    log_posts = np.array([posterior.evaluate_components(a, y_obs, exact_op)["log_posterior"] for a in eval_alpha_grid])
    max_log_post = np.max(log_posts[np.isfinite(log_posts)])
    exact_post_density = np.exp(log_posts - max_log_post)
    
    # =========================================================================
    # PHASE VIII: COMPARISON OF 3 TRAINING STRATEGIES (Equal Sample Budget N=20)
    # =========================================================================
    print("=" * 80, flush=True)
    print("PHASE VIII: UNIFORM VS PRIOR-WEIGHTED VS POSTERIOR-AWARE TRAINING", flush=True)
    print("=" * 80, flush=True)
    
    n_alpha_budget = 20
    
    # 1. Uniform
    alphas_uniform = np.exp(np.linspace(np.log(0.1134), np.log(2.2050), n_alpha_budget))
    # 2. Prior-weighted (LHS under LogNormal prior)
    u_lhs = np.linspace(0.01, 0.99, n_alpha_budget)
    alphas_prior = np.exp(np.log(0.5) + 0.5 * np.sqrt(2) * np.array([float(torch.erfinv(torch.tensor(2*u - 1, dtype=torch.float64))) for u in u_lhs]))
    # 3. Posterior-Aware (concentrated in [0.46, 0.54] with 4 anchor points in tails)
    alphas_post_aware = np.sort(np.concatenate([
        np.array([0.15, 0.25, 1.20, 1.80]),
        np.linspace(0.46, 0.54, n_alpha_budget - 4)
    ]))
    
    p8_configs = {
        "uniform_training": alphas_uniform,
        "prior_weighted_training": alphas_prior,
        "posterior_aware_training": alphas_post_aware
    }
    
    p8_records = []
    for name, a_pts in p8_configs.items():
        print(f"\n--- Strategy: {name.upper()} ---", flush=True)
        model, t_train = train_pinn_with_alphas(a_pts, adam_epochs=400, lbfgs_steps=25, device=device)
        torch.save(model.state_dict(), os.path.join(output_dir_p8, f"{name}.pth"))
        pinn_op = ParametricPINNForwardOperator(model=model, resolution=100, device=device)
        
        e_dict = evaluate_parameter_resolved_error(exact_op, pinn_op, eval_alpha_grid, obs_op)
        fwd_m = compute_global_forward_metrics(e_dict)
        weighted_m = compute_weighted_forward_errors(eval_alpha_grid, e_dict["e_alpha_field"], prior_density, exact_post_density)
        
        # MCMC
        res_sampler = TwoChainSampler(posterior=posterior, proposal=proposal, forward_solver_exact=exact_op, forward_solver_pinn=pinn_op)
        res_res = res_sampler.run(y_obs=y_obs, n_samples=n_samples, initial_alpha=alpha_0, is_control=False, verbose=False, seed=101)
        s_exact, s_pinn = res_res.get_post_burnin_samples(burn_in)
        post_m = compute_bayesian_posterior_metrics(s_exact, s_pinn, alpha_true=alpha_true)
        w1_res = post_m["discrepancy"]["wasserstein_1"]
        bfr = compute_bayesian_fidelity_ratio(w1_res, w1_ctrl_baseline)
        
        rec = {
            "strategy": name,
            "training_time_s": t_train,
            "mean_rel_l2_pct": fwd_m["mean_rel_l2"] * 100,
            "E_posterior_pct": weighted_m["E_posterior"] * 100,
            "exact_post_mean": post_m["exact"]["mean"],
            "pinn_post_mean": post_m["pinn"]["mean"],
            "mean_bias": post_m["discrepancy"]["mean_bias"],
            "var_ratio": post_m["discrepancy"]["variance_ratio"],
            "w1_research": w1_res,
            "bfr": bfr
        }
        p8_records.append(rec)
        print(f"  Train: {t_train:.1f}s | Global L2: {rec['mean_rel_l2_pct']:.2f}% | E_post: {rec['E_posterior_pct']:.2f}% | Bias: {rec['mean_bias']:.5e} | W1: {w1_res:.5e} | BFR: {bfr:.2f}", flush=True)
        
    df_p8 = pd.DataFrame(p8_records)
    df_p8.to_csv(os.path.join(output_dir_p8, "phase8_strategy_comparison.csv"), index=False)
    with open(os.path.join(output_dir_p8, "phase8_results.json"), "w") as f:
        json.dump({"strategies": p8_records}, f, indent=4)
        
    # =========================================================================
    # PHASE IX: MULTI-STAGE ADAPTIVE REFINEMENT LOOP
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("PHASE IX: MULTI-STAGE ADAPTIVE REFINEMENT LOOP", flush=True)
    print("=" * 80, flush=True)
    
    # Stage 0: Pilot PINN (Fast coarse budget: 10 parameter points, 150 Adam epochs, 0 LBFGS)
    print("\n--- [Stage 0] Training Pilot PINN ---", flush=True)
    alphas_stage0 = np.exp(np.linspace(np.log(0.1134), np.log(2.2050), 10))
    model_s0, t_train_s0 = train_pinn_with_alphas(alphas_stage0, adam_epochs=150, lbfgs_steps=0, device=device)
    pinn_op_s0 = ParametricPINNForwardOperator(model=model_s0, resolution=100, device=device)
    
    # Pilot Bayesian Inference
    print("--- [Stage 0] Running Pilot Bayesian MCMC ---", flush=True)
    sampler_s0 = TwoChainSampler(posterior=posterior, proposal=proposal, forward_solver_exact=exact_op, forward_solver_pinn=pinn_op_s0)
    res_s0 = sampler_s0.run(y_obs=y_obs, n_samples=n_samples, initial_alpha=alpha_0, is_control=False, verbose=False, seed=101)
    _, s_pinn_s0 = res_s0.get_post_burnin_samples(burn_in)
    
    mu_pilot = float(np.mean(s_pinn_s0))
    sigma_pilot = float(np.std(s_pinn_s0))
    print(f"  Pilot Posterior Estimate: alpha = {mu_pilot:.4f} +/- {sigma_pilot:.4f}", flush=True)
    
    e_dict_s0 = evaluate_parameter_resolved_error(exact_op, pinn_op_s0, eval_alpha_grid, obs_op)
    fwd_m_s0 = compute_global_forward_metrics(e_dict_s0)
    weighted_s0 = compute_weighted_forward_errors(eval_alpha_grid, e_dict_s0["e_alpha_field"], prior_density, exact_post_density)
    post_m_s0 = compute_bayesian_posterior_metrics(s_exact, s_pinn_s0, alpha_true=alpha_true)
    w1_s0 = post_m_s0["discrepancy"]["wasserstein_1"]
    bfr_s0 = compute_bayesian_fidelity_ratio(w1_s0, w1_ctrl_baseline)
    
    # Stage 1: Adaptive Refinement (Targeted collocation in [mu - 3*sigma, mu + 3*sigma])
    print(f"\n--- [Stage 1] Adaptive Collocation Refinement in [{mu_pilot - 3*sigma_pilot:.4f}, {mu_pilot + 3*sigma_pilot:.4f}] ---", flush=True)
    alphas_adaptive = np.sort(np.concatenate([
        np.array([0.15, 0.30, 0.90, 1.80]), # Coarse anchor points
        np.linspace(max(0.12, mu_pilot - 3*sigma_pilot), min(2.0, mu_pilot + 3*sigma_pilot), 16) # Targeted points
    ]))
    
    model_s1, t_train_s1 = train_pinn_with_alphas(alphas_adaptive, adam_epochs=350, lbfgs_steps=25, device=device)
    torch.save(model_s1.state_dict(), os.path.join(output_dir_p9, "adaptive_refined_pinn.pth"))
    pinn_op_s1 = ParametricPINNForwardOperator(model=model_s1, resolution=100, device=device)
    
    # Refined Bayesian Inference
    print("--- [Stage 1] Running Refined Bayesian MCMC ---", flush=True)
    sampler_s1 = TwoChainSampler(posterior=posterior, proposal=proposal, forward_solver_exact=exact_op, forward_solver_pinn=pinn_op_s1)
    res_s1 = sampler_s1.run(y_obs=y_obs, n_samples=n_samples, initial_alpha=alpha_0, is_control=False, verbose=False, seed=101)
    s_exact_s1, s_pinn_s1 = res_s1.get_post_burnin_samples(burn_in)
    
    e_dict_s1 = evaluate_parameter_resolved_error(exact_op, pinn_op_s1, eval_alpha_grid, obs_op)
    fwd_m_s1 = compute_global_forward_metrics(e_dict_s1)
    weighted_s1 = compute_weighted_forward_errors(eval_alpha_grid, e_dict_s1["e_alpha_field"], prior_density, exact_post_density)
    post_m_s1 = compute_bayesian_posterior_metrics(s_exact_s1, s_pinn_s1, alpha_true=alpha_true)
    w1_s1 = post_m_s1["discrepancy"]["wasserstein_1"]
    bfr_s1 = compute_bayesian_fidelity_ratio(w1_s1, w1_ctrl_baseline)
    
    stage_records = [
        {
            "stage": "Stage 0 (Pilot PINN)",
            "training_time_s": t_train_s0,
            "global_rel_l2_pct": fwd_m_s0["mean_rel_l2"] * 100,
            "E_posterior_pct": weighted_s0["E_posterior"] * 100,
            "post_mean": post_m_s0["pinn"]["mean"],
            "post_std": post_m_s0["pinn"]["std"],
            "mean_bias": post_m_s0["discrepancy"]["mean_bias"],
            "w1_research": w1_s0,
            "bfr": bfr_s0
        },
        {
            "stage": "Stage 1 (Adaptive Refined PINN)",
            "training_time_s": t_train_s0 + t_train_s1,
            "global_rel_l2_pct": fwd_m_s1["mean_rel_l2"] * 100,
            "E_posterior_pct": weighted_s1["E_posterior"] * 100,
            "post_mean": post_m_s1["pinn"]["mean"],
            "post_std": post_m_s1["pinn"]["std"],
            "mean_bias": post_m_s1["discrepancy"]["mean_bias"],
            "w1_research": w1_s1,
            "bfr": bfr_s1
        }
    ]
    
    df_p9 = pd.DataFrame(stage_records)
    df_p9.to_csv(os.path.join(output_dir_p9, "phase9_adaptive_progression.csv"), index=False)
    
    results_all = {
        "phase8_strategy_comparison": p8_records,
        "phase9_adaptive_progression": stage_records
    }
    with open(os.path.join(output_dir_p9, "phase9_adaptive_results.json"), "w") as f:
        json.dump(results_all, f, indent=4)
        
    print("\n" + "=" * 80, flush=True)
    print("PHASE VIII & IX SUMMARY RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"  Stage 0 (Pilot)   : E_post = {weighted_s0['E_posterior']*100:.2f}% | W1 = {w1_s0:.5e} | BFR = {bfr_s0:.2f}", flush=True)
    print(f"  Stage 1 (Refined) : E_post = {weighted_s1['E_posterior']*100:.2f}% | W1 = {w1_s1:.5e} | BFR = {bfr_s1:.2f}", flush=True)
    print(f"  Fidelity Gain     : BFR reduced by {(bfr_s0 - bfr_s1) / bfr_s0 * 100:.1f}%", flush=True)
    print(f"  Outputs saved to: {output_dir_p8} and {output_dir_p9}", flush=True)
    print("=" * 80, flush=True)
    
    return results_all


if __name__ == "__main__":
    run_phase8_and_9()
