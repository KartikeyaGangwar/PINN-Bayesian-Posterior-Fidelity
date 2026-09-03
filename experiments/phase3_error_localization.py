"""
Phase III & IV: Error Localization & Weighted Surrogate Diagnostics
===================================================================
Constructs 3 legitimate PINNs with comparable global forward error but distinct
parameter-space error localization:
  PINN-A: Posterior-focused (high accuracy near alpha ~ 0.5)
  PINN-B: Uniform coverage (uniform moderate error across alpha)
  PINN-C: Tail/boundary-focused (neglected posterior region)

Computes E_global, E_prior, E_posterior, RMS_posterior, and evaluates their
predictive power for Bayesian posterior fidelity (W1, bias, variance distortion).
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


def sample_targeted_parameters(strategy: str, n_samples: int = 25, seed: int = 42) -> np.ndarray:
    """Generates legitimate parameter training samples based on strategy."""
    rng = np.random.RandomState(seed)
    if strategy == "pinn_a_posterior_focused":
        # 60% in [0.42, 0.58], 40% in [0.11, 2.20]
        n_focus = int(0.6 * n_samples)
        n_rest = n_samples - n_focus
        alphas_focus = rng.uniform(0.42, 0.58, size=n_focus)
        alphas_rest = np.exp(rng.uniform(np.log(0.1134), np.log(2.2050), size=n_rest))
        return np.sort(np.concatenate([alphas_focus, alphas_rest]))
    elif strategy == "pinn_b_uniform":
        # Log-uniform across [0.1134, 2.2050]
        return np.sort(np.exp(rng.uniform(np.log(0.1134), np.log(2.2050), size=n_samples)))
    elif strategy == "pinn_c_tail_focused":
        # Neglect [0.45, 0.55], sample in [0.1134, 0.40] U [0.60, 2.2050]
        n_left = n_samples // 2
        n_right = n_samples - n_left
        alphas_left = rng.uniform(0.1134, 0.40, size=n_left)
        alphas_right = rng.uniform(0.60, 2.2050, size=n_right)
        return np.sort(np.concatenate([alphas_left, alphas_right]))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def train_localized_surrogate(
    strategy: str,
    device: torch.device,
    output_dir: str
) -> Tuple[str, ParametricModifiedMLP, np.ndarray]:
    model_path = os.path.join(output_dir, f"{strategy}.pth")
    model = ParametricModifiedMLP(n_input=3, n_output=1, n_hidden=64, n_layers=4, use_fourier=False).to(device).to(torch.float64)
    
    if os.path.exists(model_path):
        print(f"  [CACHE] Loading {strategy} from {model_path}", flush=True)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        train_alphas = sample_targeted_parameters(strategy, n_samples=25, seed=42)
        return model_path, model, train_alphas
        
    train_alphas = sample_targeted_parameters(strategy, n_samples=25, seed=42)
    data_dict = generate_exact_parametric_dataset(train_alphas, grid_resolution=50, output_path=None)
    dataset = ParametricPINNDataset(data_dict)
    
    trainer = ParametricPINNTrainer(
        model=model,
        dataset=dataset,
        learning_rate=2e-3,
        device=device
    )
    
    trainer.train(
        epochs=500,
        lbfgs_iters=35,
        lambda_data=1.0,
        lambda_phys=0.01,
        log_interval=100
    )
    
    model.eval()
    torch.save(model.state_dict(), model_path)
    return model_path, model, train_alphas


def run_phase3_and_4(
    n_samples: int = 5000,
    burn_in: int = 1000,
    alpha_true: float = 0.5000,
    noise_std: float = 0.0100,
    proposal_scale: float = 0.0500,
    output_dir: str = "results/phase3_error_localization"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Exact Forward and Observation Operators
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
    
    # Exact Baseline Control
    ctrl_sampler = TwoChainSampler(
        posterior=posterior,
        proposal=proposal,
        forward_solver_exact=exact_op,
        forward_solver_pinn=exact_op
    )
    ctrl_res = ctrl_sampler.run(y_obs=y_obs, n_samples=n_samples, initial_alpha=alpha_0, is_control=True, verbose=False, seed=101)
    s_ctrl1, s_ctrl2 = ctrl_res.get_post_burnin_samples(burn_in)
    ctrl_metrics = compute_bayesian_posterior_metrics(s_ctrl1, s_ctrl2, alpha_true=alpha_true)
    w1_ctrl_baseline = ctrl_metrics["discrepancy"]["wasserstein_1"]
    
    # Dense Alpha Grid for Parameter-Resolved & Weighted Error Analysis
    eval_alpha_grid = np.linspace(0.1134, 2.2050, 100)
    
    # Evaluate True Analytical Prior and Exact Posterior Densities on Grid
    prior_density = np.array([np.exp(prior.log_prior(a)) for a in eval_alpha_grid])
    log_posts = np.array([posterior.evaluate_components(a, y_obs, exact_op)["log_posterior"] for a in eval_alpha_grid])
    max_log_post = np.max(log_posts[np.isfinite(log_posts)])
    exact_post_density = np.exp(log_posts - max_log_post)
    
    strategies = [
        "pinn_a_posterior_focused",
        "pinn_b_uniform",
        "pinn_c_tail_focused"
    ]
    
    localization_records = []
    e_alpha_curves = {}
    
    print("=" * 80, flush=True)
    print("PHASE III & IV: ERROR LOCALIZATION & WEIGHTED SURROGATE DIAGNOSTICS", flush=True)
    print("=" * 80, flush=True)
    print(f"  Exact Baseline W1: {w1_ctrl_baseline:.6e}", flush=True)
    print("=" * 80, flush=True)
    
    for strat in strategies:
        print(f"\n--- Processing Strategy: {strat.upper()} ---", flush=True)
        model_path, model, train_alphas = train_localized_surrogate(strat, device, output_dir)
        pinn_op = ParametricPINNForwardOperator(model=model, resolution=100, device=device)
        
        # 1. Parameter-Resolved Error
        e_dict = evaluate_parameter_resolved_error(
            exact_solver=exact_op,
            pinn_solver=pinn_op,
            alpha_grid=eval_alpha_grid,
            obs_operator=obs_op
        )
        fwd_metrics = compute_global_forward_metrics(e_dict)
        weighted_errors = compute_weighted_forward_errors(
            alpha_grid=eval_alpha_grid,
            e_alpha=e_dict["e_alpha_field"],
            prior_density=prior_density,
            exact_posterior_density=exact_post_density
        )
        
        e_alpha_curves[strat] = {
            "alpha_grid": eval_alpha_grid.tolist(),
            "e_alpha_field": e_dict["e_alpha_field"].tolist(),
            "e_alpha_obs": e_dict["e_alpha_obs"].tolist(),
            "train_alphas": train_alphas.tolist()
        }
        
        # 2. Bayesian Inference
        res_sampler = TwoChainSampler(
            posterior=posterior,
            proposal=proposal,
            forward_solver_exact=exact_op,
            forward_solver_pinn=pinn_op
        )
        res_res = res_sampler.run(y_obs=y_obs, n_samples=n_samples, initial_alpha=alpha_0, is_control=False, verbose=False, seed=101)
        s_exact, s_pinn = res_res.get_post_burnin_samples(burn_in)
        
        post_metrics = compute_bayesian_posterior_metrics(s_exact, s_pinn, alpha_true=alpha_true)
        w1_res = post_metrics["discrepancy"]["wasserstein_1"]
        bfr = compute_bayesian_fidelity_ratio(w1_research=w1_res, w1_control=w1_ctrl_baseline)
        
        # Measure localized error at nominal alpha = 0.50
        u_e_05 = exact_op(0.50).u
        u_p_05 = pinn_op(0.50).u
        e_at_05 = float(np.linalg.norm(u_p_05 - u_e_05) / np.linalg.norm(u_e_05))
        
        rec = {
            "strategy": strat,
            "global_rel_l2_pct": fwd_metrics["mean_rel_l2"] * 100,
            "error_at_alpha_0_5_pct": e_at_05 * 100,
            "E_global_pct": weighted_errors["E_global"] * 100,
            "E_prior_pct": weighted_errors["E_prior"] * 100,
            "E_posterior_pct": weighted_errors["E_posterior"] * 100,
            "RMS_posterior_pct": weighted_errors["RMS_posterior"] * 100,
            "exact_post_mean": post_metrics["exact"]["mean"],
            "pinn_post_mean": post_metrics["pinn"]["mean"],
            "mean_bias": post_metrics["discrepancy"]["mean_bias"],
            "bias_in_std_units": post_metrics["discrepancy"]["bias_in_std_units"],
            "var_ratio": post_metrics["discrepancy"]["variance_ratio"],
            "w1_research": w1_res,
            "w1_control_baseline": w1_ctrl_baseline,
            "bfr": bfr,
            "ks_statistic": post_metrics["discrepancy"]["ks_statistic"]
        }
        localization_records.append(rec)
        print(f"  Global Rel L2: {rec['global_rel_l2_pct']:.2f}% | Local Error at 0.5: {rec['error_at_alpha_0_5_pct']:.2f}% | E_post: {rec['E_posterior_pct']:.2f}% | W1: {w1_res:.5e} | BFR: {bfr:.2f}", flush=True)

    df = pd.DataFrame(localization_records)
    csv_path = os.path.join(output_dir, "phase3_localization_summary.csv")
    df.to_csv(csv_path, index=False)
    
    # Diagnostic correlations
    corr_global_w1 = float(df["E_global_pct"].corr(df["w1_research"]))
    corr_prior_w1 = float(df["E_prior_pct"].corr(df["w1_research"]))
    corr_post_w1 = float(df["E_posterior_pct"].corr(df["w1_research"]))
    corr_rms_post_w1 = float(df["RMS_posterior_pct"].corr(df["w1_research"]))
    
    summary = {
        "experiments": localization_records,
        "curves": e_alpha_curves,
        "diagnostic_correlations_with_W1": {
            "E_global": corr_global_w1,
            "E_prior": corr_prior_w1,
            "E_posterior": corr_post_w1,
            "RMS_posterior": corr_rms_post_w1
        }
    }
    
    json_path = os.path.join(output_dir, "phase3_localization_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n" + "=" * 80, flush=True)
    print("PHASE III & IV SUMMARY: WHICH DIAGNOSTIC PREDICTS POSTERIOR FIDELITY?", flush=True)
    print("=" * 80, flush=True)
    print(f"  Correlation with Posterior W1:")
    print(f"    E_global        : r = {corr_global_w1:+.4f}")
    print(f"    E_prior         : r = {corr_prior_w1:+.4f}")
    print(f"    E_posterior     : r = {corr_post_w1:+.4f}")
    print(f"    RMS_posterior   : r = {corr_rms_post_w1:+.4f}")
    print(f"  Outputs saved to: {output_dir}", flush=True)
    print("=" * 80, flush=True)
    
    return summary


if __name__ == "__main__":
    run_phase3_and_4()
