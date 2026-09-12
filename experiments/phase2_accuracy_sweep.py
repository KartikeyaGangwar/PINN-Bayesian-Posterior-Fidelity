"""
Phase II: Forward Error vs Posterior Error Sweep
================================================
Constructs 6 surrogate accuracy configurations ranging from underconverged to converged
and computes global forward metrics vs Bayesian posterior discrepancy metrics.
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
from parametric_surrogate.parameter_sampler import sample_parameter_domain_lhs
from parametric_surrogate.dataset import ParametricPINNDataset
from parametric_surrogate.trainer import ParametricPINNTrainer

from experiments.metrics import (
    evaluate_parameter_resolved_error,
    compute_global_forward_metrics,
    compute_bayesian_posterior_metrics,
    compute_bayesian_fidelity_ratio
)


def train_surrogate_level(
    level_name: str,
    n_alpha_samples: int,
    adam_epochs: int,
    lbfgs_steps: int,
    device: torch.device,
    output_dir: str
) -> Tuple[str, ParametricModifiedMLP]:
    """Trains or loads a surrogate model for a specific convergence tier."""
    model_path = os.path.join(output_dir, f"pinn_{level_name}.pth")
    model = ParametricModifiedMLP(n_input=3, n_output=1, n_hidden=64, n_layers=4, use_fourier=False).to(device).to(torch.float64)
    
    # If checkpoint already exists, load and return
    if os.path.exists(model_path):
        print(f"  Loading pre-trained checkpoint from {model_path}", flush=True)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        return model_path, model
        
    # If level 6 (highly accurate), load canonical weights if available
    canonical_weights = os.path.join("results", "heat_equation_pinn.pth")
    if level_name == "level6_highly_accurate" and os.path.exists(canonical_weights):
        print(f"  Loading reference model weights from {canonical_weights}", flush=True)
        model.load_state_dict(torch.load(canonical_weights, map_location=device, weights_only=True))
        model.eval()
        torch.save(model.state_dict(), model_path)
        return model_path, model
        
    prior = Prior(LogNormalPrior(mu=float(np.log(0.5)), sigma=0.5))
    alphas, _, _, _ = sample_parameter_domain_lhs(prior, n_samples=n_alpha_samples, seed=42)
    
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
    torch.save(model.state_dict(), model_path)
    return model_path, model


def run_phase2_accuracy_sweep(
    n_samples: int = 5000,
    burn_in: int = 1000,
    alpha_true: float = 0.5000,
    noise_std: float = 0.0100,
    proposal_scale: float = 0.0500,
    output_dir: str = "results/phase2_accuracy_sweep"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Forward Setup & Canonical Control Baseline
    exact_op = ExactForwardOperator(resolution=100)
    x_sens = np.linspace(0.0, 1.0, 10)
    t_sens = np.array([0.1, 0.4, 0.7, 1.0])
    Xs, Ts = np.meshgrid(x_sens, t_sens, indexing="ij")
    sensor_locs = np.column_stack([Xs.ravel(), Ts.ravel()])
    obs_op = ObservationOperator(sensor_locations=sensor_locs)
    
    u_true = exact_op(alpha_true).u
    y_clean = obs_op(u_true)
    
    # Fixed representative noise realization (seed 101)
    rng = np.random.RandomState(101)
    y_obs = y_clean + rng.normal(0.0, noise_std, size=y_clean.shape)
    alpha_0 = 0.271404
    
    prior = Prior(LogNormalPrior(mu=float(np.log(0.5)), sigma=0.5))
    likelihood = GaussianLikelihood(noise_std=noise_std)
    posterior = Posterior(prior=prior, likelihood=likelihood, obs_operator=obs_op)
    proposal = GaussianRandomWalkProposal(scale=proposal_scale)
    
    # Run Control Baseline (Exact vs Exact)
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
    
    levels_config = [
        {"name": "level1_very_poor", "n_alphas": 5, "adam": 30, "lbfgs": 0},
        {"name": "level2_poor", "n_alphas": 8, "adam": 100, "lbfgs": 0},
        {"name": "level3_moderate", "n_alphas": 15, "adam": 250, "lbfgs": 15},
        {"name": "level4_good", "n_alphas": 20, "adam": 450, "lbfgs": 40},
        {"name": "level5_very_good", "n_alphas": 25, "adam": 550, "lbfgs": 30},
        {"name": "level6_highly_accurate", "n_alphas": 30, "adam": 800, "lbfgs": 150}
    ]
    
    eval_alpha_grid = np.linspace(0.1134, 2.2050, 100)
    
    sweep_records = []
    print("=" * 80, flush=True)
    print("PHASE II: FORWARD ERROR VS POSTERIOR DISCREPANCY SWEEP", flush=True)
    print("=" * 80, flush=True)
    print(f"  Exact-vs-Exact Stochastic Baseline W1: {w1_ctrl_baseline:.6e}", flush=True)
    print("=" * 80, flush=True)
    
    for cfg in levels_config:
        lvl_name = cfg["name"]
        t0 = time.time()
        print(f"\n--- Processing {lvl_name.upper()} ---", flush=True)
        model_path, model = train_surrogate_level(
            level_name=lvl_name,
            n_alpha_samples=cfg["n_alphas"],
            adam_epochs=cfg["adam"],
            lbfgs_steps=cfg["lbfgs"],
            device=device,
            output_dir=output_dir
        )
        pinn_op = ParametricPINNForwardOperator(model=model, resolution=100, device=device)
        train_time = time.time() - t0
        
        # 1. Global Forward Accuracy Evaluation
        e_dict = evaluate_parameter_resolved_error(
            exact_solver=exact_op,
            pinn_solver=pinn_op,
            alpha_grid=eval_alpha_grid,
            obs_operator=obs_op
        )
        fwd_metrics = compute_global_forward_metrics(e_dict)
        
        # 2. Bayesian Inference
        t_mcmc = time.time()
        res_sampler = TwoChainSampler(
            posterior=posterior,
            proposal=proposal,
            forward_solver_exact=exact_op,
            forward_solver_pinn=pinn_op
        )
        res_res = res_sampler.run(y_obs=y_obs, n_samples=n_samples, initial_alpha=alpha_0, is_control=False, verbose=False, seed=101)
        s_exact, s_pinn = res_res.get_post_burnin_samples(burn_in)
        mcmc_time = time.time() - t_mcmc
        
        post_metrics = compute_bayesian_posterior_metrics(s_exact, s_pinn, alpha_true=alpha_true)
        w1_res = post_metrics["discrepancy"]["wasserstein_1"]
        bfr = compute_bayesian_fidelity_ratio(w1_research=w1_res, w1_control=w1_ctrl_baseline)
        
        record = {
            "level": lvl_name,
            "adam_epochs": cfg["adam"],
            "lbfgs_steps": cfg["lbfgs"],
            "n_alpha_samples": cfg["n_alphas"],
            "mean_rel_l2_pct": fwd_metrics["mean_rel_l2"] * 100,
            "median_rel_l2_pct": fwd_metrics["median_rel_l2"] * 100,
            "p95_rel_l2_pct": fwd_metrics["p95_rel_l2"] * 100,
            "max_rel_l2_pct": fwd_metrics["max_rel_l2"] * 100,
            "mean_max_abs_error": fwd_metrics["mean_max_abs_error"],
            "mean_obs_error_pct": fwd_metrics["mean_obs_error"] * 100,
            "median_obs_error_pct": fwd_metrics["median_obs_error"] * 100,
            "p95_obs_error_pct": fwd_metrics["p95_obs_error"] * 100,
            "max_obs_error_pct": fwd_metrics["max_obs_error"] * 100,
            "exact_post_mean": post_metrics["exact"]["mean"],
            "exact_post_std": post_metrics["exact"]["std"],
            "pinn_post_mean": post_metrics["pinn"]["mean"],
            "pinn_post_std": post_metrics["pinn"]["std"],
            "posterior_mean_bias": post_metrics["discrepancy"]["mean_bias"],
            "relative_mean_bias_pct": post_metrics["discrepancy"]["relative_mean_bias"] * 100,
            "bias_in_std_units": post_metrics["discrepancy"]["bias_in_std_units"],
            "variance_ratio": post_metrics["discrepancy"]["variance_ratio"],
            "w1_research": w1_res,
            "w1_control_baseline": w1_ctrl_baseline,
            "bfr": bfr,
            "ks_statistic": post_metrics["discrepancy"]["ks_statistic"],
            "ks_p_value": post_metrics["discrepancy"]["ks_p_value"],
            "training_time_s": train_time,
            "mcmc_time_s": mcmc_time
        }
        sweep_records.append(record)
        print(f"  Forward Rel L2: {record['mean_rel_l2_pct']:.2f}% | Obs Error: {record['mean_obs_error_pct']:.2f}% | Post Bias: {record['posterior_mean_bias']:.5e} | W1: {w1_res:.5e} | BFR: {bfr:.2f}", flush=True)

    df = pd.DataFrame(sweep_records)
    csv_path = os.path.join(output_dir, "phase2_accuracy_sweep_summary.csv")
    df.to_csv(csv_path, index=False)
    
    # Compute correlation
    corr_fwd_w1 = float(df["mean_rel_l2_pct"].corr(df["w1_research"]))
    corr_obs_w1 = float(df["mean_obs_error_pct"].corr(df["w1_research"]))
    corr_fwd_bias = float(df["mean_rel_l2_pct"].corr(df["posterior_mean_bias"].abs()))
    
    summary = {
        "levels": sweep_records,
        "correlations": {
            "pearson_fwd_rel_l2_vs_w1": corr_fwd_w1,
            "pearson_obs_error_vs_w1": corr_obs_w1,
            "pearson_fwd_rel_l2_vs_abs_bias": corr_fwd_bias
        }
    }
    
    json_path = os.path.join(output_dir, "phase2_accuracy_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n" + "=" * 80, flush=True)
    print("PHASE II SUMMARY & CORRELATION ANALYSIS", flush=True)
    print("=" * 80, flush=True)
    print(f"  Pearson Correlation (Forward L2 Error vs Posterior W1): r = {corr_fwd_w1:.4f}", flush=True)
    print(f"  Pearson Correlation (Obs Error vs Posterior W1)       : r = {corr_obs_w1:.4f}", flush=True)
    print(f"  Pearson Correlation (Forward L2 Error vs |Bias|)       : r = {corr_fwd_bias:.4f}", flush=True)
    print(f"  Outputs saved to: {output_dir}", flush=True)
    print("=" * 80, flush=True)
    
    return summary


if __name__ == "__main__":
    run_phase2_accuracy_sweep()
