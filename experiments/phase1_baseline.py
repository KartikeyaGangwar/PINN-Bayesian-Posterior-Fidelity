"""
Phase I: Reproducible Baseline with Controlled Noise Realizations
================================================================
Evaluates 10 independent noise realizations for paired Exact-vs-PINN and
Exact-vs-Exact control inference to establish the empirical stochastic baseline
and compute the proposed Bayesian Fidelity Ratio (BFR).
"""

import os, sys
sys.path.insert(0, os.getcwd())
import time
import json
import numpy as np
import torch
import pandas as pd
from typing import Dict, Any, List

from bayesian.forward_operator import ExactForwardOperator, ParametricPINNForwardOperator, ParametricModifiedMLP
from bayesian.observation_operator import ObservationOperator
from bayesian.prior import LogNormalPrior, Prior
from bayesian.likelihood import GaussianLikelihood
from bayesian.posterior import Posterior
from bayesian.proposal import GaussianRandomWalkProposal
from bayesian.two_chain_sampler import TwoChainSampler
from experiments.metrics import compute_bayesian_posterior_metrics, compute_bayesian_fidelity_ratio


def run_phase1_baseline(
    n_realizations: int = 10,
    n_samples: int = 5000,
    burn_in: int = 1000,
    alpha_true: float = 0.5000,
    noise_std: float = 0.0100,
    proposal_scale: float = 0.0500,
    output_dir: str = "results/phase1_baseline"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Forward Solvers
    exact_op = ExactForwardOperator(resolution=100)
    weights_path = os.path.join("results", "heat_equation_pinn.pth")
    model = ParametricModifiedMLP(n_input=3, n_output=1, n_hidden=64, n_layers=4, use_fourier=False).to(device).to(torch.float64)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    pinn_op = ParametricPINNForwardOperator(model=model, resolution=100, device=device)
    
    # 2. Sensor Layout (M=40)
    x_sens = np.linspace(0.0, 1.0, 10)
    t_sens = np.array([0.1, 0.4, 0.7, 1.0])
    Xs, Ts = np.meshgrid(x_sens, t_sens, indexing="ij")
    sensor_locs = np.column_stack([Xs.ravel(), Ts.ravel()])
    obs_op = ObservationOperator(sensor_locations=sensor_locs)
    
    # 3. Clean True Field & Clean Observation Vector
    u_true = exact_op(alpha_true).u
    y_clean = obs_op(u_true)
    
    prior = Prior(LogNormalPrior(mu=float(np.log(0.5)), sigma=0.5))
    likelihood = GaussianLikelihood(noise_std=noise_std)
    posterior = Posterior(prior=prior, likelihood=likelihood, obs_operator=obs_op)
    proposal = GaussianRandomWalkProposal(scale=proposal_scale)
    
    sampler = TwoChainSampler(
        posterior=posterior,
        proposal=proposal,
        forward_solver_exact=exact_op,
        forward_solver_pinn=pinn_op
    )
    
    realization_records: List[Dict[str, Any]] = []
    w1_research_list = []
    w1_control_list = []
    bfr_list = []
    bias_list = []
    
    print("=" * 80)
    print("PHASE I: REPRODUCIBLE BASELINE (10 INDEPENDENT NOISE REALIZATIONS)")
    print("=" * 80)
    print(f"  alpha_true   : {alpha_true:.4f}")
    print(f"  noise_std    : {noise_std:.4f}")
    print(f"  M sensors    : {len(y_clean)}")
    print(f"  MCMC steps   : {n_samples:,} (burn-in: {burn_in:,})")
    print(f"  Realizations : {n_realizations}")
    print("=" * 80)
    
    seeds = [100 + k for k in range(1, n_realizations + 1)]
    
    all_exact_post = []
    all_pinn_post = []
    all_ctrl1_post = []
    all_ctrl2_post = []
    
    for i, seed in enumerate(seeds, 1):
        rng = np.random.RandomState(seed)
        noise = rng.normal(0.0, noise_std, size=y_clean.shape)
        y_obs = y_clean + noise
        
        # Common initial state for this realization
        alpha_0 = float(np.exp(rng.normal(np.log(0.5), 0.5)))
        
        # A. Run Control Experiment (Exact vs Exact)
        ctrl_res = sampler.run(
            y_obs=y_obs,
            n_samples=n_samples,
            initial_alpha=alpha_0,
            is_control=True,
            verbose=False,
            seed=seed
        )
        s_ctrl1, s_ctrl2 = ctrl_res.get_post_burnin_samples(burn_in)
        ctrl_metrics = compute_bayesian_posterior_metrics(s_ctrl1, s_ctrl2, alpha_true=alpha_true)
        w1_ctrl = ctrl_metrics["discrepancy"]["wasserstein_1"]
        
        # B. Run Research Experiment (Exact vs PINN)
        res_res = sampler.run(
            y_obs=y_obs,
            n_samples=n_samples,
            initial_alpha=alpha_0,
            is_control=False,
            verbose=False,
            seed=seed
        )
        s_exact, s_pinn = res_res.get_post_burnin_samples(burn_in)
        res_metrics = compute_bayesian_posterior_metrics(s_exact, s_pinn, alpha_true=alpha_true)
        w1_res = res_metrics["discrepancy"]["wasserstein_1"]
        
        bfr = compute_bayesian_fidelity_ratio(w1_research=w1_res, w1_control=w1_ctrl)
        
        bias = res_metrics["discrepancy"]["mean_bias"]
        var_ratio = res_metrics["discrepancy"]["variance_ratio"]
        ks_stat = res_metrics["discrepancy"]["ks_statistic"]
        ks_pval = res_metrics["discrepancy"]["ks_p_value"]
        
        w1_research_list.append(w1_res)
        w1_control_list.append(w1_ctrl)
        bfr_list.append(bfr)
        bias_list.append(bias)
        
        all_exact_post.append(s_exact)
        all_pinn_post.append(s_pinn)
        all_ctrl1_post.append(s_ctrl1)
        all_ctrl2_post.append(s_ctrl2)
        
        rec = {
            "realization_idx": i,
            "seed": seed,
            "alpha_0": alpha_0,
            "exact_mean": res_metrics["exact"]["mean"],
            "exact_std": res_metrics["exact"]["std"],
            "exact_ci95": res_metrics["exact"]["ci_95"],
            "pinn_mean": res_metrics["pinn"]["mean"],
            "pinn_std": res_metrics["pinn"]["std"],
            "pinn_ci95": res_metrics["pinn"]["ci_95"],
            "mean_bias": bias,
            "rel_bias_pct": res_metrics["discrepancy"]["relative_mean_bias"] * 100,
            "bias_in_std_units": res_metrics["discrepancy"]["bias_in_std_units"],
            "var_ratio": var_ratio,
            "w1_research": w1_res,
            "w1_control_baseline": w1_ctrl,
            "bfr": bfr,
            "ks_statistic": ks_stat,
            "ks_p_value": ks_pval,
            "exact_covered_true": res_metrics["exact"]["covered_true"],
            "pinn_covered_true": res_metrics["pinn"]["covered_true"]
        }
        realization_records.append(rec)
        print(f"  [Realization {i:02d}/10] Seed: {seed} | Exact: {rec['exact_mean']:.5f} | PINN: {rec['pinn_mean']:.5f} | W1(Res): {w1_res:.5e} | W1(Ctrl): {w1_ctrl:.5e} | BFR: {bfr:.2f}")

    df = pd.DataFrame(realization_records)
    csv_path = os.path.join(output_dir, "phase1_baseline_summary.csv")
    df.to_csv(csv_path, index=False)
    
    summary = {
        "n_realizations": n_realizations,
        "n_samples": n_samples,
        "burn_in": burn_in,
        "noise_std": noise_std,
        "alpha_true": alpha_true,
        "w1_research": {
            "mean": float(np.mean(w1_research_list)),
            "std": float(np.std(w1_research_list)),
            "median": float(np.median(w1_research_list)),
            "min": float(np.min(w1_research_list)),
            "max": float(np.max(w1_research_list))
        },
        "w1_control_baseline": {
            "mean": float(np.mean(w1_control_list)),
            "std": float(np.std(w1_control_list)),
            "median": float(np.median(w1_control_list)),
            "min": float(np.min(w1_control_list)),
            "max": float(np.max(w1_control_list))
        },
        "bfr": {
            "mean": float(np.mean(bfr_list)),
            "std": float(np.std(bfr_list)),
            "median": float(np.median(bfr_list)),
            "min": float(np.min(bfr_list)),
            "max": float(np.max(bfr_list))
        },
        "mean_bias": {
            "mean": float(np.mean(bias_list)),
            "std": float(np.std(bias_list)),
            "median": float(np.median(bias_list))
        },
        "realizations": realization_records
    }
    
    json_path = os.path.join(output_dir, "phase1_baseline_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    npz_path = os.path.join(output_dir, "phase1_baseline_realizations.npz")
    np.savez_compressed(
        npz_path,
        exact_posteriors=np.array(all_exact_post),
        pinn_posteriors=np.array(all_pinn_post),
        ctrl1_posteriors=np.array(all_ctrl1_post),
        ctrl2_posteriors=np.array(all_ctrl2_post),
        w1_research=np.array(w1_research_list),
        w1_control=np.array(w1_control_list),
        bfr=np.array(bfr_list),
        seeds=np.array(seeds)
    )
    
    print("\n" + "=" * 80)
    print("PHASE I SUMMARY STATISTICS (OVER 10 REALIZATIONS)")
    print("=" * 80)
    print(f"  Mean W1 (Research)       : {summary['w1_research']['mean']:.6e} +/- {summary['w1_research']['std']:.6e}")
    print(f"  Mean W1 (Control Baseline): {summary['w1_control_baseline']['mean']:.6e} +/- {summary['w1_control_baseline']['std']:.6e}")
    print(f"  Mean BFR (Fidelity Ratio): {summary['bfr']['mean']:.2f} +/- {summary['bfr']['std']:.2f} (Median: {summary['bfr']['median']:.2f})")
    print(f"  Mean Posterior Bias      : {summary['mean_bias']['mean']:.6e} +/- {summary['mean_bias']['std']:.6e}")
    print(f"  Outputs saved to: {output_dir}")
    print("=" * 80)
    
    return summary


if __name__ == "__main__":
    run_phase1_baseline()
