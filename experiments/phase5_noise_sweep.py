"""
Phase V: Observation Noise Sweep
================================
Studies how observation noise level sigma_noise in {0.001, 0.005, 0.01, 0.02, 0.05, 0.10}
modulates the sensitivity of Bayesian posterior inference to PINN surrogate error.
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


def run_phase5_noise_sweep(
    noise_levels: List[float] = [0.001, 0.005, 0.010, 0.020, 0.050, 0.100],
    n_realizations: int = 5,
    n_samples: int = 3000,
    burn_in: int = 500,
    alpha_true: float = 0.5000,
    output_dir: str = "results/phase5_noise_sweep"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exact_op = ExactForwardOperator(resolution=100)
    weights_path = os.path.join("results", "heat_equation_pinn.pth")
    model = ParametricModifiedMLP(n_input=3, n_output=1, n_hidden=64, n_layers=4, use_fourier=False).to(device).to(torch.float64)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    pinn_op = ParametricPINNForwardOperator(model=model, resolution=100, device=device)
    
    x_sens = np.linspace(0.0, 1.0, 10)
    t_sens = np.array([0.1, 0.4, 0.7, 1.0])
    Xs, Ts = np.meshgrid(x_sens, t_sens, indexing="ij")
    obs_op = ObservationOperator(sensor_locations=np.column_stack([Xs.ravel(), Ts.ravel()]))
    
    u_true = exact_op(alpha_true).u
    y_clean = obs_op(u_true)
    prior = Prior(LogNormalPrior(mu=float(np.log(0.5)), sigma=0.5))
    
    noise_summary_records = []
    all_realization_records = []
    
    print("=" * 80, flush=True)
    print("PHASE V: OBSERVATION NOISE SWEEP EXPERIMENT", flush=True)
    print("=" * 80, flush=True)
    print(f"  Noise levels: {noise_levels}", flush=True)
    print(f"  Realizations per level: {n_realizations}", flush=True)
    print("=" * 80, flush=True)
    
    for sigma in noise_levels:
        print(f"\n--- Testing Observation Noise Level: sigma = {sigma:.4f} ---", flush=True)
        likelihood = GaussianLikelihood(noise_std=sigma)
        posterior = Posterior(prior=prior, likelihood=likelihood, obs_operator=obs_op)
        # Adapt proposal scale to noise level for standard mixing
        prop_scale_adapted = min(0.06, max(0.005, 0.05 * (sigma / 0.01) ** 0.5))
        proposal = GaussianRandomWalkProposal(scale=prop_scale_adapted)
        
        sampler = TwoChainSampler(
            posterior=posterior,
            proposal=proposal,
            forward_solver_exact=exact_op,
            forward_solver_pinn=pinn_op
        )
        
        w1_res_list = []
        w1_ctrl_list = []
        bfr_list = []
        bias_list = []
        var_ratio_list = []
        exact_std_list = []
        
        for k in range(1, n_realizations + 1):
            seed = int(sigma * 100000) + k
            rng = np.random.RandomState(seed)
            y_obs = y_clean + rng.normal(0.0, sigma, size=y_clean.shape)
            alpha_0 = 0.60  # Controlled starting value
            
            # Control baseline
            ctrl_res = sampler.run(y_obs=y_obs, n_samples=n_samples, initial_alpha=alpha_0, is_control=True, verbose=False, seed=seed)
            s_ctrl1, s_ctrl2 = ctrl_res.get_post_burnin_samples(burn_in)
            ctrl_m = compute_bayesian_posterior_metrics(s_ctrl1, s_ctrl2, alpha_true=alpha_true)
            w1_ctrl = ctrl_m["discrepancy"]["wasserstein_1"]
            
            # Research experiment
            res_res = sampler.run(y_obs=y_obs, n_samples=n_samples, initial_alpha=alpha_0, is_control=False, verbose=False, seed=seed)
            s_exact, s_pinn = res_res.get_post_burnin_samples(burn_in)
            res_m = compute_bayesian_posterior_metrics(s_exact, s_pinn, alpha_true=alpha_true)
            w1_res = res_m["discrepancy"]["wasserstein_1"]
            
            bfr = compute_bayesian_fidelity_ratio(w1_research=w1_res, w1_control=w1_ctrl)
            bias = res_m["discrepancy"]["mean_bias"]
            v_ratio = res_m["discrepancy"]["variance_ratio"]
            e_std = res_m["exact"]["std"]
            
            w1_res_list.append(w1_res)
            w1_ctrl_list.append(w1_ctrl)
            bfr_list.append(bfr)
            bias_list.append(bias)
            var_ratio_list.append(v_ratio)
            exact_std_list.append(e_std)
            
            all_realization_records.append({
                "sigma_noise": sigma,
                "realization": k,
                "seed": seed,
                "exact_mean": res_m["exact"]["mean"],
                "exact_std": e_std,
                "pinn_mean": res_m["pinn"]["mean"],
                "pinn_std": res_m["pinn"]["std"],
                "mean_bias": bias,
                "bias_in_std_units": res_m["discrepancy"]["bias_in_std_units"],
                "var_ratio": v_ratio,
                "w1_research": w1_res,
                "w1_control": w1_ctrl,
                "bfr": bfr
            })
            print(f"  [sigma={sigma:.4f} | Run {k}/5] Exact: {res_m['exact']['mean']:.4f}+/-{e_std:.4f} | PINN: {res_m['pinn']['mean']:.4f} | W1(Res): {w1_res:.4e} | BFR: {bfr:.2f}", flush=True)
            
        rec_summary = {
            "sigma_noise": sigma,
            "mean_exact_std": float(np.mean(exact_std_list)),
            "mean_w1_research": float(np.mean(w1_res_list)),
            "std_w1_research": float(np.std(w1_res_list)),
            "mean_w1_control": float(np.mean(w1_ctrl_list)),
            "mean_bfr": float(np.mean(bfr_list)),
            "std_bfr": float(np.std(bfr_list)),
            "median_bfr": float(np.median(bfr_list)),
            "mean_bias": float(np.mean(bias_list)),
            "mean_abs_bias": float(np.mean(np.abs(bias_list))),
            "mean_var_ratio": float(np.mean(var_ratio_list))
        }
        noise_summary_records.append(rec_summary)
        print(f"  => AVERAGE sigma={sigma:.4f} | Post Std: {rec_summary['mean_exact_std']:.5f} | W1(Res): {rec_summary['mean_w1_research']:.5e} | BFR: {rec_summary['mean_bfr']:.2f}", flush=True)

    df_sum = pd.DataFrame(noise_summary_records)
    csv_path = os.path.join(output_dir, "phase5_noise_sweep_summary.csv")
    df_sum.to_csv(csv_path, index=False)
    
    df_all = pd.DataFrame(all_realization_records)
    df_all.to_csv(os.path.join(output_dir, "phase5_noise_sweep_all_realizations.csv"), index=False)
    
    results_json = {
        "noise_levels": noise_summary_records,
        "all_realizations": all_realization_records
    }
    with open(os.path.join(output_dir, "phase5_noise_sweep_results.json"), "w") as f:
        json.dump(results_json, f, indent=4)
        
    print("\n" + "=" * 80, flush=True)
    print("PHASE V SUMMARY: OBSERVATION NOISE MODULATION", flush=True)
    print("=" * 80, flush=True)
    print(f"  Outputs saved to: {output_dir}", flush=True)
    print("=" * 80, flush=True)
    
    return results_json


if __name__ == "__main__":
    run_phase5_noise_sweep()
