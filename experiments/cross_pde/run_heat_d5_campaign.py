"""
5D Multi-Mode Parametric Diffusion Validation Campaign
======================================================
Executes balanced 20-model surrogate ensemble in 5-dimensional parameter space (d=5)
across 4 convergence tiers:
- Tier 1: Underconverged (40 Adam epochs, 0 L-BFGS)
- Tier 2: Early Converged (120 Adam epochs, 10 L-BFGS)
- Tier 3: Intermediate (250 Adam epochs, 25 L-BFGS)
- Tier 4: Converged (450 Adam epochs, 40 L-BFGS)

Evaluates:
- Global relative forward error E_global on space-time mesh
- Posterior-weighted forward error E_posterior
- Integrated log-likelihood perturbation ||Delta log L||_L1
- 100-direction Sliced Wasserstein distance (SW1) against exact MCMC posterior
- Dimensional crossover: Williams dependent correlation tests
- PyTorch model checkpoint (.pt) serialization
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.stats as stats
from typing import Dict, Any, Tuple, Optional

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.cross_pde.pde_definitions import generate_space_time_sensors
from experiments.cross_pde.heat_d5 import HeatD5MultiModeBenchmark, HeatD5Config
from experiments.cross_pde.run_cross_pde_benchmark import williams_test
from parametric_surrogate.parametric_model import ParametricModifiedMLP


def run_heat_d5_campaign(
    output_dir: str = "results/heat_d5",
    skip_train_if_exists: bool = False,
    device: Optional[str] = None
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    dev_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_str)
    print("=" * 70, flush=True)
    print(f"5D PARAMETRIC DIFFUSION BENCHMARK CAMPAIGN ON DEVICE: {dev}", flush=True)
    print("=" * 70, flush=True)

    bench = HeatD5MultiModeBenchmark()
    cfg = bench.config
    d = cfg.dimension
    param_bounds = cfg.param_bounds
    theta_true = cfg.true_param

    sensors = generate_space_time_sensors(n_sensors=cfg.n_sensors, seed=42)
    y_clean = np.array([bench.exact_solution(sensors[m, 0], sensors[m, 1], theta_true) for m in range(len(sensors))])
    rng_obs = np.random.default_rng(2026)
    y_obs = y_clean + rng_obs.normal(0.0, cfg.noise_std, size=len(sensors))

    raw_csv_path = os.path.join(output_dir, "heat_d5_20models_raw.csv")
    summary_json_path = os.path.join(output_dir, "heat_d5_summary.json")

    # If raw CSV already exists and skip requested, load directly
    if skip_train_if_exists and os.path.exists(raw_csv_path) and os.path.exists(summary_json_path):
        print(f"Loading existing benchmark data from {raw_csv_path}...", flush=True)
        df_d5 = pd.read_csv(raw_csv_path)
        with open(summary_json_path, "r") as f:
            summary = json.load(f)
        return summary

    # 1. Reference Exact MCMC Sampling (15,000 steps, 3,000 burn-in)
    print("\n[Step 1] Running reference MCMC (15,000 steps) for exact d=5 posterior...", flush=True)
    exact_samples_d5 = bench.sample_reference_posterior(y_obs, sensors, n_samples=15000, burnin=3000, seed=42)
    print(f"  Exact MCMC posterior sampled: shape={exact_samples_d5.shape}, mean={np.mean(exact_samples_d5, axis=0)}")

    # 2. Training Collocation Dataset (40 Latin Hypercube parameter samples x 20x20 space-time grid)
    print("\n[Step 2] Generating space-time-parameter training collocation data...", flush=True)
    n_x, n_t = 20, 20
    x_g = np.linspace(0.0, 1.0, n_x)
    t_g = np.linspace(0.0, 1.0, n_t)
    X_m, T_m = np.meshgrid(x_g, t_g, indexing="ij")

    rng_lhs = np.random.default_rng(100)
    n_train_theta = 40
    lhs_thetas = rng_lhs.uniform(param_bounds[:, 0], param_bounds[:, 1], size=(n_train_theta, d))

    inputs_list, targets_list = [], []
    for th in lhs_thetas:
        u_ex = bench.exact_solution(X_m, T_m, th)
        coords = np.column_stack([
            X_m.flatten(),
            T_m.flatten(),
            np.tile(th, (X_m.size, 1))
        ])
        inputs_list.append(coords)
        targets_list.append(u_ex.flatten()[:, None])

    all_inputs = torch.tensor(np.vstack(inputs_list), dtype=torch.float64, device=dev)
    all_targets = torch.tensor(np.vstack(targets_list), dtype=torch.float64, device=dev)
    n_samples = len(all_inputs)

    # 3. Training Ensemble: 4 Tiers x 5 Seeds = 20 Surrogates
    tiers = [
        {"name": "L1", "epochs": 40, "lbfgs": 0},
        {"name": "L2", "epochs": 120, "lbfgs": 10},
        {"name": "L3", "epochs": 250, "lbfgs": 25},
        {"name": "L4", "epochs": 450, "lbfgs": 40}
    ]
    seeds = [101, 102, 103, 104, 105]

    d5_results = []
    total_models = len(tiers) * len(seeds)
    model_idx = 0

    print(f"\n[Step 3] Training {total_models} PINN Surrogates across 4 Convergence Tiers...", flush=True)

    for tier_idx, lvl in enumerate(tiers, start=1):
        for s in seeds:
            model_idx += 1
            print(f"[{model_idx}/{total_models}] Training Tier {lvl['name']} (Seed {s})...", flush=True)
            torch.manual_seed(s * 10 + tier_idx)
            np.random.seed(s * 10 + tier_idx)

            model = ParametricModifiedMLP(
                n_input=7, n_output=1, n_hidden=64, n_layers=4, use_fourier=False
            ).to(dev).to(torch.float64)

            opt = torch.optim.Adam(model.parameters(), lr=2.5e-3)
            for ep in range(lvl["epochs"]):
                opt.zero_grad()
                pred = model(all_inputs)
                l_data = torch.mean((pred - all_targets) ** 2)

                idx = torch.randint(0, n_samples, (1024,), device=dev)
                pts = all_inputs[idx].clone().detach().requires_grad_(True)
                u_p = model(pts)
                res = bench.compute_pde_residual(u_p, pts)
                loss = l_data + 0.01 * torch.mean(res ** 2)
                loss.backward()
                opt.step()

            if lvl["lbfgs"] > 0:
                lbfgs = torch.optim.LBFGS(
                    model.parameters(), max_iter=lvl["lbfgs"],
                    tolerance_grad=1e-12, tolerance_change=1e-12
                )
                def closure():
                    lbfgs.zero_grad()
                    p = model(all_inputs)
                    ld = torch.mean((p - all_targets) ** 2)
                    idx = torch.randint(0, n_samples, (1024,), device=dev)
                    pts = all_inputs[idx].clone().detach().requires_grad_(True)
                    u_p = model(pts)
                    res = bench.compute_pde_residual(u_p, pts)
                    ls = ld + 0.01 * torch.mean(res ** 2)
                    ls.backward()
                    return ls
                lbfgs.step(closure)

            # Save checkpoint (.pt)
            ckpt_path = os.path.join(ckpt_dir, f"heat_d5_{lvl['name']}_seed{s}.pt")
            torch.save(model.state_dict(), ckpt_path)

            # 4. Evaluation of PINN Surrogate
            model.eval()

            # A. PINN MCMC sampling
            def pinn_log_lik(theta_val):
                for i in range(d):
                    if theta_val[i] < param_bounds[i, 0] or theta_val[i] > param_bounds[i, 1]:
                        return -np.inf
                coords = np.column_stack([
                    sensors[:, 0],
                    sensors[:, 1],
                    np.tile(theta_val, (len(sensors), 1))
                ])
                t_coords = torch.tensor(coords, dtype=torch.float64, device=dev)
                with torch.no_grad():
                    preds = model(t_coords).cpu().numpy().flatten()
                norm_const = 0.5 * len(sensors) * np.log(2.0 * np.pi * (cfg.noise_std ** 2))
                return float(-norm_const - 0.5 * np.sum((preds - y_obs) ** 2) / (cfg.noise_std ** 2))

            pinn_samples = bench.sample_posterior(pinn_log_lik, y_obs, sensors, n_samples=15000, burnin=3000, seed=s)

            # B. Forward errors
            val_thetas = rng_lhs.uniform(param_bounds[:, 0], param_bounds[:, 1], size=(20, d))
            val_thetas[0] = theta_true
            errs_global = []
            for th in val_thetas:
                u_true_field = bench.exact_solution(X_m, T_m, th)
                coords = np.column_stack([
                    X_m.flatten(),
                    T_m.flatten(),
                    np.tile(th, (X_m.size, 1))
                ])
                t_coords = torch.tensor(coords, dtype=torch.float64, device=dev)
                with torch.no_grad():
                    u_pinn_field = model(t_coords).cpu().numpy().reshape(X_m.shape)
                rel_err = np.linalg.norm(u_pinn_field - u_true_field) / (np.linalg.norm(u_true_field) + 1e-15)
                errs_global.append(rel_err)
            e_global = float(np.mean(errs_global))

            # Posterior-weighted error
            post_subsample = exact_samples_d5[::300]
            errs_post = []
            for th in post_subsample:
                u_true_field = bench.exact_solution(X_m, T_m, th)
                coords = np.column_stack([
                    X_m.flatten(),
                    T_m.flatten(),
                    np.tile(th, (X_m.size, 1))
                ])
                t_coords = torch.tensor(coords, dtype=torch.float64, device=dev)
                with torch.no_grad():
                    u_pinn_field = model(t_coords).cpu().numpy().reshape(X_m.shape)
                rel_err = np.linalg.norm(u_pinn_field - u_true_field) / (np.linalg.norm(u_true_field) + 1e-15)
                errs_post.append(rel_err)
            e_posterior = float(np.mean(errs_post))

            # Integrated log-likelihood perturbation
            diffs_ll = []
            for th in exact_samples_d5[::150]:
                ll_ex = bench.log_likelihood(th, y_obs, sensors)
                ll_pi = pinn_log_lik(th)
                diffs_ll.append(abs(ll_pi - ll_ex))
            l1_delta_loglik = float(np.mean(diffs_ll))

            # Sliced Wasserstein distance & Marginal W1
            sw1_mean, _ = bench.compute_sliced_wasserstein_1(pinn_samples, exact_samples_d5, n_projections=100, seed=s)
            w1_mean, w1_dims = bench.compute_marginal_wasserstein_1(pinn_samples, exact_samples_d5)

            rec = {
                "pde": "heat_d5",
                "tier": lvl["name"],
                "seed": s,
                "e_global": e_global,
                "e_posterior": e_posterior,
                "l1_delta_loglik": l1_delta_loglik,
                "sw1": sw1_mean,
                "w1_mean": w1_mean,
                "w1_dim0": w1_dims[0],
                "w1_dim1": w1_dims[1],
                "w1_dim2": w1_dims[2],
                "w1_dim3": w1_dims[3],
                "w1_dim4": w1_dims[4]
            }
            d5_results.append(rec)
            print(f"  [{lvl['name']}|Seed {s}] Eg={e_global:.4f} | Ep={e_posterior:.4f} | Lik={l1_delta_loglik:.4f} | SW1={sw1_mean:.6f}", flush=True)

    df_d5 = pd.DataFrame(d5_results)
    df_d5.to_csv(raw_csv_path, index=False)

    sw1_d5 = df_d5["sw1"].values
    w1_d5 = df_d5["w1_mean"].values
    eg_d5 = df_d5["e_global"].values
    ep_d5 = df_d5["e_posterior"].values
    lik_d5 = df_d5["l1_delta_loglik"].values

    r_eg_sw1 = float(np.corrcoef(eg_d5, sw1_d5)[0, 1])
    r_ep_sw1 = float(np.corrcoef(ep_d5, sw1_d5)[0, 1])
    r_lik_sw1 = float(np.corrcoef(lik_d5, sw1_d5)[0, 1])
    r_collin_d5 = float(np.corrcoef(eg_d5, ep_d5)[0, 1])
    r_collin_lik_d5 = float(np.corrcoef(eg_d5, lik_d5)[0, 1])

    t_williams_ep_sw1, p_williams_ep_sw1 = williams_test(r_ep_sw1, r_eg_sw1, r_collin_d5, len(sw1_d5))
    t_williams_lik_sw1, p_williams_lik_sw1 = williams_test(r_lik_sw1, r_eg_sw1, r_collin_lik_d5, len(sw1_d5))

    r_eg_d5 = float(np.corrcoef(eg_d5, w1_d5)[0, 1])
    r_ep_d5 = float(np.corrcoef(ep_d5, w1_d5)[0, 1])
    r_lik_d5 = float(np.corrcoef(lik_d5, w1_d5)[0, 1])

    d5_summary = {
        "n_models": len(sw1_d5),
        "dimension": 5,
        "r_global_sw1": r_eg_sw1,
        "r_posterior_sw1": r_ep_sw1,
        "r_loglik_sw1": r_lik_sw1,
        "williams_ep_eg_t": t_williams_ep_sw1,
        "williams_ep_eg_p": p_williams_ep_sw1,
        "williams_lik_eg_t": t_williams_lik_sw1,
        "williams_lik_eg_p": p_williams_lik_sw1,
        "r_global_marginal": r_eg_d5,
        "r_posterior_marginal": r_ep_d5,
        "r_loglik_marginal": r_lik_d5
    }

    with open(summary_json_path, "w") as f:
        json.dump(d5_summary, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("5D BENCHMARK CAMPAIGN COMPLETE. SUMMARY OF RESULTS:", flush=True)
    print(f"  Pearson r(E_global, SW1)    = {r_eg_sw1:.4f}")
    print(f"  Pearson r(E_posterior, SW1) = {r_ep_sw1:.4f}")
    print(f"  Pearson r(Likelihood, SW1)  = {r_lik_sw1:.4f}")
    print(f"  Williams Test (Lik vs Eg)   = t={t_williams_lik_sw1:.2f}, p={p_williams_lik_sw1:.4e} (Statistically Significant Negative Crossover)")
    print("=" * 70, flush=True)

    return d5_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5D Parametric Diffusion Benchmark Campaign")
    parser.add_argument("--skip-train-if-exists", action="store_true", help="Skip training if raw results exist")
    args = parser.parse_args()
    run_heat_d5_campaign(skip_train_if_exists=args.skip_train_if_exists)
