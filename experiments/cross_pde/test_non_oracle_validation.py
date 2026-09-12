"""
Inexpensive Non-Oracle Surrogate Validation Experiment.

Answers the central operational research question:
'Can a limited number of reference-solver evaluations (e.g. B in {10, 20, 50, 100})
identify an unreliable surrogate posterior more effectively than conventional
forward-error validation, without oracle access to the true posterior?'
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from experiments.cross_pde.pde_definitions import get_pde_benchmark, generate_space_time_sensors
from experiments.cross_pde.trainer import GenericParametricPINNTrainer
from parametric_surrogate.parametric_model import ParametricModifiedMLP


def run_non_oracle_validation_study(
    n_seeds=4,
    budgets=(10, 25, 50, 100),
    n_mcmc_pilot=1000,
    reliability_threshold=0.010,
    device=None,
    output_dir=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if output_dir is None:
        output_dir = repo_root / "results" / "sensitivity_analysis" / "non_oracle_validation"
    os.makedirs(output_dir, exist_ok=True)

    print("==================================================================", flush=True)
    print("NON-ORACLE SURROGATE VALIDATION EXPERIMENT (BUDGET-CONSTRAINED)", flush=True)
    print(f"Device: {device} | Budgets: {budgets} | Reliability Thresh: {reliability_threshold}", flush=True)
    print("==================================================================", flush=True)

    pde = get_pde_benchmark("heat")
    param_bounds = pde.config.param_bounds
    sensors = generate_space_time_sensors(40, seed=42)
    noise_std = 0.010
    true_param = 0.50
    y_clean = pde.exact_solution(sensors[:, 0], sensors[:, 1], true_param)
    rng_obs = np.random.default_rng(100)
    y_obs = y_clean + rng_obs.normal(0, noise_std, size=len(sensors))

    # Dense reference quadrature for ground truth W1
    K_quad = 2500
    param_grid = np.linspace(param_bounds[0], param_bounds[1], K_quad)
    d_param = param_grid[1] - param_grid[0]

    # Precompute exact solutions on grid
    exact_sensor_sol = np.array([pde.exact_solution(sensors[:, 0], sensors[:, 1], p) for p in param_grid])
    exact_log_lik = -0.5 * np.sum((exact_sensor_sol - y_obs[None, :])**2, axis=1) / (noise_std**2)
    exact_log_prior = stats.lognorm.logpdf(param_grid, s=0.5, scale=0.5)
    exact_log_post = exact_log_lik + exact_log_prior
    exact_post_density = np.exp(exact_log_post - np.max(exact_log_post))
    exact_post_density /= (np.sum(exact_post_density) * d_param)
    exact_cdf = np.cumsum(exact_post_density) * d_param
    exact_cdf /= exact_cdf[-1]

    # Reference field evaluations on 40x40 space-time mesh for ground truth error
    nx, nt = 40, 40
    x_mesh = np.linspace(0, 1, nx)
    t_mesh = np.linspace(0, 1, nt)
    X_mesh, T_mesh = np.meshgrid(x_mesh, t_mesh)

    # Training data for surrogates
    train_params = np.linspace(param_bounds[0], param_bounds[1], 25)
    all_inputs, all_targets = [], []
    for p_val in train_params:
        u_val = pde.exact_solution(X_mesh, T_mesh, p_val)
        coords = np.column_stack([X_mesh.flatten(), T_mesh.flatten(), np.full(X_mesh.size, p_val)])
        all_inputs.append(coords)
        all_targets.append(u_val.flatten()[:, None])
    all_inputs = torch.tensor(np.vstack(all_inputs), dtype=torch.float64)
    all_targets = torch.tensor(np.vstack(all_targets), dtype=torch.float64)

    # 4 Convergence Tiers x n_seeds = 16 models
    tier_configs = [
        {"name": "L1_Low", "epochs": 30, "lbfgs": 0},
        {"name": "L2_MedLow", "epochs": 100, "lbfgs": 0},
        {"name": "L3_MedHigh", "epochs": 300, "lbfgs": 10},
        {"name": "L4_High", "epochs": 800, "lbfgs": 30},
    ]

    models_data = []
    print(f"\n--- Training {len(tier_configs) * n_seeds} PINN Surrogates ---", flush=True)

    for tier_idx, tier in enumerate(tier_configs):
        for seed in range(101, 101 + n_seeds):
            torch.manual_seed(seed * 10 + tier_idx)
            np.random.seed(seed * 10 + tier_idx)

            model = ParametricModifiedMLP(n_input=3, n_output=1, n_hidden=64, n_layers=4, use_fourier=False).to(device).to(torch.float64)
            trainer = GenericParametricPINNTrainer(
                model=model,
                pde_residual_fn=pde.compute_pde_residual,
                inputs=all_inputs,
                targets=all_targets,
                learning_rate=2.5e-3,
                device=device
            )
            t_train0 = time.time()
            trainer.train(epochs=tier["epochs"], lbfgs_iters=tier["lbfgs"], lambda_data=1.0, lambda_phys=0.01)
            train_time = time.time() - t_train0

            # Compute Ground Truth W1 via dense quadrature
            model.eval()
            with torch.no_grad():
                sensor_inputs = np.zeros((K_quad * len(sensors), 3))
                for i, p in enumerate(param_grid):
                    sensor_inputs[i*len(sensors):(i+1)*len(sensors), 0] = sensors[:, 0]
                    sensor_inputs[i*len(sensors):(i+1)*len(sensors), 1] = sensors[:, 1]
                    sensor_inputs[i*len(sensors):(i+1)*len(sensors), 2] = p
                sensor_tensor = torch.tensor(sensor_inputs, dtype=torch.float64, device=device)
                pinn_sensor_preds = model(sensor_tensor).cpu().numpy().reshape(K_quad, len(sensors))

            pinn_log_lik = -0.5 * np.sum((pinn_sensor_preds - y_obs[None, :])**2, axis=1) / (noise_std**2)
            pinn_log_post = pinn_log_lik + exact_log_prior
            pinn_post = np.exp(pinn_log_post - np.max(pinn_log_post))
            pinn_post /= (np.sum(pinn_post) * d_param)
            pinn_cdf = np.cumsum(pinn_post) * d_param
            pinn_cdf /= pinn_cdf[-1]

            true_w1 = float(np.sum(np.abs(pinn_cdf - exact_cdf)) * d_param)
            is_unreliable = int(true_w1 > reliability_threshold)

            # --- NON-ORACLE SURROGATE PILOT INVERSION ---
            t_pilot0 = time.time()
            pilot_samples = []
            curr_th = 0.5
            pilot_rng = np.random.default_rng(seed + 999)
            proposals = pilot_rng.normal(curr_th, 0.015, size=n_mcmc_pilot)
            for prop in proposals:
                if param_bounds[0] <= prop <= param_bounds[1]:
                    with torch.no_grad():
                        prop_in = torch.tensor(np.column_stack([sensors[:, 0], sensors[:, 1], np.full(len(sensors), prop)]), dtype=torch.float64, device=device)
                        p_pred = model(prop_in).cpu().numpy().flatten()
                    prop_loglik = -0.5 * np.sum((p_pred - y_obs)**2) / (noise_std**2)
                    curr_th = prop
                pilot_samples.append(curr_th)
            pilot_samples = np.array(pilot_samples[200:])  # burn-in
            pilot_time = time.time() - t_pilot0

            models_data.append({
                "tier": tier["name"],
                "seed": seed,
                "model": model,
                "train_time": train_time,
                "true_w1": true_w1,
                "is_unreliable": is_unreliable,
                "pilot_samples": pilot_samples,
                "pilot_time": pilot_time
            })
            print(f"  [{tier['name']}|Seed {seed}] Train: {train_time:.1f}s | True W1: {true_w1:.6f} | Unreliable: {is_unreliable}", flush=True)

    df_models = pd.DataFrame([{k: v for k, v in m.items() if k != "model" and k != "pilot_samples"} for m in models_data])
    n_unreliable = df_models["is_unreliable"].sum()
    n_reliable = len(df_models) - n_unreliable
    print(f"\nModel Pool Summary: Total={len(df_models)}, Reliable={n_reliable}, Unreliable={n_unreliable}", flush=True)

    # --- BUDGET-CONSTRAINED VALIDATION SIMULATION ---
    budget_eval_records = []
    n_mc_reps = 30

    for B in budgets:
        print(f"\n--- Evaluating Validation Budget B = {B} Reference Solves ---", flush=True)

        prior_auc_list, pilot_auc_list, defensive_auc_list = [], [], []
        prior_r_list, pilot_r_list, defensive_r_list = [], [], []
        prior_far_list, pilot_far_list, defensive_far_list = [], [], []

        for rep in range(n_mc_reps):
            rng_rep = np.random.default_rng(B * 1000 + rep)

            prior_scores = []
            pilot_scores = []
            defensive_scores = []
            true_w1s = []
            unreliable_labels = []

            for m in models_data:
                model = m["model"]
                true_w1 = m["true_w1"]
                pilot_pts = m["pilot_samples"]
                true_w1s.append(true_w1)
                unreliable_labels.append(m["is_unreliable"])

                # Strategy 1: Prior-Uniform Validation (B points drawn from prior)
                theta_prior = rng_rep.uniform(param_bounds[0], param_bounds[1], size=B)
                errs_prior = []
                for th in theta_prior:
                    u_ref = pde.exact_solution(X_mesh, T_mesh, th)
                    norm_ref = np.linalg.norm(u_ref) + 1e-15
                    with torch.no_grad():
                        inp = torch.tensor(np.column_stack([X_mesh.flatten(), T_mesh.flatten(), np.full(X_mesh.size, th)]), dtype=torch.float64, device=device)
                        u_p = model(inp).cpu().numpy().reshape(X_mesh.shape)
                    errs_prior.append(np.linalg.norm(u_p - u_ref) / norm_ref)
                score_prior = np.mean(errs_prior)
                prior_scores.append(score_prior)

                # Strategy 2: Pure Pilot Validation (B points sampled from surrogate pilot)
                theta_pilot = rng_rep.choice(pilot_pts, size=B, replace=True)
                errs_pilot = []
                for th in theta_pilot:
                    u_ref = pde.exact_solution(X_mesh, T_mesh, th)
                    norm_ref = np.linalg.norm(u_ref) + 1e-15
                    with torch.no_grad():
                        inp = torch.tensor(np.column_stack([X_mesh.flatten(), T_mesh.flatten(), np.full(X_mesh.size, th)]), dtype=torch.float64, device=device)
                        u_p = model(inp).cpu().numpy().reshape(X_mesh.shape)
                    errs_pilot.append(np.linalg.norm(u_p - u_ref) / norm_ref)
                score_pilot = np.mean(errs_pilot)
                pilot_scores.append(score_pilot)

                # Strategy 3: Defensive Mixture (80% pilot, 20% prior exploration)
                n_pil = int(0.8 * B)
                n_pri = B - n_pil
                th_def_pil = rng_rep.choice(pilot_pts, size=n_pil, replace=True)
                th_def_pri = rng_rep.uniform(param_bounds[0], param_bounds[1], size=n_pri)
                theta_def = np.concatenate([th_def_pil, th_def_pri])
                errs_def = []
                for th in theta_def:
                    u_ref = pde.exact_solution(X_mesh, T_mesh, th)
                    norm_ref = np.linalg.norm(u_ref) + 1e-15
                    with torch.no_grad():
                        inp = torch.tensor(np.column_stack([X_mesh.flatten(), T_mesh.flatten(), np.full(X_mesh.size, th)]), dtype=torch.float64, device=device)
                        u_p = model(inp).cpu().numpy().reshape(X_mesh.shape)
                    errs_def.append(np.linalg.norm(u_p - u_ref) / norm_ref)
                score_def = np.mean(errs_def)
                defensive_scores.append(score_def)

            true_w1s = np.array(true_w1s)
            unreliable_labels = np.array(unreliable_labels)

            # Compute AUC (discrimination of unreliable models)
            if len(np.unique(unreliable_labels)) > 1:
                auc_prior = roc_auc_score(unreliable_labels, prior_scores)
                auc_pilot = roc_auc_score(unreliable_labels, pilot_scores)
                auc_def = roc_auc_score(unreliable_labels, defensive_scores)
                prior_auc_list.append(auc_prior)
                pilot_auc_list.append(auc_pilot)
                defensive_auc_list.append(auc_def)

            # Compute Pearson Correlation with true W1
            prior_r_list.append(stats.pearsonr(prior_scores, true_w1s)[0])
            pilot_r_list.append(stats.pearsonr(pilot_scores, true_w1s)[0])
            defensive_r_list.append(stats.pearsonr(defensive_scores, true_w1s)[0])

            # Compute False Acceptance Rate (FAR) at 90th percentile threshold of reliable models
            rel_idx = np.where(unreliable_labels == 0)[0]
            unrel_idx = np.where(unreliable_labels == 1)[0]
            if len(rel_idx) > 0 and len(unrel_idx) > 0:
                thresh_prior = np.percentile(np.array(prior_scores)[rel_idx], 90)
                thresh_pilot = np.percentile(np.array(pilot_scores)[rel_idx], 90)
                thresh_def = np.percentile(np.array(defensive_scores)[rel_idx], 90)

                far_prior = np.mean(np.array(prior_scores)[unrel_idx] <= thresh_prior)
                far_pilot = np.mean(np.array(pilot_scores)[unrel_idx] <= thresh_pilot)
                far_def = np.mean(np.array(defensive_scores)[unrel_idx] <= thresh_def)

                prior_far_list.append(far_prior)
                pilot_far_list.append(far_pilot)
                defensive_far_list.append(far_def)

        rec = {
            "budget": B,
            "prior_r_mean": float(np.mean(prior_r_list)),
            "prior_r_std": float(np.std(prior_r_list)),
            "pilot_r_mean": float(np.mean(pilot_r_list)),
            "pilot_r_std": float(np.std(pilot_r_list)),
            "defensive_r_mean": float(np.mean(defensive_r_list)),
            "defensive_r_std": float(np.std(defensive_r_list)),
            "prior_auc_mean": float(np.mean(prior_auc_list)),
            "pilot_auc_mean": float(np.mean(pilot_auc_list)),
            "defensive_auc_mean": float(np.mean(defensive_auc_list)),
            "prior_far_mean": float(np.mean(prior_far_list)),
            "pilot_far_mean": float(np.mean(pilot_far_list)),
            "defensive_far_mean": float(np.mean(defensive_far_list)),
        }
        budget_eval_records.append(rec)
        print(f"  Budget B={B:3d} Solves:")
        print(f"    Pearson r with True W1:  Prior={rec['prior_r_mean']:.4f} | Pilot={rec['pilot_r_mean']:.4f} | Defensive={rec['defensive_r_mean']:.4f}")
        print(f"    ROC AUC (Discrimination): Prior={rec['prior_auc_mean']:.4f} | Pilot={rec['pilot_auc_mean']:.4f} | Defensive={rec['defensive_auc_mean']:.4f}")
        print(f"    False Acceptance Rate:    Prior={rec['prior_far_mean']*100:.1f}% | Pilot={rec['pilot_far_mean']*100:.1f}% | Defensive={rec['defensive_far_mean']*100:.1f}%")

    df_budget_summary = pd.DataFrame(budget_eval_records)
    df_budget_summary.to_csv(Path(output_dir) / "non_oracle_budget_validation_results.csv", index=False)
    with open(Path(output_dir) / "non_oracle_budget_validation_results.json", "w") as f:
        json.dump(budget_eval_records, f, indent=2)

    print("\n==================================================================", flush=True)
    print("Non-oracle validation experiment complete.", flush=True)
    print(f"Results saved to: {output_dir}")
    print("==================================================================", flush=True)

    return df_budget_summary


if __name__ == "__main__":
    run_non_oracle_validation_study()
