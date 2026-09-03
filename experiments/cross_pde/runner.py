"""
High-Performance Vectorized Cross-PDE Experimental Campaign Runner
==================================================================
Executes Tier 2 validation across 4 PDE benchmarks in vectorized GPU batches:
- Heat Equation (Parabolic Diffusion)
- Wave Equation (Hyperbolic Wave)
- Advection-Diffusion Equation (Transport + Diffusion)
- Viscous Burgers Equation (Nonlinear Convection-Diffusion)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
from typing import Optional, Dict, Any, List, Tuple

from .pde_definitions import get_pde_benchmark, generate_space_time_sensors, PDEBenchmarkConfig
from .trainer import GenericParametricPINNTrainer
from parametric_surrogate.parametric_model import ParametricModifiedMLP


def williams_test(r13, r23, r12, n):
    """
    Williams (1959) t-test for comparing two dependent overlapping correlations:
    H0: rho_13 = rho_23 given collinearity rho_12.
    """
    diff = r13 - r23
    det = 1.0 - r13**2 - r23**2 - r12**2 + 2.0 * r13 * r23 * r12
    det = max(det, 1e-15)
    denom = np.sqrt(2.0 * (n - 1.0) / (n - 3.0) * det + ((r13 + r23)**2 / 4.0) * (1.0 - r12)**3)
    denom = max(denom, 1e-15)
    t_val = diff * np.sqrt((n - 1.0) * (1.0 + r12)) / denom
    p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_val), df=n - 3))
    return float(t_val), float(p_val)


def run_single_pde_campaign(
    pde_name: str,
    n_models_per_level: int = 3,
    n_levels: int = 5,
    quad_resolution: int = 5000,
    device: Optional[torch.device] = None,
    output_dir: str = "results/cross_pde"
) -> Dict[str, Any]:
    print(f"\n========================================================", flush=True)
    print(f"RUNNING CROSS-PDE VALIDATION: {pde_name.upper()}", flush=True)
    print(f"========================================================", flush=True)
    
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pde = get_pde_benchmark(pde_name)
    cfg = pde.config
    pde_out_dir = os.path.join(output_dir, pde_name)
    os.makedirs(pde_out_dir, exist_ok=True)
    
    # 1. Setup space-time grids & sensors
    n_x, n_t = 40, 40
    x_grid = np.linspace(0.0, 1.0, n_x)
    t_grid = np.linspace(0.0, 1.0, n_t)
    X_mesh, T_mesh = np.meshgrid(x_grid, t_grid, indexing="ij")
    
    sensors = generate_space_time_sensors(n_sensors=40, seed=42)
    
    # 2. Compute exact ground truth observations at true parameter
    y_clean = np.zeros(len(sensors))
    for m in range(len(sensors)):
        y_clean[m] = pde.exact_solution(sensors[m, 0], sensors[m, 1], cfg.true_param)
        
    rng_obs = np.random.default_rng(100)
    y_obs = y_clean + rng_obs.normal(0.0, cfg.noise_std, size=len(sensors))
    
    # 3. Dense Parameter-Space Quadrature Grid (Vectorized)
    param_grid = np.linspace(cfg.param_bounds[0], cfg.param_bounds[1], quad_resolution)
    d_param = param_grid[1] - param_grid[0]
    
    # Prior density
    prior_dist = stats.lognorm(s=cfg.prior_sigma, scale=np.exp(cfg.prior_mu))
    prior_density = prior_dist.pdf(param_grid)
    
    # Vectorized exact likelihood across all 5,000 parameter points
    # sensors: [M, 2], param_grid: [K]
    exact_sensor_preds = np.zeros((quad_resolution, len(sensors)))
    for m in range(len(sensors)):
        exact_sensor_preds[:, m] = pde.exact_solution(sensors[m, 0], sensors[m, 1], param_grid)
        
    misfits_exact = 0.5 * np.sum((exact_sensor_preds - y_obs[None, :])**2, axis=1) / (cfg.noise_std ** 2)
    norm_const = 0.5 * len(sensors) * np.log(2.0 * np.pi * (cfg.noise_std ** 2))
    exact_log_lik = -norm_const - misfits_exact
    
    # Exact posterior density
    exact_log_post = exact_log_lik + np.log(np.maximum(prior_density, 1e-300))
    exact_post_unnorm = np.exp(exact_log_post - np.max(exact_log_post))
    exact_post_density = exact_post_unnorm / (np.sum(exact_post_unnorm) * d_param)
    exact_cdf = np.cumsum(exact_post_density) * d_param
    exact_cdf /= exact_cdf[-1]
    
    # 4. Generate Training Dataset across parameter domain
    n_train_params = 25
    train_params = np.linspace(cfg.param_bounds[0], cfg.param_bounds[1], n_train_params)
    
    inputs_list, targets_list = [], []
    for p_val in train_params:
        u_exact_train = pde.exact_solution(X_mesh, T_mesh, p_val)
        coords = np.column_stack([X_mesh.flatten(), T_mesh.flatten(), np.full(X_mesh.size, p_val)])
        inputs_list.append(coords)
        targets_list.append(u_exact_train.flatten()[:, None])
        
    all_inputs = torch.tensor(np.vstack(inputs_list), dtype=torch.float64)
    all_targets = torch.tensor(np.vstack(targets_list), dtype=torch.float64)
    
    # Pre-build evaluation batches for high performance
    # A. Sensor coordinates across all 5,000 parameter points [K * M, 3]
    sensor_eval_inputs = []
    for p_val in param_grid:
        sensor_eval_inputs.append(np.column_stack([sensors[:, 0], sensors[:, 1], np.full(len(sensors), p_val)]))
    tensor_sensor_inputs = torch.tensor(np.vstack(sensor_eval_inputs), dtype=torch.float64, device=device)
    
    # B. Mesh coordinates across 50 test parameter points [50 * n_mesh, 3]
    n_eval_field_params = 50
    field_eval_params = np.linspace(cfg.param_bounds[0], cfg.param_bounds[1], n_eval_field_params)
    field_eval_inputs = []
    for p_val in field_eval_params:
        field_eval_inputs.append(np.column_stack([X_mesh.flatten(), T_mesh.flatten(), np.full(X_mesh.size, p_val)]))
    tensor_field_inputs = torch.tensor(np.vstack(field_eval_inputs), dtype=torch.float64, device=device)
    
    exact_field_norms = []
    exact_field_solutions = []
    for p_val in field_eval_params:
        u_ex = pde.exact_solution(X_mesh, T_mesh, p_val)
        exact_field_solutions.append(u_ex)
        exact_field_norms.append(np.linalg.norm(u_ex) + 1e-15)
        
    # Interpolation weights for posterior density on 50 field params
    exact_post_density_50 = np.interp(field_eval_params, param_grid, exact_post_density)
    d_param_50 = field_eval_params[1] - field_eval_params[0]
    exact_post_density_50 /= (np.sum(exact_post_density_50) * d_param_50)
    
    # 5. Training Ensemble (5 levels x 3 seeds = 15 models)
    level_configs = [
        {"name": "L1", "epochs": 20, "lbfgs": 0},
        {"name": "L2", "epochs": 60, "lbfgs": 0},
        {"name": "L3", "epochs": 150, "lbfgs": 0},
        {"name": "L4", "epochs": 350, "lbfgs": 0},
        {"name": "L5", "epochs": 700, "lbfgs": 25}
    ]
    
    models_data = []
    
    for lvl_idx, lvl in enumerate(level_configs):
        for s in range(n_models_per_level):
            seed = 100 * (lvl_idx + 1) + s + 1
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = ParametricModifiedMLP(
                n_input=3, n_output=1, n_hidden=64, n_layers=4, use_fourier=False
            ).to(device).to(torch.float64)
            
            trainer = GenericParametricPINNTrainer(
                model=model,
                pde_residual_fn=pde.compute_pde_residual,
                inputs=all_inputs,
                targets=all_targets,
                learning_rate=2.5e-3,
                device=device
            )
            
            t_res = trainer.train(epochs=lvl["epochs"], lbfgs_iters=lvl["lbfgs"], lambda_data=1.0, lambda_phys=0.01)
            
            # --- Vectorized Fast Evaluation ---
            with torch.no_grad():
                # 1. Evaluate sensor predictions [K, M]
                y_pinn_all = model(tensor_sensor_inputs).cpu().numpy().reshape(quad_resolution, len(sensors))
                # 2. Evaluate field predictions [50, N_mesh]
                u_pinn_field = model(tensor_field_inputs).cpu().numpy().reshape(n_eval_field_params, X_mesh.shape[0], X_mesh.shape[1])
                
            # Compute PINN log-likelihood for all 5,000 parameter points
            misfits_pinn = 0.5 * np.sum((y_pinn_all - y_obs[None, :])**2, axis=1) / (cfg.noise_std ** 2)
            pinn_log_lik = -norm_const - misfits_pinn
            
            # Compute field relative L2 errors on 50 points
            e_param_50 = np.zeros(n_eval_field_params)
            for j in range(n_eval_field_params):
                diff = u_pinn_field[j] - exact_field_solutions[j]
                e_param_50[j] = np.linalg.norm(diff) / exact_field_norms[j]
                
            e_global = float(np.mean(e_param_50))
            e_posterior = float(np.sum(e_param_50 * exact_post_density_50) * d_param_50)
            
            # Integrated log-likelihood perturbation norm on 5,000 quadrature points
            delta_log_lik = np.abs(pinn_log_lik - exact_log_lik)
            l1_delta_loglik = float(np.sum(delta_log_lik * exact_post_density) * d_param)
            
            # Surrogate posterior density & continuous W1
            pinn_log_post = pinn_log_lik + np.log(np.maximum(prior_density, 1e-300))
            pinn_post_unnorm = np.exp(pinn_log_post - np.max(pinn_log_post))
            pinn_post_density = pinn_post_unnorm / (np.sum(pinn_post_unnorm) * d_param)
            pinn_cdf = np.cumsum(pinn_post_density) * d_param
            pinn_cdf /= pinn_cdf[-1]
            
            w1 = float(np.sum(np.abs(pinn_cdf - exact_cdf)) * d_param)
            kl = float(np.sum(np.where(pinn_post_density > 1e-12, pinn_post_density * np.log(np.maximum(pinn_post_density / (exact_post_density + 1e-15), 1e-15)), 0.0)) * d_param)
            tv = float(0.5 * np.sum(np.abs(pinn_post_density - exact_post_density)) * d_param)
            
            models_data.append({
                "pde": pde_name,
                "level": lvl["name"],
                "seed": seed,
                "e_global": e_global,
                "e_posterior": e_posterior,
                "l1_delta_loglik": l1_delta_loglik,
                "w1": w1,
                "kl": kl,
                "tv": tv,
                "training_time_s": t_res["training_time_s"]
            })
            print(f"  [{pde_name.upper()} | {lvl['name']} | Seed {seed}] E_glob={e_global:.4%}, E_post={e_posterior:.4%}, L1_dlogL={l1_delta_loglik:.4f}, W1={w1:.4e}", flush=True)
            
    df_models = pd.DataFrame(models_data)
    df_models.to_csv(os.path.join(pde_out_dir, f"{pde_name}_models_raw.csv"), index=False)
    
    # 6. Statistical Correlations & Williams Tests
    w1_arr = df_models["w1"].values
    eg_arr = df_models["e_global"].values
    ep_arr = df_models["e_posterior"].values
    lik_arr = df_models["l1_delta_loglik"].values
    N = len(w1_arr)
    
    r_eg_w1 = float(np.corrcoef(eg_arr, w1_arr)[0, 1])
    rho_eg_w1 = float(stats.spearmanr(eg_arr, w1_arr).correlation)
    
    r_ep_w1 = float(np.corrcoef(ep_arr, w1_arr)[0, 1])
    rho_ep_w1 = float(stats.spearmanr(ep_arr, w1_arr).correlation)
    
    r_lik_w1 = float(np.corrcoef(lik_arr, w1_arr)[0, 1])
    rho_lik_w1 = float(stats.spearmanr(lik_arr, w1_arr).correlation)
    
    r_eg_ep = float(np.corrcoef(eg_arr, ep_arr)[0, 1])
    r_eg_lik = float(np.corrcoef(eg_arr, lik_arr)[0, 1])
    
    t_ep_eg, p_ep_eg = williams_test(r_ep_w1, r_eg_w1, r_eg_ep, N)
    t_lik_eg, p_lik_eg = williams_test(r_lik_w1, r_eg_w1, r_eg_lik, N)
    
    # 7. Controlled Localization Perturbation Experiment (Delta log L = 5.0)
    delta_log_L_mag = 5.0
    
    # Support perturbation
    pert_supp_lik = exact_log_lik.copy()
    idx_supp = np.where((param_grid >= cfg.support_interval[0]) & (param_grid <= cfg.support_interval[1]))[0]
    pert_supp_lik[idx_supp] += delta_log_L_mag
    post_supp = np.exp(pert_supp_lik + np.log(np.maximum(prior_density, 1e-300)) - np.max(pert_supp_lik + np.log(np.maximum(prior_density, 1e-300))))
    post_supp_dens = post_supp / (np.sum(post_supp) * d_param)
    cdf_supp = np.cumsum(post_supp_dens) * d_param
    cdf_supp /= cdf_supp[-1]
    w1_supp = float(np.sum(np.abs(cdf_supp - exact_cdf)) * d_param)
    kl_supp = float(np.sum(np.where(post_supp_dens > 1e-12, post_supp_dens * np.log(np.maximum(post_supp_dens / (exact_post_density + 1e-15), 1e-15)), 0.0)) * d_param)
    
    # Tail perturbation
    pert_tail_lik = exact_log_lik.copy()
    idx_tail = np.where((param_grid >= cfg.tail_interval[0]) & (param_grid <= cfg.tail_interval[1]))[0]
    pert_tail_lik[idx_tail] += delta_log_L_mag
    post_tail = np.exp(pert_tail_lik + np.log(np.maximum(prior_density, 1e-300)) - np.max(pert_tail_lik + np.log(np.maximum(prior_density, 1e-300))))
    post_tail_dens = post_tail / (np.sum(post_tail) * d_param)
    cdf_tail = np.cumsum(post_tail_dens) * d_param
    cdf_tail /= cdf_tail[-1]
    w1_tail = float(np.sum(np.abs(cdf_tail - exact_cdf)) * d_param)
    kl_tail = float(np.sum(np.where(post_tail_dens > 1e-12, post_tail_dens * np.log(np.maximum(post_tail_dens / (exact_post_density + 1e-15), 1e-15)), 0.0)) * d_param)
    
    distortion_ratio = float(w1_supp / max(w1_tail, 1e-15))
    
    summary_results = {
        "pde": pde_name,
        "pde_class": cfg.pde_class,
        "n_models": N,
        "r_global": r_eg_w1,
        "rho_global": rho_eg_w1,
        "r_posterior": r_ep_w1,
        "rho_posterior": rho_ep_w1,
        "r_loglik": r_lik_w1,
        "rho_loglik": rho_lik_w1,
        "r_collinearity": r_eg_ep,
        "williams_ep_eg_t": t_ep_eg,
        "williams_ep_eg_p": p_ep_eg,
        "williams_lik_eg_t": t_lik_eg,
        "williams_lik_eg_p": p_lik_eg,
        "pert_w1_support": w1_supp,
        "pert_kl_support": kl_supp,
        "pert_w1_tail": w1_tail,
        "pert_kl_tail": kl_tail,
        "distortion_ratio": distortion_ratio
    }
    
    with open(os.path.join(pde_out_dir, f"{pde_name}_summary.json"), "w") as f:
        json.dump(summary_results, f, indent=2)
        
    print(f"\n[{pde_name.upper()} SUMMARY]", flush=True)
    print(f"  r(E_global, W1)    = {r_eg_w1:.4f} (rho = {rho_eg_w1:.4f})", flush=True)
    print(f"  r(E_posterior, W1) = {r_ep_w1:.4f} (rho = {rho_ep_w1:.4f}) | Williams p = {p_ep_eg:.4e}", flush=True)
    print(f"  r(L1_dlogL, W1)    = {r_lik_w1:.4f} (rho = {rho_lik_w1:.4f}) | Williams p = {p_lik_eg:.4e}", flush=True)
    print(f"  Causal Perturbation: W1_supp = {w1_supp:.4e} vs W1_tail = {w1_tail:.4e} (Ratio > {distortion_ratio:.1e})", flush=True)
    
    return summary_results


def run_cross_pde_campaign(output_dir: str = "results/cross_pde"):
    pdes = ["heat", "wave", "advection_diffusion", "burgers"]
    all_summaries = []
    
    for pde_name in pdes:
        res = run_single_pde_campaign(pde_name=pde_name, output_dir=output_dir)
        all_summaries.append(res)
        
    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(os.path.join(output_dir, "cross_pde_meta_summary.csv"), index=False)
    
    with open(os.path.join(output_dir, "cross_pde_meta_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)
        
    print("\n" + "=" * 80, flush=True)
    print("CROSS-PDE META-ANALYSIS SUMMARY COMPLETED", flush=True)
    print("=" * 80, flush=True)
    print(df_summary[["pde", "pde_class", "r_global", "r_posterior", "r_loglik", "williams_ep_eg_p", "distortion_ratio"]].to_string(), flush=True)
