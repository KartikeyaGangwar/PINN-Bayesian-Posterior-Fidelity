"""
2D Incompressible Navier-Stokes (Taylor-Green Vortex) Validation Campaign
=========================================================================
Executes balanced 20-model surrogate ensemble across 5 convergence tiers:
- Tier 1: Underconverged (40 Adam epochs)
- Tier 2: Early Converged (100 Adam epochs)
- Tier 3: Intermediate (200 Adam epochs)
- Tier 4: Near-Optimal (400 Adam epochs)
- Tier 5: Converged (600 Adam epochs + 15 L-BFGS iterations)

Evaluates:
- Global relative forward error E_global
- Posterior-weighted forward error E_posterior
- Integrated log-likelihood perturbation ||Delta log L||_L1
- True continuous Wasserstein-1 posterior distance W_1 on 5,000 quadrature nodes
- Williams dependent t-tests
- Causal error localization sweep (support vs tail)
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.stats as stats
from typing import Dict, Any, List, Tuple, Optional

from experiments.cross_pde.navier_stokes_2d import (
    NavierStokes2DTaylorGreenBenchmark,
    NavierStokes2DConfig,
    generate_space_time_sensors_2d,
)
from parametric_surrogate.parametric_model import ParametricModifiedMLP


def williams_test(r13: float, r23: float, r12: float, n: int) -> Tuple[float, float]:
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


def run_navier_stokes_2d_campaign(
    output_dir: str = "results/navier_stokes_2d",
    quad_resolution: int = 5000,
    device: Optional[str] = None
) -> Dict[str, Any]:
    device_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_str)
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "Host CPU"
    print("=" * 70, flush=True)
    print("RUNNING 2D INCOMPRESSIBLE NAVIER-STOKES (TAYLOR-GREEN) CAMPAIGN", flush=True)
    print(f"COMPUTE DEVICE: {dev.type.upper()} ({gpu_name})", flush=True)
    print("=" * 70, flush=True)

    os.makedirs(output_dir, exist_ok=True)
    bench = NavierStokes2DTaylorGreenBenchmark()
    cfg = bench.config

    # 1. Sensors & Observations
    sensors = generate_space_time_sensors_2d(n_sensors=40, seed=42)
    u_clean, v_clean, _ = bench.exact_velocity_and_pressure(
        sensors[:, 0], sensors[:, 1], sensors[:, 2], cfg.true_param
    )
    rng_obs = np.random.default_rng(100)
    # 40 velocity sensors observing both (u, v) -> 80 measurements
    y_obs_u = u_clean + rng_obs.normal(0.0, cfg.noise_std, size=len(sensors))
    y_obs_v = v_clean + rng_obs.normal(0.0, cfg.noise_std, size=len(sensors))
    y_obs = np.concatenate([y_obs_u, y_obs_v])  # [80]

    # 2. Quadrature Grid (5,000 points)
    param_grid = np.linspace(cfg.param_bounds[0], cfg.param_bounds[1], quad_resolution)
    d_param = param_grid[1] - param_grid[0]

    prior_dist = stats.lognorm(s=cfg.prior_sigma, scale=np.exp(cfg.prior_mu))
    prior_density = prior_dist.pdf(param_grid)

    exact_u_sensors = np.zeros((quad_resolution, len(sensors)))
    exact_v_sensors = np.zeros((quad_resolution, len(sensors)))
    for m in range(len(sensors)):
        decay = np.exp(-2.0 * param_grid * sensors[m, 2])
        exact_u_sensors[:, m] = -np.cos(sensors[m, 0]) * np.sin(sensors[m, 1]) * decay
        exact_v_sensors[:, m] = np.sin(sensors[m, 0]) * np.cos(sensors[m, 1]) * decay

    exact_preds = np.hstack([exact_u_sensors, exact_v_sensors])  # [K, 80]
    misfits_exact = 0.5 * np.sum((exact_preds - y_obs[None, :]) ** 2, axis=1) / (cfg.noise_std ** 2)
    norm_const = 0.5 * len(y_obs) * np.log(2.0 * np.pi * (cfg.noise_std ** 2))
    exact_log_lik = -norm_const - misfits_exact

    exact_log_post = exact_log_lik + np.log(np.maximum(prior_density, 1e-300))
    exact_post_unnorm = np.exp(exact_log_post - np.max(exact_log_post))
    exact_post_density = exact_post_unnorm / (np.sum(exact_post_unnorm) * d_param)
    exact_cdf = np.cumsum(exact_post_density) * d_param
    exact_cdf /= exact_cdf[-1]

    # 3. Training Collocation Dataset
    n_x, n_y, n_t = 12, 12, 8
    x_train = np.linspace(0.0, 2.0 * np.pi, n_x)
    y_train = np.linspace(0.0, 2.0 * np.pi, n_y)
    t_train = np.linspace(0.0, 1.0, n_t)
    X_m, Y_m, T_m = np.meshgrid(x_train, y_train, t_train, indexing="ij")

    train_params = np.linspace(cfg.param_bounds[0], cfg.param_bounds[1], 15)
    inputs_list, targets_list = [], []
    for nu_val in train_params:
        u_t, v_t, p_t = bench.exact_velocity_and_pressure(X_m, Y_m, T_m, nu_val)
        coords = np.column_stack([
            X_m.flatten(), Y_m.flatten(), T_m.flatten(), np.full(X_m.size, nu_val)
        ])
        targets = np.column_stack([u_t.flatten(), v_t.flatten(), p_t.flatten()])
        inputs_list.append(coords)
        targets_list.append(targets)

    all_inputs = torch.tensor(np.vstack(inputs_list), dtype=torch.float64, device=dev)
    all_targets = torch.tensor(np.vstack(targets_list), dtype=torch.float64, device=dev)
    n_samples = len(all_inputs)

    # Pre-build sensor inputs for evaluation across all 5,000 quadrature points [K * M, 4]
    sensor_eval_inputs = []
    for nu_val in param_grid:
        sensor_eval_inputs.append(np.column_stack([
            sensors[:, 0], sensors[:, 1], sensors[:, 2], np.full(len(sensors), nu_val)
        ]))
    tensor_sensor_inputs = torch.tensor(np.vstack(sensor_eval_inputs), dtype=torch.float64, device=dev)

    # Pre-build validation field mesh for forward error on 50 parameter nodes
    n_eval_field_params = 50
    field_eval_params = np.linspace(cfg.param_bounds[0], cfg.param_bounds[1], n_eval_field_params)
    d_param_50 = field_eval_params[1] - field_eval_params[0]
    
    # Interpolation weights for posterior density on 50 field params
    exact_post_density_50 = np.interp(field_eval_params, param_grid, exact_post_density)
    exact_post_density_50 /= (np.sum(exact_post_density_50) * d_param_50)

    n_xv, n_yv, n_tv = 10, 10, 6
    xv = np.linspace(0.0, 2.0 * np.pi, n_xv)
    yv = np.linspace(0.0, 2.0 * np.pi, n_yv)
    tv = np.linspace(0.0, 1.0, n_tv)
    X_val, Y_val, T_val = np.meshgrid(xv, yv, tv, indexing="ij")
    val_points = np.column_stack([X_val.flatten(), Y_val.flatten(), T_val.flatten()])

    val_eval_inputs = []
    exact_field_solutions = []
    exact_field_norms = []
    for nu_val in field_eval_params:
        uv, vv, pv = bench.exact_velocity_and_pressure(X_val, Y_val, T_val, nu_val)
        val_eval_inputs.append(np.column_stack([val_points, np.full(len(val_points), nu_val)]))
        vel_exact = np.column_stack([uv.flatten(), vv.flatten()])
        exact_field_solutions.append(vel_exact)
        exact_field_norms.append(np.linalg.norm(vel_exact) + 1e-12)
        
    tensor_val_inputs = torch.tensor(np.vstack(val_eval_inputs), dtype=torch.float64, device=dev)

    # 4. Training Ensemble across 5 Tiers
    tiers = [
        {"name": "Tier 1: Underconverged", "epochs": 40, "lbfgs": 0},
        {"name": "Tier 2: Early Converged", "epochs": 100, "lbfgs": 0},
        {"name": "Tier 3: Intermediate", "epochs": 200, "lbfgs": 0},
        {"name": "Tier 4: Near-Optimal", "epochs": 400, "lbfgs": 0},
        {"name": "Tier 5: Converged", "epochs": 600, "lbfgs": 15},
    ]
    seeds = [42, 101, 202, 303]

    model_records = []
    model_idx = 0
    total_models = len(tiers) * len(seeds)

    for tier_id, tier in enumerate(tiers, start=1):
        for seed in seeds:
            model_idx += 1
            print(f"[{model_idx}/{total_models}] Training {tier['name']} (Seed {seed})...", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = ParametricModifiedMLP(
                n_input=4, n_output=3, n_hidden=64, n_layers=4, use_fourier=False
            ).to(dev).to(torch.float64)

            optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
            start_train = time.time()

            # Adam Phase
            for epoch in range(tier["epochs"]):
                optimizer.zero_grad()
                uvp_pred = model(all_inputs)
                l_data = torch.mean((uvp_pred - all_targets) ** 2)

                # Physics loss on collocation subsample
                idx = torch.randint(0, n_samples, (1024,), device=dev)
                pts_grad = all_inputs[idx].clone().detach().requires_grad_(True)
                uvp_pde = model(pts_grad)
                r_cont, r_u, r_v = bench.compute_pde_residual(uvp_pde, pts_grad)
                l_phys = torch.mean(r_cont ** 2 + r_u ** 2 + r_v ** 2)

                loss = l_data + 0.01 * l_phys
                loss.backward()
                optimizer.step()

            # L-BFGS Phase if applicable
            if tier["lbfgs"] > 0:
                lbfgs = torch.optim.LBFGS(
                    model.parameters(), max_iter=tier["lbfgs"],
                    tolerance_grad=1e-12, tolerance_change=1e-12
                )
                def closure():
                    lbfgs.zero_grad()
                    u_p = model(all_inputs)
                    ld = torch.mean((u_p - all_targets) ** 2)
                    idx_l = torch.randint(0, n_samples, (1024,), device=dev)
                    pts_g = all_inputs[idx_l].clone().detach().requires_grad_(True)
                    uvp_p = model(pts_g)
                    rc, ru, rv = bench.compute_pde_residual(uvp_p, pts_g)
                    lp = torch.mean(rc ** 2 + ru ** 2 + rv ** 2)
                    ls = ld + 0.01 * lp
                    ls.backward()
                    return ls
                lbfgs.step(closure)

            train_time = time.time() - start_train

            # Save model checkpoint (.pt)
            ckpt_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"ns2d_tier{tier['id']}_seed{seed}.pt")
            torch.save(model.state_dict(), ckpt_path)

            # Evaluation
            model.eval()
            with torch.no_grad():
                # A. Sensor predictions [K, 80]
                pred_sensors = model(tensor_sensor_inputs).cpu().numpy()  # [K*M, 3]
                u_pred_s = pred_sensors[:, 0].reshape(quad_resolution, len(sensors))
                v_pred_s = pred_sensors[:, 1].reshape(quad_resolution, len(sensors))
                surr_preds = np.hstack([u_pred_s, v_pred_s])

                # Surrogate Likelihood & Posterior
                misfits_surr = 0.5 * np.sum((surr_preds - y_obs[None, :]) ** 2, axis=1) / (cfg.noise_std ** 2)
                surr_log_lik = -norm_const - misfits_surr
                surr_log_post = surr_log_lik + np.log(np.maximum(prior_density, 1e-300))
                surr_post_unnorm = np.exp(surr_log_post - np.max(surr_log_post))
                surr_post_density = surr_post_unnorm / (np.sum(surr_post_unnorm) * d_param)
                surr_cdf = np.cumsum(surr_post_density) * d_param
                surr_cdf /= surr_cdf[-1]

                # True Wasserstein-1 Distance
                w1 = float(np.sum(np.abs(surr_cdf - exact_cdf)) * d_param)

                # B. Forward Field Errors on 50 parameter nodes
                val_pred = model(tensor_val_inputs).cpu().numpy()  # [50 * N_val, 3]
                n_val_pts = len(val_points)
                e_param_50 = np.zeros(n_eval_field_params)
                for j in range(n_eval_field_params):
                    pred_vel_j = val_pred[j * n_val_pts : (j + 1) * n_val_pts, :2]
                    diff = pred_vel_j - exact_field_solutions[j]
                    e_param_50[j] = np.linalg.norm(diff) / exact_field_norms[j]

                e_global = float(np.mean(e_param_50))
                e_posterior = float(np.sum(e_param_50 * exact_post_density_50) * d_param_50)

                # C. Likelihood Perturbation
                delta_log_lik = np.abs(surr_log_lik - exact_log_lik)
                l1_delta_loglik = float(np.sum(delta_log_lik * exact_post_density) * d_param)

            model_records.append({
                "model_idx": model_idx,
                "tier": tier["name"],
                "tier_id": tier_id,
                "seed": seed,
                "epochs": tier["epochs"],
                "lbfgs": tier["lbfgs"],
                "train_time_s": train_time,
                "w1": w1,
                "e_global": e_global,
                "e_posterior": e_posterior,
                "l1_delta_loglik": l1_delta_loglik,
            })
            print(f"   W1: {w1:.6f} | E_post: {e_posterior:.6f} | E_glob: {e_global:.6f} | Lik: {l1_delta_loglik:.4f} ({train_time:.1f}s)", flush=True)

    df_models = pd.DataFrame(model_records)
    csv_path = os.path.join(output_dir, "ns2d_20models_raw.csv")
    df_models.to_csv(csv_path, index=False)
    print(f"\nSaved raw records to {csv_path}", flush=True)

    # 5. Statistical Correlations & Williams Tests
    r_post, p_post = stats.pearsonr(df_models["e_posterior"], df_models["w1"])
    r_glob, p_glob = stats.pearsonr(df_models["e_global"], df_models["w1"])
    r_lik, p_lik = stats.pearsonr(df_models["l1_delta_loglik"], df_models["w1"])

    rho_post, _ = stats.spearmanr(df_models["e_posterior"], df_models["w1"])
    rho_glob, _ = stats.spearmanr(df_models["e_global"], df_models["w1"])
    rho_lik, _ = stats.spearmanr(df_models["l1_delta_loglik"], df_models["w1"])

    # Williams Tests
    r_post_glob, _ = stats.pearsonr(df_models["e_posterior"], df_models["e_global"])
    r_lik_glob, _ = stats.pearsonr(df_models["l1_delta_loglik"], df_models["e_global"])

    t_williams_post, p_williams_post = williams_test(r_post, r_glob, r_post_glob, len(df_models))
    t_williams_lik, p_williams_lik = williams_test(r_lik, r_glob, r_lik_glob, len(df_models))

    # 6. Multi-Magnitude Causal Error Localization Sweep
    print("\nRunning Causal Error Localization Sweep on 2D Navier-Stokes...", flush=True)
    deltas = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    loc_records = []
    
    # Support interval: [0.0485, 0.0515]
    mask_support = (param_grid >= cfg.support_interval[0]) & (param_grid <= cfg.support_interval[1])
    mass_support = float(np.sum(exact_post_density[mask_support]) * d_param)

    # Tail interval: [0.13, 0.17]
    mask_tail = (param_grid >= cfg.tail_interval[0]) & (param_grid <= cfg.tail_interval[1])
    mass_tail = float(np.sum(exact_post_density[mask_tail]) * d_param)

    for delta in deltas:
        # Support perturbation
        pert_support_ll = exact_log_lik.copy()
        pert_support_ll[mask_support] += delta
        pert_post_s = np.exp(pert_support_ll + np.log(np.maximum(prior_density, 1e-300)) - np.max(pert_support_ll))
        pert_dens_s = pert_post_s / (np.sum(pert_post_s) * d_param)
        cdf_s = np.cumsum(pert_dens_s) * d_param
        cdf_s /= cdf_s[-1]
        w1_support = float(np.sum(np.abs(cdf_s - exact_cdf)) * d_param)
        tv_support = 0.5 * float(np.sum(np.abs(pert_dens_s - exact_post_density)) * d_param)

        # Theoretical TV bound
        a = mass_support
        tv_theory = a * (1.0 - a) * (np.exp(delta) - 1.0) / (1.0 + a * (np.exp(delta) - 1.0)) if delta < 50 else (1.0 - a)

        # Tail perturbation
        pert_tail_ll = exact_log_lik.copy()
        pert_tail_ll[mask_tail] += delta
        pert_post_t = np.exp(pert_tail_ll + np.log(np.maximum(prior_density, 1e-300)) - np.max(pert_tail_ll))
        pert_dens_t = pert_post_t / (np.sum(pert_post_t) * d_param)
        cdf_t = np.cumsum(pert_dens_t) * d_param
        cdf_t /= cdf_t[-1]
        w1_tail = float(np.sum(np.abs(cdf_t - exact_cdf)) * d_param)
        tv_tail = 0.5 * float(np.sum(np.abs(pert_dens_t - exact_post_density)) * d_param)

        loc_records.append({
            "delta": delta,
            "mass_support": mass_support,
            "mass_tail": mass_tail,
            "w1_support": w1_support,
            "tv_support": tv_support,
            "tv_theory": tv_theory,
            "w1_tail": w1_tail,
            "tv_tail": tv_tail,
            "distortion_ratio": (w1_support / max(w1_tail, 1e-16)) if w1_tail > 0 else float("inf")
        })

    df_loc = pd.DataFrame(loc_records)
    loc_csv_path = os.path.join(output_dir, "ns2d_localization_sweep.csv")
    df_loc.to_csv(loc_csv_path, index=False)

    summary_results = {
        "benchmark": "2D Incompressible Navier-Stokes (Taylor-Green)",
        "domain": "2D Space (x, y in [0, 2*pi]^2) x 1D Time (t in [0, 1])",
        "n_models": len(df_models),
        "n_sensors": len(sensors),
        "sensor_channels": len(y_obs),
        "quad_resolution": quad_resolution,
        "true_viscosity": cfg.true_param,
        "mass_support": mass_support,
        "mass_tail": mass_tail,
        "correlations": {
            "r_posterior": float(r_post),
            "p_posterior": float(p_post),
            "rho_posterior": float(rho_post),
            "r_global": float(r_glob),
            "p_global": float(p_glob),
            "rho_global": float(rho_glob),
            "r_likelihood": float(r_lik),
            "p_likelihood": float(p_lik),
            "rho_likelihood": float(rho_lik),
        },
        "williams_tests": {
            "t_post_vs_glob": float(t_williams_post),
            "p_post_vs_glob": float(p_williams_post),
            "t_lik_vs_glob": float(t_williams_lik),
            "p_lik_vs_glob": float(p_williams_lik),
        },
        "localization_evidence": {
            "delta_max": deltas[-1],
            "w1_support_max": float(df_loc["w1_support"].iloc[-1]),
            "w1_tail_max": float(df_loc["w1_tail"].iloc[-1]),
            "max_distortion_ratio": float(df_loc["distortion_ratio"].iloc[-1]),
        }
    }

    summary_json_path = os.path.join(output_dir, "ns2d_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("Validation campaign complete. Summary of results:", flush=True)
    print(f"r(E_posterior, W1)  = {r_post:.4f} (rho = {rho_post:.4f})", flush=True)
    print(f"r(E_global, W1)     = {r_glob:.4f} (rho = {rho_glob:.4f})", flush=True)
    print(f"r(Likelihood, W1)   = {r_lik:.4f} (rho = {rho_lik:.4f})", flush=True)
    print(f"Williams E_post vs E_glob: t = {t_williams_post:+.4f}, p = {p_williams_post:.4e}", flush=True)
    print(f"Williams Lik vs E_glob:    t = {t_williams_lik:+.4f}, p = {p_williams_lik:.4e}", flush=True)
    print(f"Causal Support Mass a = {mass_support:.4f}, Tail Mass = {mass_tail:.4e}", flush=True)
    print(f"Support W1 (delta=20) = {df_loc['w1_support'].iloc[-1]:.4e} vs Tail W1 = {df_loc['w1_tail'].iloc[-1]:.4e}", flush=True)
    print("=" * 70, flush=True)

    return summary_results


if __name__ == "__main__":
    run_navier_stokes_2d_campaign()
