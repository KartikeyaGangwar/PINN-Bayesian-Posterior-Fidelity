"""
Controlled 3-Way Ablation Study: Pure Physics PINN vs. Data-Only vs. Hybrid
=============================================================================
Evaluates on 1D Linear Heat Equation:
  Regime 1: Pure Physics PINN (lambda_data = 0, PDE residual + IC + BC)
  Regime 2: Data-Supervised Neural Network (lambda_phys = 0, 40,000 labels)
  Regime 3: Hybrid Physics-Regularized Surrogate (lambda_data = 1.0, lambda_phys = 0.01)

Also evaluates continuous sample-based E_posterior vs. dense quadrature.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.cross_pde.pde_definitions import (
    get_pde_benchmark, generate_space_time_sensors
)
from parametric_surrogate.parametric_model import ParametricModifiedMLP


class PureParametricHeatTrainer:
    """Trains a Pure Physics PINN on 1D Heat equation without any internal data labels."""
    def __init__(
        self,
        model: nn.Module,
        pde_fn,
        param_bounds=(0.1, 2.0),
        n_pde=4000,
        n_ic=1000,
        n_bc=1000,
        learning_rate=2.5e-3,
        device=None
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).to(torch.float64)
        self.pde_fn = pde_fn
        self.param_bounds = param_bounds
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Pre-generate collocation points
        # Interior points
        x_pde = np.random.uniform(0.0, 1.0, (n_pde, 1))
        t_pde = np.random.uniform(0.0, 1.0, (n_pde, 1))
        a_pde = np.random.uniform(param_bounds[0], param_bounds[1], (n_pde, 1))
        self.pts_pde = torch.tensor(np.hstack([x_pde, t_pde, a_pde]), dtype=torch.float64, device=self.device)
        
        # IC points (t=0)
        x_ic = np.random.uniform(0.0, 1.0, (n_ic, 1))
        t_ic = np.zeros((n_ic, 1))
        a_ic = np.random.uniform(param_bounds[0], param_bounds[1], (n_ic, 1))
        self.pts_ic = torch.tensor(np.hstack([x_ic, t_ic, a_ic]), dtype=torch.float64, device=self.device)
        self.targets_ic = torch.tensor(np.sin(np.pi * x_ic), dtype=torch.float64, device=self.device)
        
        # BC points (x=0 and x=1)
        half = n_bc // 2
        x_bc0 = np.zeros((half, 1))
        x_bc1 = np.ones((n_bc - half, 1))
        x_bc = np.vstack([x_bc0, x_bc1])
        t_bc = np.random.uniform(0.0, 1.0, (n_bc, 1))
        a_bc = np.random.uniform(param_bounds[0], param_bounds[1], (n_bc, 1))
        self.pts_bc = torch.tensor(np.hstack([x_bc, t_bc, a_bc]), dtype=torch.float64, device=self.device)
        self.targets_bc = torch.zeros((n_bc, 1), dtype=torch.float64, device=self.device)

    def compute_losses(self):
        # 1. PDE residual
        pts_grad = self.pts_pde.clone().detach().requires_grad_(True)
        u_pde = self.model(pts_grad)
        res = self.pde_fn(u_pde, pts_grad)
        l_pde = torch.mean(res**2)
        
        # 2. IC loss
        u_ic = self.model(self.pts_ic)
        l_ic = torch.mean((u_ic - self.targets_ic)**2)
        
        # 3. BC loss
        u_bc = self.model(self.pts_bc)
        l_bc = torch.mean((u_bc - self.targets_bc)**2)
        
        return l_pde, l_ic, l_bc

    def train(self, epochs=600, lbfgs_iters=30, lambda_pde=1.0, lambda_ic=10.0, lambda_bc=10.0):
        self.model.train()
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            l_pde, l_ic, l_bc = self.compute_losses()
            loss = lambda_pde * l_pde + lambda_ic * l_ic + lambda_bc * l_bc
            loss.backward()
            self.optimizer.step()

        if lbfgs_iters > 0:
            lbfgs = torch.optim.LBFGS(self.model.parameters(), max_iter=lbfgs_iters, tolerance_grad=1e-12, tolerance_change=1e-12)
            def closure():
                lbfgs.zero_grad()
                lp, lic, lbc = self.compute_losses()
                ls = lambda_pde * lp + lambda_ic * lic + lambda_bc * lbc
                ls.backward()
                return ls
            lbfgs.step(closure)

        self.model.eval()
        return {"training_time_s": time.time() - t0}


class SupervisedParametricHeatTrainer:
    """Trains Data-Supervised or Hybrid Surrogate."""
    def __init__(self, model: nn.Module, pde_fn, inputs, targets, learning_rate=2.5e-3, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).to(torch.float64)
        self.pde_fn = pde_fn
        self.inputs = inputs.to(self.device).to(torch.float64)
        self.targets = targets.to(self.device).to(torch.float64)
        self.n_samples = len(self.inputs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def compute_physics_loss(self, n_points=2048):
        pts = min(n_points, self.n_samples)
        idx = torch.randint(0, self.n_samples, (pts,), device=self.device)
        inputs_sub = self.inputs[idx]
        inputs_grad = inputs_sub.clone().detach().requires_grad_(True)
        u_pred = self.model(inputs_grad)
        res = self.pde_fn(u_pred, inputs_grad)
        return torch.mean(res**2)

    def train(self, epochs=600, lbfgs_iters=30, lambda_data=1.0, lambda_phys=0.0):
        self.model.train()
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            u_pred = self.model(self.inputs)
            l_data = torch.mean((u_pred - self.targets)**2)
            if lambda_phys > 0.0:
                l_phys = self.compute_physics_loss(n_points=2048)
            else:
                l_phys = torch.tensor(0.0, device=self.device, dtype=torch.float64)
            loss = lambda_data * l_data + lambda_phys * l_phys
            loss.backward()
            self.optimizer.step()

        if lbfgs_iters > 0:
            lbfgs = torch.optim.LBFGS(self.model.parameters(), max_iter=lbfgs_iters, tolerance_grad=1e-12, tolerance_change=1e-12)
            def closure():
                lbfgs.zero_grad()
                u_p = self.model(self.inputs)
                ld = torch.mean((u_p - self.targets)**2)
                lp = self.compute_physics_loss(n_points=2048) if lambda_phys > 0.0 else torch.tensor(0.0, device=self.device, dtype=torch.float64)
                ls = lambda_data * ld + lambda_phys * lp
                ls.backward()
                return ls
            lbfgs.step(closure)

        self.model.eval()
        return {"training_time_s": time.time() - t0}


def run_pure_pinn_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing 3-Way Controlled Ablation on Device: {device}", flush=True)

    pde = get_pde_benchmark("heat")
    cfg = pde.config
    output_dir = os.path.join(repo_root, "results", "audit", "final_publication_hardening", "pure_pinn_ablation")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Sensors & Observations
    sensors = generate_space_time_sensors(n_sensors=40, seed=42)
    y_clean = np.array([pde.exact_solution(sensors[m, 0], sensors[m, 1], cfg.true_param) for m in range(len(sensors))])
    rng_obs = np.random.default_rng(100)
    y_obs = y_clean + rng_obs.normal(0.0, cfg.noise_std, size=len(sensors))

    # 2. Quadrature Grid (K=5,000)
    quad_resolution = 5000
    param_grid = np.linspace(cfg.param_bounds[0], cfg.param_bounds[1], quad_resolution)
    d_param = param_grid[1] - param_grid[0]
    prior_dist = stats.lognorm(s=cfg.prior_sigma, scale=np.exp(cfg.prior_mu))
    prior_density = prior_dist.pdf(param_grid)

    exact_sensor_preds = np.zeros((quad_resolution, len(sensors)))
    for m in range(len(sensors)):
        exact_sensor_preds[:, m] = pde.exact_solution(sensors[m, 0], sensors[m, 1], param_grid)

    misfits_exact = 0.5 * np.sum((exact_sensor_preds - y_obs[None, :])**2, axis=1) / (cfg.noise_std ** 2)
    norm_const = 0.5 * len(sensors) * np.log(2.0 * np.pi * (cfg.noise_std ** 2))
    exact_log_lik = -norm_const - misfits_exact
    exact_log_post = exact_log_lik + np.log(np.maximum(prior_density, 1e-300))
    exact_post_unnorm = np.exp(exact_log_post - np.max(exact_log_post))
    exact_post_density = exact_post_unnorm / (np.sum(exact_post_unnorm) * d_param)
    exact_cdf = np.cumsum(exact_post_density) * d_param
    exact_cdf /= exact_cdf[-1]

    # Generate exact MCMC posterior samples for sample-based continuous E_posterior
    # We sample from exact_post_density using inverse CDF
    n_post_samples = 200
    rng_mcmc = np.random.default_rng(42)
    u_rand = rng_mcmc.uniform(0.0, 1.0, n_post_samples)
    post_samples = np.interp(u_rand, exact_cdf, param_grid)
    print(f"Generated S={n_post_samples} exact posterior samples: mean={np.mean(post_samples):.4f}, std={np.std(post_samples):.4f}", flush=True)

    # 3. Supervised Data Setup (25 parameters x 40 x 40 grid = 40,000 pairs)
    n_x, n_t = 40, 40
    x_grid = np.linspace(0.0, 1.0, n_x)
    t_grid = np.linspace(0.0, 1.0, n_t)
    X_mesh, T_mesh = np.meshgrid(x_grid, t_grid, indexing="ij")
    
    train_params = np.linspace(cfg.param_bounds[0], cfg.param_bounds[1], 25)
    inputs_list, targets_list = [], []
    for p_val in train_params:
        u_ex = pde.exact_solution(X_mesh, T_mesh, p_val)
        coords = np.column_stack([X_mesh.flatten(), T_mesh.flatten(), np.full(X_mesh.size, p_val)])
        inputs_list.append(coords)
        targets_list.append(u_ex.flatten()[:, None])
    all_inputs = torch.tensor(np.vstack(inputs_list), dtype=torch.float64)
    all_targets = torch.tensor(np.vstack(targets_list), dtype=torch.float64)

    # Pre-build tensor inputs for sensors
    sensor_eval_inputs = []
    for p_val in param_grid:
        sensor_eval_inputs.append(np.column_stack([sensors[:, 0], sensors[:, 1], np.full(len(sensors), p_val)]))
    tensor_sensor_inputs = torch.tensor(np.vstack(sensor_eval_inputs), dtype=torch.float64, device=device)

    # Pre-build evaluation grids
    # Dense field evaluation over 50 parameter values (for E_global and coarse E_posterior)
    n_eval_50 = 50
    eval_params_50 = np.linspace(cfg.param_bounds[0], cfg.param_bounds[1], n_eval_50)
    field_inputs_50 = []
    for p_val in eval_params_50:
        field_inputs_50.append(np.column_stack([X_mesh.flatten(), T_mesh.flatten(), np.full(X_mesh.size, p_val)]))
    tensor_field_inputs_50 = torch.tensor(np.vstack(field_inputs_50), dtype=torch.float64, device=device)
    exact_fields_50 = [pde.exact_solution(X_mesh, T_mesh, p) for p in eval_params_50]
    exact_norms_50 = [np.linalg.norm(u) + 1e-15 for u in exact_fields_50]

    # Pre-build tensor inputs for the S=200 posterior samples (for continuous E_posterior^MCMC)
    field_inputs_post = []
    for p_val in post_samples:
        field_inputs_post.append(np.column_stack([X_mesh.flatten(), T_mesh.flatten(), np.full(X_mesh.size, p_val)]))
    tensor_field_inputs_post = torch.tensor(np.vstack(field_inputs_post), dtype=torch.float64, device=device)
    exact_fields_post = [pde.exact_solution(X_mesh, T_mesh, p) for p in post_samples]
    exact_norms_post = [np.linalg.norm(u) + 1e-15 for u in exact_fields_post]

    # Interpolate exact posterior onto 50-node grid
    exact_post_50 = np.interp(eval_params_50, param_grid, exact_post_density)
    d_param_50 = eval_params_50[1] - eval_params_50[0]
    exact_post_50 /= (np.sum(exact_post_50) * d_param_50)

    # 4. Execute 3 Regimes x 5 Seeds = 15 Models
    seeds = [101, 102, 103, 104, 105]
    regimes = [
        {"name": "Pure Physics PINN", "type": "pure_pinn", "epochs": 800, "lbfgs": 40},
        {"name": "Data-Supervised Only", "type": "data_only", "epochs": 400, "lbfgs": 20},
        {"name": "Hybrid Regularized", "type": "hybrid", "epochs": 400, "lbfgs": 20}
    ]

    records = []
    print(f"\nStarting 3-Way Training Campaign (15 Models)...", flush=True)

    for reg in regimes:
        reg_name = reg["name"]
        print(f"\n--- Regime: {reg_name} ---", flush=True)
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)

            model = ParametricModifiedMLP(
                n_input=3, n_output=1, n_hidden=64, n_layers=4, use_fourier=False
            ).to(device).to(torch.float64)

            if reg["type"] == "pure_pinn":
                trainer = PureParametricHeatTrainer(
                    model=model,
                    pde_fn=pde.compute_pde_residual,
                    param_bounds=cfg.param_bounds,
                    n_pde=4000, n_ic=1000, n_bc=1000,
                    learning_rate=2.5e-3, device=device
                )
                t_res = trainer.train(epochs=reg["epochs"], lbfgs_iters=reg["lbfgs"], lambda_pde=1.0, lambda_ic=10.0, lambda_bc=10.0)
            elif reg["type"] == "data_only":
                trainer = SupervisedParametricHeatTrainer(
                    model=model, pde_fn=pde.compute_pde_residual,
                    inputs=all_inputs, targets=all_targets,
                    learning_rate=2.5e-3, device=device
                )
                t_res = trainer.train(epochs=reg["epochs"], lbfgs_iters=reg["lbfgs"], lambda_data=1.0, lambda_phys=0.0)
            else: # hybrid
                trainer = SupervisedParametricHeatTrainer(
                    model=model, pde_fn=pde.compute_pde_residual,
                    inputs=all_inputs, targets=all_targets,
                    learning_rate=2.5e-3, device=device
                )
                t_res = trainer.train(epochs=reg["epochs"], lbfgs_iters=reg["lbfgs"], lambda_data=1.0, lambda_phys=0.01)

            # Evaluate surrogate
            with torch.no_grad():
                y_pinn_all = model(tensor_sensor_inputs).cpu().numpy().reshape(quad_resolution, len(sensors))
                u_pred_50 = model(tensor_field_inputs_50).cpu().numpy().reshape(n_eval_50, n_x, n_t)
                u_pred_post = model(tensor_field_inputs_post).cpu().numpy().reshape(n_post_samples, n_x, n_t)

            # Compute field errors
            rel_errs_50 = [np.linalg.norm(u_pred_50[i] - exact_fields_50[i]) / exact_norms_50[i] for i in range(n_eval_50)]
            e_global = float(np.mean(rel_errs_50))
            e_post_coarse = float(np.sum(np.array(rel_errs_50) * exact_post_50) * d_param_50)

            rel_errs_post = [np.linalg.norm(u_pred_post[i] - exact_fields_post[i]) / exact_norms_post[i] for i in range(n_post_samples)]
            e_post_mcmc = float(np.mean(rel_errs_post))

            # Likelihood and posterior
            misfits_pinn = 0.5 * np.sum((y_pinn_all - y_obs[None, :])**2, axis=1) / (cfg.noise_std ** 2)
            pinn_log_lik = -norm_const - misfits_pinn
            pinn_log_post = pinn_log_lik + np.log(np.maximum(prior_density, 1e-300))
            pinn_post_unnorm = np.exp(pinn_log_post - np.max(pinn_log_post))
            pinn_post_density = pinn_post_unnorm / (np.sum(pinn_post_unnorm) * d_param)
            pinn_cdf = np.cumsum(pinn_post_density) * d_param
            pinn_cdf /= pinn_cdf[-1]

            # Metrics
            l1_delta_loglik = float(np.sum(np.abs(pinn_log_lik - exact_log_lik) * exact_post_density) * d_param)
            w1 = float(np.sum(np.abs(pinn_cdf - exact_cdf)) * d_param)
            tv = 0.5 * float(np.sum(np.abs(pinn_post_density - exact_post_density)) * d_param)

            records.append({
                "regime": reg_name,
                "seed": s,
                "e_global": e_global,
                "e_post_coarse": e_post_coarse,
                "e_post_mcmc": e_post_mcmc,
                "l1_delta_loglik": l1_delta_loglik,
                "w1": w1,
                "tv": tv,
                "training_time_s": t_res["training_time_s"]
            })
            print(f"  Seed={s}: Eg={e_global:.4f}, Ep_coarse={e_post_coarse:.4f}, Ep_mcmc={e_post_mcmc:.4f}, Lik={l1_delta_loglik:.4f}, W1={w1:.6f}, time={t_res['training_time_s']:.2f}s", flush=True)

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "pure_pinn_ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved ablation results to {csv_path}", flush=True)

    # Summary table
    agg = df.groupby("regime").agg({
        "e_global": ["mean", "std"],
        "e_post_coarse": ["mean", "std"],
        "e_post_mcmc": ["mean", "std"],
        "l1_delta_loglik": ["mean", "std"],
        "w1": ["mean", "std"],
        "training_time_s": ["mean"]
    })
    print("\n=== ABLATION SUMMARY TABLE ===")
    print(agg)

    # Plotting comparison figure
    plt.rcParams.update({
        "font.family": "serif", "axes.labelsize": 11, "figure.dpi": 300
    })
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    regimes_order = ["Pure Physics PINN", "Data-Supervised Only", "Hybrid Regularized"]
    means_eg = [df[df.regime == r]["e_global"].mean() for r in regimes_order]
    stds_eg = [df[df.regime == r]["e_global"].std() for r in regimes_order]

    means_ep = [df[df.regime == r]["e_post_mcmc"].mean() for r in regimes_order]
    stds_ep = [df[df.regime == r]["e_post_mcmc"].std() for r in regimes_order]

    means_w1 = [df[df.regime == r]["w1"].mean() for r in regimes_order]
    stds_w1 = [df[df.regime == r]["w1"].std() for r in regimes_order]

    x_pos = np.arange(len(regimes_order))
    colors = ["#d62728", "#1f77b4", "#2ca02c"]

    # Panel (a): Forward Errors
    axes[0].bar(x_pos - 0.18, means_eg, width=0.35, yerr=stds_eg, capsize=5, label=r"Global Error $E_{\mathrm{global}}$", color="#4682b4", alpha=0.85)
    axes[0].bar(x_pos + 0.18, means_ep, width=0.35, yerr=stds_ep, capsize=5, label=r"Posterior Error $E_{\mathrm{post}}^{\mathrm{MCMC}}$", color="#2e8b57", alpha=0.85)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(["Pure PINN", "Data-Only", "Hybrid"], fontsize=10)
    axes[0].set_ylabel("Relative $L_2$ Error")
    axes[0].set_title("(a) Forward Surrogate Accuracy", fontweight="bold")
    axes[0].legend(frameon=True, fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Panel (b): Posterior Distortion W1
    axes[1].bar(x_pos, means_w1, width=0.5, yerr=stds_w1, capsize=5, color=colors, alpha=0.85)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(["Pure PINN", "Data-Only", "Hybrid"], fontsize=10)
    axes[1].set_ylabel(r"Wasserstein-1 Discrepancy $\mathcal{W}_1$")
    axes[1].set_title(r"(b) Posterior Fidelity $\mathcal{W}_1(\mu^y, \widehat{\mu}^y)$", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    # Panel (c): Coarse vs Continuous Ep comparison
    for r, col in zip(regimes_order, colors):
        sub = df[df.regime == r]
        axes[2].scatter(sub["e_post_coarse"], sub["e_post_mcmc"], label=r, color=col, s=50, alpha=0.8)
    lims = [0, max(df["e_post_coarse"].max(), df["e_post_mcmc"].max()) * 1.1]
    axes[2].plot(lims, lims, "k--", alpha=0.5, label="Identity line")
    axes[2].set_xlim(lims)
    axes[2].set_ylim(lims)
    axes[2].set_xlabel(r"Coarse 50-node $E_{\mathrm{posterior}}^{\mathrm{coarse}}$")
    axes[2].set_ylabel(r"Continuous $S=200$ $E_{\mathrm{posterior}}^{\mathrm{MCMC}}$")
    axes[2].set_title(r"(c) Coarse vs Continuous $E_{\mathrm{posterior}}$", fontweight="bold")
    axes[2].legend(frameon=True, fontsize=8)
    axes[2].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(repo_root, "paper", "figures", "fig25_pure_pinn_ablation.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ablation figure to {fig_path}", flush=True)


if __name__ == "__main__":
    run_pure_pinn_ablation()
