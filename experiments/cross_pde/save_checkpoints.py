"""
Navier-Stokes 2D Checkpoint Exporter
===================================
Trains and serializes PyTorch model checkpoints (.pt) across all 5 convergence tiers
so that trained weights are permanently preserved on disk for future evaluation.
"""

import os
import time
import torch
import numpy as np
from experiments.cross_pde.navier_stokes_2d import (
    NavierStokes2DTaylorGreenBenchmark,
    generate_space_time_sensors_2d,
)
from parametric_surrogate.parametric_model import ParametricModifiedMLP


def save_tier_checkpoints(output_dir: str = "results/navier_stokes_2d/checkpoints"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65, flush=True)
    print(f"EXPORTING NAVIER-STOKES 2D CHECKPOINTS (.pt) ON: {dev}", flush=True)
    print("=" * 65, flush=True)

    bench = NavierStokes2DTaylorGreenBenchmark()
    cfg = bench.config

    nx, ny, nt = 12, 12, 8
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    t = np.linspace(0, 1.0, nt)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")

    train_params = np.linspace(cfg.param_bounds[0], cfg.param_bounds[1], 15)
    inputs_list, targets_list = [], []
    for nu_val in train_params:
        u_t, v_t, p_t = bench.exact_velocity_and_pressure(X, Y, T, nu_val)
        coords = np.column_stack([X.flatten(), Y.flatten(), T.flatten(), np.full(X.size, nu_val)])
        targets = np.column_stack([u_t.flatten(), v_t.flatten(), p_t.flatten()])
        inputs_list.append(coords)
        targets_list.append(targets)

    all_inputs = torch.tensor(np.vstack(inputs_list), dtype=torch.float64, device=dev)
    all_targets = torch.tensor(np.vstack(targets_list), dtype=torch.float64, device=dev)
    n_samples = len(all_inputs)

    os.makedirs(output_dir, exist_ok=True)

    tiers = [
        {"id": 1, "name": "Tier 1: Underconverged", "epochs": 40, "lbfgs": 0},
        {"id": 2, "name": "Tier 2: Early Converged", "epochs": 100, "lbfgs": 0},
        {"id": 3, "name": "Tier 3: Intermediate", "epochs": 200, "lbfgs": 0},
        {"id": 4, "name": "Tier 4: Near-Optimal", "epochs": 400, "lbfgs": 0},
        {"id": 5, "name": "Tier 5: Converged", "epochs": 600, "lbfgs": 15},
    ]

    for tier in tiers:
        t_id = tier["id"]
        t_name = tier["name"]
        print(f"\n[Training Checkpoint] {t_name} (ID={t_id})...", flush=True)
        t0 = time.time()

        torch.manual_seed(42)
        model = ParametricModifiedMLP(
            n_input=4, n_output=3, n_hidden=64, n_layers=4, use_fourier=False
        ).to(dev).to(torch.float64)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(tier["epochs"]):
            opt.zero_grad()
            pred = model(all_inputs)
            ld = torch.mean((pred - all_targets) ** 2)
            idx = torch.randint(0, n_samples, (1024,), device=dev)
            pts = all_inputs[idx].clone().detach().requires_grad_(True)
            uvp_p = model(pts)
            rc, ru, rv = bench.compute_pde_residual(uvp_p, pts)
            loss = ld + 0.01 * torch.mean(rc ** 2 + ru ** 2 + rv ** 2)
            loss.backward()
            opt.step()

        if tier["lbfgs"] > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(), max_iter=tier["lbfgs"],
                tolerance_grad=1e-12, tolerance_change=1e-12
            )
            def closure():
                lbfgs.zero_grad()
                p = model(all_inputs)
                ld = torch.mean((p - all_targets) ** 2)
                idx = torch.randint(0, n_samples, (1024,), device=dev)
                pts = all_inputs[idx].clone().detach().requires_grad_(True)
                uvp_p = model(pts)
                rc, ru, rv = bench.compute_pde_residual(uvp_p, pts)
                ls = ld + 0.01 * torch.mean(rc ** 2 + ru ** 2 + rv ** 2)
                ls.backward()
                return ls
            lbfgs.step(closure)

        save_path = os.path.join(output_dir, f"ns2d_tier{t_id}_seed42.pt")
        torch.save(model.state_dict(), save_path)
        dt = time.time() - t0
        print(f"-> Saved: {save_path} ({os.path.getsize(save_path)} bytes) in {dt:.1f}s", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("ALL 5 TIER MODEL CHECKPOINTS (.pt) SUCCESSFULLY CREATED & SAVED!", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    save_tier_checkpoints()
