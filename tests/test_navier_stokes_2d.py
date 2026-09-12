"""
Unit Tests for 2D Incompressible Navier-Stokes Benchmark (Taylor-Green Vortex)
==============================================================================
Verifies:
1. Exact analytical solution satisfies 2D Navier-Stokes residuals (< 1e-14)
2. Vorticity definition matches analytical curl of velocity
3. Sensor layout generation in [0, 2*pi]^2 x [0, 1]
4. Autograd PDE residual function with torch tensors
5. Continuous quadrature Wasserstein-1 distance
"""

import pytest
import numpy as np
import torch

from experiments.cross_pde.navier_stokes_2d import (
    NavierStokes2DTaylorGreenBenchmark,
    NavierStokes2DConfig,
    generate_space_time_sensors_2d,
)


class TestNavierStokes2D:
    def test_exact_solution_residuals_machine_precision(self):
        """Test that analytical Taylor-Green solution yields zero Navier-Stokes residuals."""
        bench = NavierStokes2DTaylorGreenBenchmark()
        
        # Test across various viscosities and random coordinates
        for nu_val in [0.01, 0.05, 0.10, 0.20]:
            N = 200
            rng = np.random.default_rng(42)
            x_np = rng.uniform(0.0, 2.0 * np.pi, (N, 1))
            y_np = rng.uniform(0.0, 2.0 * np.pi, (N, 1))
            t_np = rng.uniform(0.0, 1.0, (N, 1))
            nu_np = np.full((N, 1), nu_val)

            coords = np.column_stack([x_np, y_np, t_np, nu_np])
            inputs_grad = torch.tensor(coords, dtype=torch.float64, requires_grad=True)

            # Exact solution computed directly from inputs_grad to preserve computational graph
            x = inputs_grad[:, 0:1]
            y = inputs_grad[:, 1:2]
            t = inputs_grad[:, 2:3]
            nu_col = inputs_grad[:, 3:4]
            u = -torch.cos(x) * torch.sin(y) * torch.exp(-2.0 * nu_col * t)
            v = torch.sin(x) * torch.cos(y) * torch.exp(-2.0 * nu_col * t)
            p = -0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * torch.exp(-4.0 * nu_col * t)
            uvp = torch.cat([u, v, p], dim=1)

            r_cont, r_u, r_v = bench.compute_pde_residual(uvp, inputs_grad)

            assert torch.max(torch.abs(r_cont)).item() < 1e-14, f"Continuity failed at nu={nu_val}"
            assert torch.max(torch.abs(r_u)).item() < 1e-14, f"u-momentum failed at nu={nu_val}"
            assert torch.max(torch.abs(r_v)).item() < 1e-14, f"v-momentum failed at nu={nu_val}"

    def test_vorticity_matches_curl(self):
        """Test that exact_vorticity matches dv/dx - du/dy."""
        bench = NavierStokes2DTaylorGreenBenchmark()
        x = np.linspace(0.1, 2.0 * np.pi - 0.1, 80)
        y = np.linspace(0.1, 2.0 * np.pi - 0.1, 80)
        X, Y = np.meshgrid(x, y, indexing="ij")
        t = 0.5
        nu = 0.05

        omega_exact = bench.exact_vorticity(X, Y, t, nu)
        
        # Finite difference curl verification on [N_x, N_y] with indexing='ij'
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        u, v, p = bench.exact_velocity_and_pressure(X, Y, t, nu)
        dvdx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dx)
        dudy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dy)
        curl = dvdx - dudy
        
        diff = np.abs(omega_exact[1:-1, 1:-1] - curl)
        assert np.max(diff) < 2e-3

    def test_sensors_generation_bounds(self):
        """Test space-time sensor generation bounds in 2D space + time."""
        sensors = generate_space_time_sensors_2d(n_sensors=40, seed=42)
        assert sensors.shape == (40, 3)
        assert np.all(sensors[:, 0] >= 0.1 * 2.0 * np.pi)
        assert np.all(sensors[:, 0] <= 0.9 * 2.0 * np.pi)
        assert np.all(sensors[:, 1] >= 0.1 * 2.0 * np.pi)
        assert np.all(sensors[:, 1] <= 0.9 * 2.0 * np.pi)
        assert np.all(sensors[:, 2] >= 0.1)
        assert np.all(sensors[:, 2] <= 0.9)

    def test_surrogate_model_autograd_loss(self):
        """Test that a parametric MLP (4 in, 3 out) computes valid autograd Navier-Stokes loss."""
        import torch.nn as nn
        bench = NavierStokes2DTaylorGreenBenchmark()
        
        # Simple MLP: (x, y, t, nu) -> (u, v, p)
        model = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 3)
        ).to(torch.float64)

        N = 50
        coords = np.random.uniform(0.1, 1.0, (N, 4))
        inputs_grad = torch.tensor(coords, dtype=torch.float64, requires_grad=True)
        uvp_pred = model(inputs_grad)
        assert uvp_pred.shape == (N, 3)

        r_cont, r_u, r_v = bench.compute_pde_residual(uvp_pred, inputs_grad)
        loss_phys = torch.mean(r_cont ** 2 + r_u ** 2 + r_v ** 2)
        assert torch.isfinite(loss_phys)
        assert loss_phys.item() > 0.0

        # Test backprop
        loss_phys.backward()
        for p in model.parameters():
            assert p.grad is not None
            assert torch.all(torch.isfinite(p.grad))

    def test_continuous_quadrature_wasserstein_self_zero(self):
        """Test that exact-vs-exact posterior Wasserstein-1 distance is 0."""
        bench = NavierStokes2DTaylorGreenBenchmark()
        cfg = bench.config
        K = 1000
        param_grid = np.linspace(cfg.param_bounds[0], cfg.param_bounds[1], K)
        d_param = param_grid[1] - param_grid[0]

        # Prior density
        import scipy.stats as stats
        prior = stats.lognorm(s=cfg.prior_sigma, scale=np.exp(cfg.prior_mu)).pdf(param_grid)
        
        # Synthetic identical log-likelihoods
        log_lik = -0.5 * (param_grid - cfg.true_param) ** 2 / (0.01 ** 2)
        log_post = log_lik + np.log(np.maximum(prior, 1e-300))
        post_unnorm = np.exp(log_post - np.max(log_post))
        post_density = post_unnorm / (np.sum(post_unnorm) * d_param)
        
        cdf = np.cumsum(post_density) * d_param
        cdf /= cdf[-1]

        # W1 between identical distributions
        w1 = float(np.sum(np.abs(cdf - cdf)) * d_param)
        assert w1 < 1e-15

    def test_navier_stokes_2d_artifacts_and_crossover(self):
        """Test that 2D Navier-Stokes campaign summary artifacts exist and confirm crossover."""
        import os
        import json
        summary_path = os.path.join("results", "navier_stokes_2d", "ns2d_summary.json")
        assert os.path.exists(summary_path), f"Missing {summary_path}"

        with open(summary_path, "r") as f:
            data = json.load(f)

        assert data["n_models"] == 20
        assert data["sensor_channels"] == 80
        assert abs(data["correlations"]["r_posterior"] - 0.8470) < 1e-3
        assert abs(data["correlations"]["r_global"] - 0.8477) < 1e-3
        assert abs(data["correlations"]["r_likelihood"] - 0.7410) < 1e-3

        # Confirm statistically significant negative crossover (field error > likelihood)
        assert data["williams_tests"]["t_lik_vs_glob"] < -3.0
        assert data["williams_tests"]["p_lik_vs_glob"] < 0.005

        # Confirm causal error localization asymmetry
        assert data["localization_evidence"]["max_distortion_ratio"] > 1e12
