"""
Unit Tests for 5D Multi-Mode Diffusion Benchmark (d=5)
=======================================================
Verifies:
1. Exact analytical solution satisfies 5D multi-mode diffusion to machine precision
2. Boundary conditions and mode orthogonality/decomposition
3. Observation sensor layout and noise statistics
4. Parametric neural network surrogate forward pass and autograd physics loss
5. 100-direction Sliced Wasserstein-1 distance metric properties
6. Statistically significant dimensional crossover (Williams test p < 0.05)
"""

import os
import json
import pytest
import numpy as np
import torch
import pandas as pd
import scipy.stats as stats

from experiments.cross_pde.heat_d5 import HeatD5MultiModeBenchmark, HeatD5Config
from experiments.cross_pde.pde_definitions import generate_space_time_sensors
from experiments.cross_pde.run_cross_pde_benchmark import williams_test
from parametric_surrogate.parametric_model import ParametricModifiedMLP


class TestHeatD5:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.bench = HeatD5MultiModeBenchmark()
        self.cfg = self.bench.config

    def test_exact_solution_residuals_machine_precision(self):
        """Verify analytical solution satisfies PDE residual to machine precision (< 1e-14)."""
        x = np.linspace(0.1, 0.9, 8)
        t = np.linspace(0.1, 0.9, 8)
        X, T = np.meshgrid(x, t, indexing="ij")

        coords = np.column_stack([
            X.flatten(),
            T.flatten(),
            np.tile(self.cfg.true_param, (X.size, 1))
        ])
        inputs = torch.tensor(coords, dtype=torch.float64, requires_grad=True)

        u_tensor = self.bench.exact_solution_torch(inputs)

        res = self.bench.compute_pde_residual(u_tensor, inputs)
        max_res = float(torch.max(torch.abs(res)))
        assert max_res < 1e-14, f"PDE residual too large: {max_res}"

    def test_spatial_modes_and_boundary_conditions(self):
        """Verify Dirichlet boundary conditions u(0, t) = u(1, t) = 0 for all t."""
        t_vals = np.linspace(0.0, 1.0, 10)
        u_bc0 = self.bench.exact_solution(np.zeros_like(t_vals), t_vals, self.cfg.true_param)
        u_bc1 = self.bench.exact_solution(np.ones_like(t_vals), t_vals, self.cfg.true_param)

        assert np.allclose(u_bc0, 0.0, atol=1e-15)
        assert np.allclose(u_bc1, 0.0, atol=1e-15)

    def test_sensors_generation_bounds(self):
        """Verify sensors are properly placed within valid space-time bounds."""
        sensors = generate_space_time_sensors(n_sensors=40, seed=42)
        assert sensors.shape == (40, 2)
        assert np.all(sensors[:, 0] >= 0.0) and np.all(sensors[:, 0] <= 1.0)
        assert np.all(sensors[:, 1] >= 0.0) and np.all(sensors[:, 1] <= 1.0)

    def test_surrogate_model_autograd_loss(self):
        """Verify 7-input surrogate model backprop and physics loss evaluation."""
        model = ParametricModifiedMLP(
            n_input=7, n_output=1, n_hidden=32, n_layers=3, use_fourier=False
        ).to(torch.float64)

        x = torch.rand(25, 7, dtype=torch.float64, requires_grad=True)
        pred = model(x)
        assert pred.shape == (25, 1)

        res = self.bench.compute_pde_residual(pred, x)
        loss = torch.mean(res ** 2) + torch.mean(pred ** 2)
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None

    def test_sliced_wasserstein_metric_properties(self):
        """Verify Sliced Wasserstein distance identity of indiscernibles and symmetry."""
        rng = np.random.default_rng(42)
        samples_a = rng.normal(0.0, 1.0, size=(1000, 5))
        samples_b = rng.normal(0.5, 1.0, size=(1000, 5))

        # Identity of indiscernibles
        sw1_self, _ = self.bench.compute_sliced_wasserstein_1(samples_a, samples_a, n_projections=50, seed=42)
        assert np.isclose(sw1_self, 0.0, atol=1e-12)

        # Non-negativity and symmetry
        sw1_ab, _ = self.bench.compute_sliced_wasserstein_1(samples_a, samples_b, n_projections=50, seed=42)
        sw1_ba, _ = self.bench.compute_sliced_wasserstein_1(samples_b, samples_a, n_projections=50, seed=42)
        assert sw1_ab > 0.1
        assert np.isclose(sw1_ab, sw1_ba, rtol=1e-2)

    def test_heat_d5_artifacts_and_crossover(self):
        """Verify empirical results confirm statistically significant negative crossover."""
        raw_csv = os.path.join("results", "heat_d5", "heat_d5_20models_raw.csv")
        if not os.path.exists(raw_csv):
            raw_csv = os.path.join("results", "cross_pde_n60", "heat_d5_stress_test_results.csv")

        df = pd.read_csv(raw_csv)
        assert len(df) == 20

        eg = df["e_global"].values
        lik = df["l1_delta_loglik"].values
        sw1 = df["sw1"].values

        r_eg = float(np.corrcoef(eg, sw1)[0, 1])
        r_lik = float(np.corrcoef(lik, sw1)[0, 1])
        r12 = float(np.corrcoef(eg, lik)[0, 1])

        assert r_eg > 0.80, f"r(E_global, SW1) = {r_eg} < 0.80"
        assert r_lik < 0.65, f"r(Likelihood, SW1) = {r_lik} > 0.65"

        t_val, p_val = williams_test(r_lik, r_eg, r12, len(df))
        assert t_val < -2.0, f"Williams t-value = {t_val} not negative enough"
        assert p_val < 0.05, f"Williams p-value = {p_val} not significant"

    def test_sample_posterior_custom_likelihood(self):
        """Verify that sample_posterior evaluates and accepts based on a custom surrogate likelihood."""
        # Define a biased surrogate likelihood centered at a shifted target
        shifted_target = self.cfg.true_param + np.array([0.05, -0.02, 0.02, -0.01, 0.01])
        def mock_surrogate_log_lik(theta_val):
            return float(-0.5 * np.sum(((theta_val - shifted_target) / 0.02) ** 2))

        # Sample with mock surrogate log-likelihood
        surr_samples = self.bench.sample_posterior(
            log_lik_fn=mock_surrogate_log_lik,
            n_samples=1000,
            burnin=200,
            seed=42
        )
        assert surr_samples.shape == (800, 5)
        # Sample mean should be close to shifted_target, not true_param
        mean_surr = np.mean(surr_samples, axis=0)
        assert np.linalg.norm(mean_surr - shifted_target) < np.linalg.norm(mean_surr - self.cfg.true_param)
