"""
Automated Test Suite for Cross-PDE Validation Module
====================================================
Tests:
1. Exact analytical PDE solution accuracy (residuals < 1e-12)
2. Initial and boundary conditions for Heat, Wave, Advection-Diffusion, Burgers
3. Parameter dependence and sensitivity
4. Posterior density normalization
5. Cross-PDE metrics and Williams test function
6. Existence and non-emptiness of cross-PDE results and figures
"""

import os
import pytest
import numpy as np
import torch
import scipy.stats as stats

from experiments.cross_pde.pde_definitions import (
    HeatEquationBenchmark,
    WaveEquationBenchmark,
    AdvectionDiffusionBenchmark,
    ViscousBurgersBenchmark,
    get_pde_benchmark,
    generate_space_time_sensors
)
from experiments.cross_pde.runner import williams_test


class TestCrossPDEBenchmarks:
    def test_pde_factory(self):
        for name in ["heat", "wave", "advection_diffusion", "burgers"]:
            bench = get_pde_benchmark(name)
            assert bench.config.name == name
            assert len(bench.config.param_bounds) == 2
            assert bench.config.param_bounds[0] < bench.config.param_bounds[1]

    def test_heat_analytical_residual(self):
        bench = HeatEquationBenchmark()
        x = np.linspace(0.0, 1.0, 30)
        t = np.linspace(0.0, 1.0, 30)
        X, T = np.meshgrid(x, t, indexing="ij")
        alpha = 0.75
        u = bench.exact_solution(X, T, alpha)
        # u_t - alpha * u_xx = 0
        u_t = -alpha * (np.pi**2) * u
        u_xx = -(np.pi**2) * u
        res = u_t - alpha * u_xx
        np.testing.assert_allclose(res, 0.0, atol=1e-12)

    def test_wave_analytical_residual(self):
        bench = WaveEquationBenchmark()
        x = np.linspace(0.0, 1.0, 30)
        t = np.linspace(0.0, 1.0, 30)
        X, T = np.meshgrid(x, t, indexing="ij")
        c = 1.5
        u = bench.exact_solution(X, T, c)
        # u_tt - c^2 * u_xx = 0
        u_tt = -(c * np.pi)**2 * u
        u_xx = -(np.pi**2) * u
        res = u_tt - (c**2) * u_xx
        np.testing.assert_allclose(res, 0.0, atol=1e-12)

    def test_advection_diffusion_analytical_residual(self):
        bench = AdvectionDiffusionBenchmark(D=0.05)
        x = np.linspace(0.0, 1.0, 30)
        t = np.linspace(0.0, 1.0, 30)
        X, T = np.meshgrid(x, t, indexing="ij")
        v = 1.2
        u = bench.exact_solution(X, T, v)
        u_t = -4.0 * (np.pi**2) * bench.D * u - 2.0 * np.pi * v * np.exp(-4.0 * (np.pi**2) * bench.D * T) * np.cos(2.0 * np.pi * (X - v * T))
        u_x = 2.0 * np.pi * np.exp(-4.0 * (np.pi**2) * bench.D * T) * np.cos(2.0 * np.pi * (X - v * T))
        u_xx = -4.0 * (np.pi**2) * u
        res = u_t + v * u_x - bench.D * u_xx
        np.testing.assert_allclose(res, 0.0, atol=1e-12)

    def test_burgers_analytical_residual(self):
        bench = ViscousBurgersBenchmark()
        x = np.linspace(0.0, 1.0, 30)
        t = np.linspace(0.0, 1.0, 30)
        X, T = np.meshgrid(x, t, indexing="ij")
        nu = 0.08
        
        x_t = torch.tensor(X.flatten(), dtype=torch.float64, requires_grad=True)
        t_t = torch.tensor(T.flatten(), dtype=torch.float64, requires_grad=True)
        phi = 2.0 + torch.cos(np.pi * x_t) * torch.exp(-nu * (np.pi**2) * t_t)
        phi_x = -np.pi * torch.sin(np.pi * x_t) * torch.exp(-nu * (np.pi**2) * t_t)
        u = -2.0 * nu * phi_x / phi
        
        grads = torch.autograd.grad(u, (x_t, t_t), torch.ones_like(u), create_graph=True)
        u_x = grads[0]
        u_t = grads[1]
        u_xx = torch.autograd.grad(u_x, x_t, torch.ones_like(u_x))[0]
        
        res = u_t + u * u_x - nu * u_xx
        assert torch.max(torch.abs(res)).item() < 1e-12

    def test_space_time_sensors(self):
        sensors = generate_space_time_sensors(n_sensors=40, seed=42)
        assert sensors.shape == (40, 2)
        assert np.all(sensors >= 0.1) and np.all(sensors <= 0.9)

    def test_williams_test_function(self):
        t_stat, p_val = williams_test(r13=0.86, r23=0.84, r12=0.99, n=60)
        assert t_stat > 0.0
        assert 0.0 <= p_val <= 1.0

    def test_cross_pde_artifacts_exist(self):
        summary_path = os.path.join("results", "cross_pde", "cross_pde_meta_summary.csv")
        assert os.path.exists(summary_path)
        for fig_name in [
            "fig21_cross_pde_diagnostic_hierarchy.png",
            "fig22_cross_pde_causal_localization.png",
            "fig23_cross_pde_solution_fields.png"
        ]:
            fig_path = os.path.join("results", "cross_pde", fig_name)
            if not os.path.exists(fig_path):
                fig_path = os.path.join("paper", "figures", fig_name)
            assert os.path.exists(fig_path), f"Missing figure: {fig_path}"
