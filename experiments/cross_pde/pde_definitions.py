"""
Unified PDE Definitions Module for Cross-PDE Bayesian PINN Validation Study
============================================================================
Defines 4 representative PDE classes:
1. 1D Heat Equation (Linear Parabolic Diffusion)
2. 1D Wave Equation (Hyperbolic Dynamics)
3. 1D Advection-Diffusion Equation (Transport + Diffusion)
4. 1D Viscous Burgers Equation (Nonlinear Convection-Diffusion)

All formulations possess exact analytical solutions to machine precision.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Callable, Optional


@dataclass
class PDEBenchmarkConfig:
    name: str
    display_name: str
    pde_class: str
    param_symbol: str
    param_bounds: Tuple[float, float]
    true_param: float
    prior_mu: float
    prior_sigma: float
    noise_std: float
    support_interval: Tuple[float, float]
    tail_interval: Tuple[float, float]


class HeatEquationBenchmark:
    """1D Linear Heat Equation: u_t = alpha * u_xx"""
    def __init__(self):
        self.config = PDEBenchmarkConfig(
            name="heat",
            display_name="1D Heat Equation",
            pde_class="Parabolic Diffusion",
            param_symbol=r"\alpha",
            param_bounds=(0.1, 2.0),
            true_param=0.5000,
            prior_mu=float(np.log(0.5)),
            prior_sigma=0.5,
            noise_std=0.010,
            support_interval=(0.495, 0.515),
            tail_interval=(1.20, 1.40)
        )

    def exact_solution(self, x: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
        return np.exp(-alpha * (np.pi ** 2) * t) * np.sin(np.pi * x)

    def compute_pde_residual(self, u_pred: torch.Tensor, inputs_grad: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(u_pred, inputs_grad, torch.ones_like(u_pred), create_graph=True)[0]
        dudx = grads[:, 0:1]
        dudt = grads[:, 1:2]
        param = inputs_grad[:, 2:3]
        d2udx2 = torch.autograd.grad(dudx, inputs_grad, torch.ones_like(dudx), create_graph=True)[0][:, 0:1]
        return dudt - param * d2udx2


class WaveEquationBenchmark:
    """1D Hyperbolic Wave Equation: u_tt = c^2 * u_xx"""
    def __init__(self):
        self.config = PDEBenchmarkConfig(
            name="wave",
            display_name="1D Wave Equation",
            pde_class="Hyperbolic Wave",
            param_symbol="c",
            param_bounds=(0.5, 2.5),
            true_param=1.0000,
            prior_mu=float(np.log(1.0)),
            prior_sigma=0.5,
            noise_std=0.010,
            support_interval=(0.990, 1.010),
            tail_interval=(2.00, 2.30)
        )

    def exact_solution(self, x: np.ndarray, t: np.ndarray, c: float) -> np.ndarray:
        return np.sin(np.pi * x) * np.cos(c * np.pi * t)

    def compute_pde_residual(self, u_pred: torch.Tensor, inputs_grad: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(u_pred, inputs_grad, torch.ones_like(u_pred), create_graph=True)[0]
        dudx = grads[:, 0:1]
        dudt = grads[:, 1:2]
        param = inputs_grad[:, 2:3]
        d2udx2 = torch.autograd.grad(dudx, inputs_grad, torch.ones_like(dudx), create_graph=True)[0][:, 0:1]
        d2udt2 = torch.autograd.grad(dudt, inputs_grad, torch.ones_like(dudt), create_graph=True)[0][:, 1:2]
        return d2udt2 - (param ** 2) * d2udx2


class AdvectionDiffusionBenchmark:
    """1D Transport-Diffusion Equation: u_t + v * u_x = D * u_xx (fixed D = 0.05)"""
    def __init__(self, D: float = 0.05):
        self.D = D
        self.config = PDEBenchmarkConfig(
            name="advection_diffusion",
            display_name="1D Advection-Diffusion",
            pde_class="Transport + Diffusion",
            param_symbol="v",
            param_bounds=(0.2, 2.0),
            true_param=1.0000,
            prior_mu=float(np.log(1.0)),
            prior_sigma=0.5,
            noise_std=0.010,
            support_interval=(0.985, 1.015),
            tail_interval=(1.60, 1.90)
        )

    def exact_solution(self, x: np.ndarray, t: np.ndarray, v: float) -> np.ndarray:
        return np.exp(-4.0 * (np.pi ** 2) * self.D * t) * np.sin(2.0 * np.pi * (x - v * t))

    def compute_pde_residual(self, u_pred: torch.Tensor, inputs_grad: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(u_pred, inputs_grad, torch.ones_like(u_pred), create_graph=True)[0]
        dudx = grads[:, 0:1]
        dudt = grads[:, 1:2]
        param = inputs_grad[:, 2:3]
        d2udx2 = torch.autograd.grad(dudx, inputs_grad, torch.ones_like(dudx), create_graph=True)[0][:, 0:1]
        return dudt + param * dudx - self.D * d2udx2


class ViscousBurgersBenchmark:
    """1D Viscous Burgers Equation (Cole-Hopf exact): u_t + u * u_x = nu * u_xx"""
    def __init__(self):
        self.config = PDEBenchmarkConfig(
            name="burgers",
            display_name="1D Viscous Burgers",
            pde_class="Nonlinear Convection-Diffusion",
            param_symbol=r"\nu",
            param_bounds=(0.02, 0.20),
            true_param=0.0500,
            prior_mu=float(np.log(0.05)),
            prior_sigma=0.5,
            noise_std=0.005,  # scaled to amplitude max|u| ~ 0.15
            support_interval=(0.0485, 0.0515),
            tail_interval=(0.14, 0.18)
        )

    def exact_solution(self, x: np.ndarray, t: np.ndarray, nu: float) -> np.ndarray:
        phi = 2.0 + np.cos(np.pi * x) * np.exp(-nu * (np.pi ** 2) * t)
        phi_x = -np.pi * np.sin(np.pi * x) * np.exp(-nu * (np.pi ** 2) * t)
        return -2.0 * nu * phi_x / phi

    def compute_pde_residual(self, u_pred: torch.Tensor, inputs_grad: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(u_pred, inputs_grad, torch.ones_like(u_pred), create_graph=True)[0]
        dudx = grads[:, 0:1]
        dudt = grads[:, 1:2]
        param = inputs_grad[:, 2:3]
        d2udx2 = torch.autograd.grad(dudx, inputs_grad, torch.ones_like(dudx), create_graph=True)[0][:, 0:1]
        return dudt + u_pred * dudx - param * d2udx2


def get_pde_benchmark(pde_name: str):
    """Factory function returning the benchmark instance."""
    pde_map = {
        "heat": HeatEquationBenchmark,
        "wave": WaveEquationBenchmark,
        "advection_diffusion": AdvectionDiffusionBenchmark,
        "burgers": ViscousBurgersBenchmark
    }
    if pde_name not in pde_map:
        raise ValueError(f"Unknown PDE benchmark: {pde_name}. Available: {list(pde_map.keys())}")
    return pde_map[pde_name]()


def generate_space_time_sensors(n_sensors: int = 40, seed: int = 42) -> np.ndarray:
    """Generates space-time sensor coordinates [M, 2] in [0.1, 0.9] x [0.1, 0.9]."""
    rng = np.random.default_rng(seed)
    x_s = rng.uniform(0.1, 0.9, n_sensors)
    t_s = rng.uniform(0.1, 0.9, n_sensors)
    return np.column_stack([x_s, t_s])
