"""
2D Incompressible Navier-Stokes Benchmark (Taylor-Green Vortex)
================================================================
Defines the 2D Incompressible Navier-Stokes benchmark on 2D space + 1D time:
    u_t + (u . grad)u = -grad(p) + nu * Laplacian(u)
    div(u) = 0
on (x, y) in [0, 2*pi]^2, t in [0, 1.0].

Possesses an exact closed-form analytical solution (Taylor-Green vortex) with
machine-precision (< 1e-15) Navier-Stokes residual.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional


@dataclass
class NavierStokes2DConfig:
    name: str = "navier_stokes_2d"
    display_name: str = "2D Incompressible Navier-Stokes (Taylor-Green)"
    pde_class: str = "Nonlinear Fluid Dynamics (Incompressible)"
    param_symbol: str = r"\nu"
    param_bounds: Tuple[float, float] = (0.01, 0.20)
    true_param: float = 0.0500
    prior_mu: float = float(np.log(0.05))
    prior_sigma: float = 0.5
    noise_std: float = 0.010
    support_interval: Tuple[float, float] = (0.0485, 0.0515)
    tail_interval: Tuple[float, float] = (0.13, 0.17)
    domain_x: Tuple[float, float] = (0.0, 2.0 * np.pi)
    domain_y: Tuple[float, float] = (0.0, 2.0 * np.pi)
    domain_t: Tuple[float, float] = (0.0, 1.0)


class NavierStokes2DTaylorGreenBenchmark:
    """
    2D Incompressible Navier-Stokes benchmark with exact Taylor-Green vortex solution:
        u(x, y, t; nu) = -cos(x) * sin(y) * exp(-2*nu*t)
        v(x, y, t; nu) =  sin(x) * cos(y) * exp(-2*nu*t)
        p(x, y, t; nu) = -1/4 * (cos(2x) + cos(2y)) * exp(-4*nu*t)
        omega(x, y, t; nu) = 2 * cos(x) * cos(y) * exp(-2*nu*t)
    """

    def __init__(self, config: Optional[NavierStokes2DConfig] = None):
        self.config = config if config is not None else NavierStokes2DConfig()

    def exact_velocity_and_pressure(
        self, x: np.ndarray, y: np.ndarray, t: np.ndarray, nu: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (u, v, p) numpy arrays."""
        decay_vel = np.exp(-2.0 * nu * t)
        decay_pres = np.exp(-4.0 * nu * t)
        u = -np.cos(x) * np.sin(y) * decay_vel
        v = np.sin(x) * np.cos(y) * decay_vel
        p = -0.25 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * decay_pres
        return u, v, p

    def exact_vorticity(
        self, x: np.ndarray, y: np.ndarray, t: np.ndarray, nu: float
    ) -> np.ndarray:
        """Returns vorticity omega = dv/dx - du/dy."""
        return 2.0 * np.cos(x) * np.cos(y) * np.exp(-2.0 * nu * t)

    def compute_pde_residual(
        self, uvp_pred: torch.Tensor, inputs_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the continuity and momentum residuals via PyTorch autograd.
        inputs_grad: [N, 4] tensor of (x, y, t, nu) with requires_grad=True
        uvp_pred: [N, 3] tensor of (u, v, p) predicted by neural network
        Returns:
            r_cont: [N, 1] continuity residual (u_x + v_y)
            r_u:    [N, 1] x-momentum residual
            r_v:    [N, 1] y-momentum residual
        """
        u = uvp_pred[:, 0:1]
        v = uvp_pred[:, 1:2]
        p = uvp_pred[:, 2:3]
        nu = inputs_grad[:, 3:4]

        # First derivatives
        grad_u = torch.autograd.grad(u, inputs_grad, torch.ones_like(u), create_graph=True, allow_unused=True)[0]
        u_x = grad_u[:, 0:1] if grad_u is not None else torch.zeros_like(u)
        u_y = grad_u[:, 1:2] if grad_u is not None else torch.zeros_like(u)
        u_t = grad_u[:, 2:3] if grad_u is not None else torch.zeros_like(u)

        grad_v = torch.autograd.grad(v, inputs_grad, torch.ones_like(v), create_graph=True, allow_unused=True)[0]
        v_x = grad_v[:, 0:1] if grad_v is not None else torch.zeros_like(v)
        v_y = grad_v[:, 1:2] if grad_v is not None else torch.zeros_like(v)
        v_t = grad_v[:, 2:3] if grad_v is not None else torch.zeros_like(v)

        grad_p = torch.autograd.grad(p, inputs_grad, torch.ones_like(p), create_graph=True, allow_unused=True)[0]
        p_x = grad_p[:, 0:1] if grad_p is not None else torch.zeros_like(p)
        p_y = grad_p[:, 1:2] if grad_p is not None else torch.zeros_like(p)

        # Second derivatives (Laplacians)
        grad_u_x = torch.autograd.grad(u_x, inputs_grad, torch.ones_like(u_x), create_graph=True, allow_unused=True)[0]
        u_xx = grad_u_x[:, 0:1] if grad_u_x is not None else torch.zeros_like(u)
        grad_u_y = torch.autograd.grad(u_y, inputs_grad, torch.ones_like(u_y), create_graph=True, allow_unused=True)[0]
        u_yy = grad_u_y[:, 1:2] if grad_u_y is not None else torch.zeros_like(u)

        grad_v_x = torch.autograd.grad(v_x, inputs_grad, torch.ones_like(v_x), create_graph=True, allow_unused=True)[0]
        v_xx = grad_v_x[:, 0:1] if grad_v_x is not None else torch.zeros_like(v)
        grad_v_y = torch.autograd.grad(v_y, inputs_grad, torch.ones_like(v_y), create_graph=True, allow_unused=True)[0]
        v_yy = grad_v_y[:, 1:2] if grad_v_y is not None else torch.zeros_like(v)

        # Navier-Stokes residuals
        r_cont = u_x + v_y
        r_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        r_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

        return r_cont, r_u, r_v


def generate_space_time_sensors_2d(
    n_sensors: int = 40, seed: int = 42
) -> np.ndarray:
    """
    Generates M space-time sensor coordinates [M, 3] in:
        x in [0.1, 0.9] * 2*pi
        y in [0.1, 0.9] * 2*pi
        t in [0.1, 0.9]
    """
    rng = np.random.default_rng(seed)
    x_s = rng.uniform(0.1, 0.9, n_sensors) * (2.0 * np.pi)
    y_s = rng.uniform(0.1, 0.9, n_sensors) * (2.0 * np.pi)
    t_s = rng.uniform(0.1, 0.9, n_sensors)
    return np.column_stack([x_s, y_s, t_s])
