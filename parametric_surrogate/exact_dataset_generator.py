r"""
Exact Analytical Dataset & Derivatives Generator Module (1D Heat Equation)
===========================================================================
Mathematical Formulation:
    PDE:
        u_t = \alpha u_{xx}, \quad x \in [0, 1], \quad t \in [0, 1]
    Exact Analytical Solution:
        u_{\text{exact}}(x, t; \alpha) = \exp(-\alpha \pi^2 t) \sin(\pi x)

    Exact Derivatives:
        \partial_x u = \pi \exp(-\alpha \pi^2 t) \cos(\pi x)
        \partial_{xx} u = -\pi^2 \exp(-\alpha \pi^2 t) \sin(\pi x) = -\pi^2 u
        \partial_t u = -\alpha \pi^2 \exp(-\alpha \pi^2 t) \sin(\pi x) = -\alpha \pi^2 u
"""

import numpy as np
import os
from typing import Dict, Any, Tuple, Optional


def evaluate_exact_pde_field_and_derivatives(
    alpha: float,
    resolution: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates exact solution u(x,t; alpha) and exact physical derivatives
    dudx, dudt, d2udx2 on an (N_x x N_t) spatio-temporal grid.
    """
    x_val = np.linspace(0.0, 1.0, resolution)
    t_val = np.linspace(0.0, 1.0, resolution)
    X_grid, T_grid = np.meshgrid(x_val, t_val, indexing="ij")
    
    decay_factor = np.exp(-alpha * (np.pi ** 2) * T_grid)
    sin_x = np.sin(np.pi * X_grid)
    cos_x = np.cos(np.pi * X_grid)
    
    u_exact = decay_factor * sin_x
    dudx_exact = np.pi * decay_factor * cos_x
    d2udx2_exact = - (np.pi ** 2) * u_exact
    dudt_exact = - alpha * (np.pi ** 2) * u_exact
    
    return X_grid, T_grid, u_exact, dudx_exact, dudt_exact, d2udx2_exact


def generate_exact_parametric_dataset(
    alpha_values: np.ndarray,
    grid_resolution: int = 100,
    output_path: Optional[str] = "parametric_surrogate/parametric_training_dataset.npz"
) -> Dict[str, np.ndarray]:
    """
    Generates full offline dataset of solution fields and derivatives for all alpha values.
    """
    n_params = len(alpha_values)
    x_val = np.linspace(0.0, 1.0, grid_resolution)
    t_val = np.linspace(0.0, 1.0, grid_resolution)
    X_grid, T_grid = np.meshgrid(x_val, t_val, indexing="ij")
    
    u_all = np.zeros((n_params, grid_resolution, grid_resolution), dtype=np.float64)
    dudx_all = np.zeros((n_params, grid_resolution, grid_resolution), dtype=np.float64)
    dudt_all = np.zeros((n_params, grid_resolution, grid_resolution), dtype=np.float64)
    d2udx2_all = np.zeros((n_params, grid_resolution, grid_resolution), dtype=np.float64)
    
    for i, a_val in enumerate(alpha_values):
        _, _, u_sol, dudx_sol, dudt_sol, d2udx2_sol = evaluate_exact_pde_field_and_derivatives(
            alpha=float(a_val), resolution=grid_resolution
        )
        u_all[i] = u_sol
        dudx_all[i] = dudx_sol
        dudt_all[i] = dudt_sol
        d2udx2_all[i] = d2udx2_sol
        
    dataset_dict = {
        "X_grid": X_grid,
        "T_grid": T_grid,
        "alpha_values": np.array(alpha_values, dtype=np.float64),
        "u_fields": u_all,
        "dudx_fields": dudx_all,
        "dudt_fields": dudt_all,
        "d2udx2_fields": d2udx2_all,
        "resolution": grid_resolution
    }
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez_compressed(output_path, **dataset_dict)
        print(f"Offline dataset exported to {output_path} ({n_params} alpha values)")
        
    return dataset_dict
