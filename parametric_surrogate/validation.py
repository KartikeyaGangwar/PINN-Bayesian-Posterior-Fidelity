r"""
Parametric Surrogate Validation Module (1D Heat Equation)
==========================================================
Evaluates surrogate model accuracy across thermal diffusivity \alpha:
- Field relative L2 error: ||u_PINN - u_exact||_2 / ||u_exact||_2
- Derivative accuracy: dudx, dudt, d2udx2
- Observation-space error
- PDE residual: |u_t - alpha * u_xx|
"""

import time
import numpy as np
import torch
import torch.nn as nn
import os
from typing import List, Dict, Any, Tuple, Optional

from .exact_dataset_generator import evaluate_exact_pde_field_and_derivatives
from .metrics import SurrogateMetricsContainer, compute_surrogate_metrics


def evaluate_surrogate_and_derivatives(
    model: nn.Module,
    alpha_val: float,
    resolution: int = 100,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Evaluates surrogate prediction u_PINN and automatic differentiation derivatives
    (dudx, dudt, d2udx2) via autograd for a single alpha value.
    """
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    x_val = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64, device=device)
    t_val = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64, device=device)
    X_test, T_test = torch.meshgrid(x_val, t_val, indexing="ij")
    
    X_flat = X_test.reshape(-1, 1)
    T_flat = T_test.reshape(-1, 1)
    alpha_tensor = torch.full_like(X_flat, alpha_val, dtype=torch.float64, device=device)
    
    inputs = torch.cat([X_flat, T_flat, alpha_tensor], dim=1).requires_grad_(True)
    
    t0 = time.time()
    u_pred = model(inputs)
    
    grads = torch.autograd.grad(u_pred, inputs, torch.ones_like(u_pred), create_graph=True)[0]
    dudx = grads[:, 0:1]
    dudt = grads[:, 1:2]
    d2udx2 = torch.autograd.grad(dudx, inputs, torch.ones_like(dudx), create_graph=True)[0][:, 0:1]
    
    f_res = torch.abs(dudt - alpha_val * d2udx2)
    latency_ms = (time.time() - t0) * 1000.0
    
    u_np = u_pred.detach().reshape(resolution, resolution).cpu().numpy()
    dudx_np = dudx.detach().reshape(resolution, resolution).cpu().numpy()
    dudt_np = dudt.detach().reshape(resolution, resolution).cpu().numpy()
    d2udx2_np = d2udx2.detach().reshape(resolution, resolution).cpu().numpy()
    res_np = f_res.detach().reshape(resolution, resolution).cpu().numpy()
    
    return u_np, dudx_np, dudt_np, d2udx2_np, res_np, latency_ms


def validate_surrogate_grid(
    model: nn.Module,
    alpha_grid: np.ndarray,
    resolution: int = 100,
    device: Optional[torch.device] = None
) -> Tuple[List[SurrogateMetricsContainer], np.ndarray, np.ndarray]:
    """Evaluates surrogate model over a 1D grid of alpha values."""
    metrics_list = []
    rel_l2_arr = np.zeros(len(alpha_grid), dtype=np.float64)
    max_err_arr = np.zeros(len(alpha_grid), dtype=np.float64)
    
    for i, a_v in enumerate(alpha_grid):
        _, _, u_ex, dudx_ex, dudt_ex, d2udx2_ex = evaluate_exact_pde_field_and_derivatives(a_v, resolution)
        u_p, dudx_p, dudt_p, d2udx2_p, res_p, lat_ms = evaluate_surrogate_and_derivatives(model, a_v, resolution, device)
        
        metric = compute_surrogate_metrics(
            u_pred=u_p, u_exact=u_ex, alpha=a_v, pde_residual=res_p,
            dudx_pred=dudx_p, dudx_exact=dudx_ex,
            dudt_pred=dudt_p, dudt_exact=dudt_ex,
            d2udx2_pred=d2udx2_p, d2udx2_exact=d2udx2_ex,
            latency_ms=lat_ms
        )
        metrics_list.append(metric)
        rel_l2_arr[i] = metric.rel_l2
        max_err_arr[i] = metric.max_error
        
    return metrics_list, rel_l2_arr, max_err_arr
