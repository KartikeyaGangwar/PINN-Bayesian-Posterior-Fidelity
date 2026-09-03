r"""
Surrogate Metrics Quantification Module (1D Heat Equation)
===========================================================
Quantifies quantitative surrogate validation metrics over thermal diffusivity \alpha:
    1. Relative L2 error: ||u_PINN - u_exact||_2 / ||u_exact||_2
    2. Root Mean Square Error (RMSE)
    3. Maximum Absolute Error (L_\infty norm)
    4. Observation-space relative error: ||H(u_PINN) - H(u_exact)||_2 / ||H(u_exact)||_2
    5. Relative First & Second Derivative Errors (dudx, dudt, d2udx2)
"""

import time
import numpy as np
import torch
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional, List


@dataclass
class SurrogateMetricsContainer:
    alpha: float
    rel_l2: float
    rmse: float
    max_error: float
    obs_rel_error: float
    pde_residual_mean: float
    pde_residual_max: float
    dudx_rel_error: float
    dudt_rel_error: float
    d2udx2_rel_error: float
    inference_latency_ms: float
    gpu_vram_mb: float
    passed_validation: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_surrogate_metrics(
    u_pred: np.ndarray,
    u_exact: np.ndarray,
    alpha: float,
    y_pred_obs: Optional[np.ndarray] = None,
    y_exact_obs: Optional[np.ndarray] = None,
    pde_residual: Optional[np.ndarray] = None,
    dudx_pred: Optional[np.ndarray] = None,
    dudx_exact: Optional[np.ndarray] = None,
    dudt_pred: Optional[np.ndarray] = None,
    dudt_exact: Optional[np.ndarray] = None,
    d2udx2_pred: Optional[np.ndarray] = None,
    d2udx2_exact: Optional[np.ndarray] = None,
    latency_ms: float = 0.0,
    rel_l2_threshold: float = 0.02
) -> SurrogateMetricsContainer:
    """
    Computes complete quantitative validation metrics for a single alpha value.
    """
    error_field = u_pred - u_exact
    
    rel_l2 = float(np.linalg.norm(error_field) / (np.linalg.norm(u_exact) + 1e-15))
    rmse = float(np.sqrt(np.mean(error_field ** 2)))
    max_error = float(np.max(np.abs(error_field)))
    
    if y_pred_obs is not None and y_exact_obs is not None:
        obs_rel_err = float(np.linalg.norm(y_pred_obs - y_exact_obs) / (np.linalg.norm(y_exact_obs) + 1e-15))
    else:
        obs_rel_err = rel_l2

    res_mean = float(np.mean(pde_residual)) if pde_residual is not None else 0.0
    res_max = float(np.max(pde_residual)) if pde_residual is not None else 0.0
    
    dudx_err = float(np.linalg.norm(dudx_pred - dudx_exact) / (np.linalg.norm(dudx_exact) + 1e-15)) if (dudx_pred is not None and dudx_exact is not None) else 0.0
    dudt_err = float(np.linalg.norm(dudt_pred - dudt_exact) / (np.linalg.norm(dudt_exact) + 1e-15)) if (dudt_pred is not None and dudt_exact is not None) else 0.0
    d2udx2_err = float(np.linalg.norm(d2udx2_pred - d2udx2_exact) / (np.linalg.norm(d2udx2_exact) + 1e-15)) if (d2udx2_pred is not None and d2udx2_exact is not None) else 0.0
    
    gpu_mb = float(torch.cuda.memory_allocated(0) / 1e6) if torch.cuda.is_available() else 0.0
    passed = bool(rel_l2 < rel_l2_threshold)
    
    return SurrogateMetricsContainer(
        alpha=float(alpha),
        rel_l2=rel_l2,
        rmse=rmse,
        max_error=max_error,
        obs_rel_error=obs_rel_err,
        pde_residual_mean=res_mean,
        pde_residual_max=res_max,
        dudx_rel_error=dudx_err,
        dudt_rel_error=dudt_err,
        d2udx2_rel_error=d2udx2_err,
        inference_latency_ms=latency_ms,
        gpu_vram_mb=gpu_mb,
        passed_validation=passed
    )
