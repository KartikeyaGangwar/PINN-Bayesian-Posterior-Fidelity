r"""
Parametric Surrogate Reproducibility Artifact Export Module (1D Heat Equation)
==============================================================================
Exports neural network weights, JSON summaries, and CSV tables.
"""

import os
import json
import csv
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

from .metrics import SurrogateMetricsContainer


def export_reproducibility_artifacts(
    model: nn.Module,
    metrics: List[SurrogateMetricsContainer],
    config: Optional[Dict[str, Any]] = None,
    output_dir: str = "results/surrogate_validation"
) -> List[str]:
    """Exports reproducibility bundle."""
    os.makedirs(output_dir, exist_ok=True)
    exported_files = []
    
    # 1. Neural Network Weights
    weights_path = os.path.join(output_dir, "parametric_pinn_weights.pth")
    torch.save(model.state_dict(), weights_path)
    exported_files.append(weights_path)
    
    # 2. Fourier Frequencies Matrix B (if present)
    if hasattr(model, "fourier") and hasattr(model.fourier, "B"):
        fourier_path = os.path.join(output_dir, "fourier_matrix_B.pt")
        torch.save(model.fourier.B.detach().cpu(), fourier_path)
        exported_files.append(fourier_path)
        
    # 3. Validation CSV Tables
    csv_grid_path = os.path.join(output_dir, "surrogate_validation_grid.csv")
    with open(csv_grid_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "rel_l2", "rmse", "max_error", "obs_rel_error", "res_mean", "dudx_err", "dudt_err", "d2udx2_err", "latency_ms", "passed"])
        for m in metrics:
            writer.writerow([m.alpha, m.rel_l2, m.rmse, m.max_error, m.obs_rel_error, m.pde_residual_mean, m.dudx_rel_error, m.dudt_rel_error, m.d2udx2_rel_error, m.inference_latency_ms, m.passed_validation])
    exported_files.append(csv_grid_path)
    
    # 4. Reproducibility Manifest (JSON)
    manifest = {
        "benchmark": "1D Parametric Heat Equation",
        "pde": "u_t = alpha * u_xx",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "network_architecture": {
            "n_input": 3,
            "n_output": 1,
            "n_hidden": 64,
            "n_layers": 4,
            "use_fourier": True,
            "activation": "Tanh"
        },
        "configuration": config if config is not None else {},
        "validation_summary": {
            "mean_relative_l2": float(sum(m.rel_l2 for m in metrics) / len(metrics)) if metrics else 0.0,
            "max_relative_l2": float(max(m.rel_l2 for m in metrics)) if metrics else 0.0,
            "mean_obs_error": float(sum(m.obs_rel_error for m in metrics) / len(metrics)) if metrics else 0.0,
            "pass_rate_percent": float(sum(1 for m in metrics if m.passed_validation) / len(metrics)) * 100.0 if metrics else 0.0,
            "n_evaluations": len(metrics)
        }
    }
    
    manifest_path = os.path.join(output_dir, "reproducibility_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    exported_files.append(manifest_path)
    
    return exported_files
