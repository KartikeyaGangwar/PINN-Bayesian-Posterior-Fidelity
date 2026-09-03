r"""
Parametric Surrogate Publication Plotting Module (1D Heat Equation)
====================================================================
Generates validation figures:
    - Error vs \alpha curves (Relative L2, Max Error, Observation Error)
    - Field comparison (Exact vs PINN vs Absolute Error)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_surrogate_error_curves(
    alpha_grid: np.ndarray,
    rel_l2_errors: np.ndarray,
    max_errors: np.ndarray,
    obs_errors: Optional[np.ndarray] = None,
    output_dir: str = "results/surrogate_validation",
    true_alpha: Optional[float] = 0.5
) -> List[str]:
    """Generates error curve figures across thermal diffusivity alpha."""
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(alpha_grid, rel_l2_errors * 100, color="#1f77b4", linewidth=2.0, label="Relative $L_2$ Error (%)")
    if obs_errors is not None:
        ax.plot(alpha_grid, obs_errors * 100, color="#2ca02c", linewidth=2.0, linestyle="--", label="Observation Error (%)")
    ax.plot(alpha_grid, max_errors * 100, color="#d62728", linewidth=1.5, linestyle=":", label="Max Absolute Error (x100)")

    if true_alpha is not None:
        ax.axvline(true_alpha, color="black", linestyle="-.", label=rf"True $\alpha^* = {true_alpha:.3f}$")

    ax.set_xlabel(r"Thermal Diffusivity $\alpha$", fontsize=11)
    ax.set_ylabel("Error (%)", fontsize=11)
    ax.set_title(r"Parametric Heat Equation PINN: Forward Error Across $\alpha$", fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    
    p1 = os.path.join(output_dir, "surrogate_error_vs_alpha.png")
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    generated_files.append(p1)

    return generated_files


def plot_solution_field_comparison(
    u_exact: np.ndarray,
    u_pinn: np.ndarray,
    alpha: float,
    output_path: str = "results/surrogate_validation/field_comparison.png"
) -> str:
    """Generates a 3-panel field comparison: Exact vs PINN vs Absolute Error."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    err = np.abs(u_pinn - u_exact)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    
    u_ex_img = np.ascontiguousarray(u_exact.T, dtype=np.float32)
    u_pi_img = np.ascontiguousarray(u_pinn.T, dtype=np.float32)
    err_img = np.ascontiguousarray(err.T, dtype=np.float32)

    im0 = axes[0].imshow(u_ex_img, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", aspect="auto")
    axes[0].set_title(rf"Exact Analytical $u_{{\mathrm{{exact}}}}(x,t;\alpha={alpha:.3f})$")
    axes[0].set_xlabel("Space $x$")
    axes[0].set_ylabel("Time $t$")
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(u_pi_img, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", aspect="auto")
    axes[1].set_title(rf"PINN Surrogate $u_{{\mathrm{{PINN}}}}(x,t;\alpha={alpha:.3f})$")
    axes[1].set_xlabel("Space $x$")
    axes[1].set_ylabel("Time $t$")
    fig.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(err_img, origin="lower", extent=[0, 1, 0, 1], cmap="inferno", aspect="auto")
    axes[2].set_title(rf"Absolute Error $|u_{{\mathrm{{PINN}}}} - u_{{\mathrm{{exact}}}}|$")
    axes[2].set_xlabel("Space $x$")
    axes[2].set_ylabel("Time $t$")
    fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
