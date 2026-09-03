r"""
Parametric PINN Surrogate Pipeline Runner (1D Heat Equation)
============================================================
Orchestrates:
    1. Compute training bounds for alpha from Bayesian Prior.
    2. Generate Latin Hypercube / Log-Uniform samples of alpha.
    3. Generate exact analytical dataset of solution fields and derivatives.
    4. Train Parametric PINN model in float64 using Adam + L-BFGS.
    5. Validate surrogate model across alpha grid.
    6. Export reproducibility bundle (weights, plots, JSON, CSV).
"""

import time
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

from bayesian.prior import Prior, LogNormalPrior
from .parameter_sampler import (
    sample_parameter_domain_lhs,
    compute_parameter_bounds,
    PriorDomainConfig
)
from .exact_dataset_generator import generate_exact_parametric_dataset, evaluate_exact_pde_field_and_derivatives
from .dataset import ParametricPINNDataset
from .parametric_model import ParametricModifiedMLP
from .trainer import ParametricPINNTrainer
from .validation import validate_surrogate_grid, evaluate_surrogate_and_derivatives
from .plots import plot_surrogate_error_curves, plot_solution_field_comparison
from .exports import export_reproducibility_artifacts


def run_parametric_surrogate_pipeline(
    prior: Optional[Prior] = None,
    domain_config: Optional[PriorDomainConfig] = None,
    n_train_lhs: int = 50,
    epochs: int = 1500,
    lbfgs_iters: int = 300,
    output_dir: str = "results/surrogate_validation"
) -> Tuple[ParametricModifiedMLP, Dict[str, Any]]:
    """
    Executes end-to-end PINN training and validation pipeline for 1D Heat Equation.
    """
    print("\n" + "=" * 75)
    print("LAUNCHING PARAMETRIC PINN SURROGATE PIPELINE (1D HEAT EQUATION)")
    print("=" * 75)
    
    start_time = time.time()
    
    if prior is None:
        prior = Prior(LogNormalPrior(mu=float(np.log(0.5)), sigma=0.5))
        
    if domain_config is None:
        domain_config = PriorDomainConfig(credible_mass=0.997)
        
    # 1. Sample alpha domain
    train_alphas, a_low, a_high, summary = sample_parameter_domain_lhs(
        prior=prior,
        n_samples=n_train_lhs,
        config=domain_config,
        seed=42
    )
    print(f"[OK] Derived alpha bounds: [{a_low:.4f}, {a_high:.4f}] from Prior")
    print(f"[OK] Generated {len(train_alphas)} parameter samples for training")
    
    # 2. Generate exact offline training data
    dataset_dict = generate_exact_parametric_dataset(
        alpha_values=train_alphas,
        grid_resolution=100,
        output_path=f"{output_dir}/parametric_training_dataset.npz"
    )
    dataset = ParametricPINNDataset(dataset_dict)
    
    # 3. Instantiate & Train Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ParametricModifiedMLP(
        n_input=3, n_output=1, n_hidden=64, n_layers=4,
        use_fourier=True, fourier_scale=1.0
    ).to(device).to(torch.float64)
    
    trainer = ParametricPINNTrainer(model=model, dataset=dataset, device=device)
    train_info = trainer.train(epochs=epochs, lbfgs_iters=lbfgs_iters)
    
    # 4. Dense Grid Validation
    val_grid = np.linspace(a_low, a_high, 30)
    metrics, rel_l2_arr, max_err_arr = validate_surrogate_grid(model, val_grid, resolution=100, device=device)
    mean_l2 = float(np.mean(rel_l2_arr))
    print(f"[VALIDATION] Evaluated {len(val_grid)} alpha points | Mean Rel L2 Error: {mean_l2:.2%}")
    
    # 5. Generate Plots
    plot_surrogate_error_curves(val_grid, rel_l2_arr, max_err_arr, output_dir=output_dir, true_alpha=0.5)
    
    # Field comparison at nominal alpha=0.5
    _, _, u_ex, _, _, _ = evaluate_exact_pde_field_and_derivatives(0.5, resolution=100)
    u_p, _, _, _, _, _ = evaluate_surrogate_and_derivatives(model, 0.5, resolution=100, device=device)
    plot_solution_field_comparison(u_ex, u_p, alpha=0.5, output_path=f"{output_dir}/field_comparison_alpha0.5.png")
    
    # 6. Export Artifacts
    export_reproducibility_artifacts(
        model=model,
        metrics=metrics,
        config={"n_train_lhs": n_train_lhs, "epochs": epochs, "lbfgs_iters": lbfgs_iters},
        output_dir=output_dir
    )
    
    total_time = time.time() - start_time
    print(f"\n[SUCCESS] Pipeline Completed in {total_time:.2f}s | Artifacts saved to {output_dir}\n")
    
    return model, {"total_time": total_time, "mean_rel_l2": mean_l2, "metrics": metrics}
