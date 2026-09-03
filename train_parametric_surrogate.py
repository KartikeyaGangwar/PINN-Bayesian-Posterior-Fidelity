r"""
Train Parametric PINN Surrogate for 1D Heat Equation
====================================================
Governing PDE:
    u_t = \alpha u_{xx}, \quad x \in [0, 1], \quad t \in [0, 1]
Boundary Conditions:
    u(0, t) = 0, \quad u(1, t) = 0
Initial Condition:
    u(x, 0) = \sin(\pi x)

Exact Analytical Solution:
    u_{\text{exact}}(x, t; \alpha) = \exp(-\alpha \pi^2 t) \sin(\pi x)
"""

import os
import time
import torch
import numpy as np

from bayesian.prior import Prior, LogNormalPrior
from parametric_surrogate.parameter_sampler import sample_parameter_domain_lhs, PriorDomainConfig
from parametric_surrogate.exact_dataset_generator import generate_exact_parametric_dataset, evaluate_exact_pde_field_and_derivatives
from parametric_surrogate.dataset import ParametricPINNDataset
from parametric_surrogate.parametric_model import ParametricModifiedMLP
from parametric_surrogate.trainer import ParametricPINNTrainer
from parametric_surrogate.validation import validate_surrogate_grid, evaluate_surrogate_and_derivatives
from parametric_surrogate.plots import plot_surrogate_error_curves, plot_solution_field_comparison
from parametric_surrogate.exports import export_reproducibility_artifacts


def main():
    print("=" * 80)
    print("TRAINING PARAMETRIC PINN SURROGATE (1D HEAT EQUATION BENCHMARK)")
    print("=" * 80)
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Prior-derived domain sampling
    prior = Prior(LogNormalPrior(mu=float(np.log(0.5)), sigma=0.5))
    config = PriorDomainConfig(credible_mass=0.997)
    
    train_alphas, a_low, a_high, summary = sample_parameter_domain_lhs(
        prior=prior,
        n_samples=30,
        config=config,
        seed=42
    )
    print(f"[OK] Derived alpha training domain: [{a_low:.4f}, {a_high:.4f}]")
    print(f"[OK] Generated {len(train_alphas)} Latin Hypercube parameter samples")
    
    # 2. Exact training dataset (60x60 grid per parameter = 108,000 points)
    dataset_dict = generate_exact_parametric_dataset(
        alpha_values=train_alphas,
        grid_resolution=60,
        output_path=f"{output_dir}/parametric_training_dataset.npz"
    )
    dataset = ParametricPINNDataset(dataset_dict)
    print(f"[OK] Training dataset created with {len(dataset):,} space-time-parameter points")
    
    # 3. Model Architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] Hardware device: {device} | dtype: torch.float64")
    
    model = ParametricModifiedMLP(
        n_input=3,
        n_output=1,
        n_hidden=64,
        n_layers=4,
        use_fourier=False,
        fourier_scale=1.0
    ).to(device).to(torch.float64)
    
    # 4. Training (Adam + L-BFGS)
    trainer = ParametricPINNTrainer(
        model=model,
        dataset=dataset,
        learning_rate=2e-3,
        batch_size=16384,
        device=device
    )
    
    trainer.train(
        epochs=800,
        lbfgs_iters=150,
        lambda_data=1.0,
        lambda_phys=0.01,
        log_interval=200
    )
    
    # 5. Validation on Dense 50-point Alpha Grid (at full 100x100 resolution)
    val_grid = np.linspace(a_low, a_high, 50)
    metrics, rel_l2_arr, max_err_arr = validate_surrogate_grid(model, val_grid, resolution=100, device=device)
    
    mean_rel_l2 = float(np.mean(rel_l2_arr))
    median_rel_l2 = float(np.median(rel_l2_arr))
    max_rel_l2 = float(np.max(rel_l2_arr))
    mean_max_err = float(np.mean(max_err_arr))
    
    print("\n" + "=" * 80)
    print("SURROGATE VALIDATION RESULTS (50 EVALUATION POINTS ACROSS ALPHA)")
    print("=" * 80)
    print(f"  Mean Relative L2 Error   : {mean_rel_l2:.4%}")
    print(f"  Median Relative L2 Error : {median_rel_l2:.4%}")
    print(f"  Max Relative L2 Error    : {max_rel_l2:.4%}")
    print(f"  Mean Max Absolute Error  : {mean_max_err:.4e}")
    print("=" * 80)
    
    # 6. Save Model Checkpoint
    weights_path = os.path.join(output_dir, "heat_equation_pinn.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"[OK] Model weights saved to {weights_path}")
    torch.save(model.state_dict(), os.path.join(output_dir, "parametric_pinn_weights.pth"))
    
    # 7. Generate Plots
    plot_surrogate_error_curves(val_grid, rel_l2_arr, max_err_arr, output_dir=f"{output_dir}/surrogate_validation", true_alpha=0.5)
    
    # Field comparison at true alpha = 0.5
    _, _, u_ex, _, _, _ = evaluate_exact_pde_field_and_derivatives(0.5, resolution=100)
    u_p, _, _, _, _, _ = evaluate_surrogate_and_derivatives(model, 0.5, resolution=100, device=device)
    plot_solution_field_comparison(u_ex, u_p, alpha=0.5, output_path=f"{output_dir}/surrogate_validation/field_comparison_alpha0.5.png")
    
    # Export reproducibility artifacts
    export_reproducibility_artifacts(
        model=model,
        metrics=metrics,
        config={"n_train_lhs": 30, "epochs": 800, "lbfgs_iters": 150},
        output_dir=f"{output_dir}/surrogate_validation"
    )
    print("[SUCCESS] All training artifacts and validation plots successfully saved.")


if __name__ == "__main__":
    main()
