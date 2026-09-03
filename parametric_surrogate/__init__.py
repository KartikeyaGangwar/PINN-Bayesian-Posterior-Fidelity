"""
Parametric PINN Surrogate Package (1D Heat Equation)
====================================================
Provides offline dataset generation, parametric surrogate training, grid validation,
error metric quantification across alpha, and publication plotting routines.
"""

from .parameter_sampler import sample_parameter_domain_lhs, compute_parameter_bounds, PriorDomainConfig
from .exact_dataset_generator import generate_exact_parametric_dataset, evaluate_exact_pde_field_and_derivatives
from .dataset import ParametricPINNDataset
from .parametric_model import ParametricModifiedMLP, ParametricFourierFeatures
from .trainer import ParametricPINNTrainer
from .metrics import SurrogateMetricsContainer, compute_surrogate_metrics
from .validation import validate_surrogate_grid, evaluate_surrogate_and_derivatives
from .plots import plot_surrogate_error_curves, plot_solution_field_comparison
from .exports import export_reproducibility_artifacts
from .runner import run_parametric_surrogate_pipeline

__all__ = [
    "sample_parameter_domain_lhs",
    "compute_parameter_bounds",
    "PriorDomainConfig",
    "generate_exact_parametric_dataset",
    "evaluate_exact_pde_field_and_derivatives",
    "ParametricPINNDataset",
    "ParametricModifiedMLP",
    "ParametricFourierFeatures",
    "ParametricPINNTrainer",
    "SurrogateMetricsContainer",
    "compute_surrogate_metrics",
    "validate_surrogate_grid",
    "evaluate_surrogate_and_derivatives",
    "plot_surrogate_error_curves",
    "plot_solution_field_comparison",
    "export_reproducibility_artifacts",
    "run_parametric_surrogate_pipeline"
]
