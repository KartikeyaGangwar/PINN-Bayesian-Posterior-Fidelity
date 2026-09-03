"""
Experiments Package: Bayesian PINN Fidelity & Error Localization
===============================================================
Comprehensive research suite investigating forward surrogate accuracy,
likelihood perturbations, error localization, and posterior fidelity.
"""

from .metrics import (
    compute_global_forward_metrics,
    evaluate_parameter_resolved_error,
    compute_weighted_forward_errors,
    compute_bayesian_posterior_metrics,
    compute_bayesian_fidelity_ratio
)

__all__ = [
    "compute_global_forward_metrics",
    "evaluate_parameter_resolved_error",
    "compute_weighted_forward_errors",
    "compute_bayesian_posterior_metrics",
    "compute_bayesian_fidelity_ratio"
]
