"""
Standardized Metrics Module for Bayesian PINN Fidelity Research
===============================================================
Computes:
1. Global forward field and observation-space metrics
2. Parameter-resolved forward error e(alpha)
3. Prior-weighted and posterior-weighted forward errors
4. Bayesian posterior discrepancy metrics (W1, KS, bias, variance ratio)
5. Proposed Bayesian Fidelity Ratio (BFR)
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, Any, Tuple, Optional, Callable
import torch


def evaluate_parameter_resolved_error(
    exact_solver: Callable[[float], Any],
    pinn_solver: Callable[[float], Any],
    alpha_grid: np.ndarray,
    obs_operator: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, np.ndarray]:
    """
    Computes parameter-resolved forward error function e(alpha) across a dense grid of alpha.
    """
    rel_l2_errors = []
    max_abs_errors = []
    obs_errors = []
    
    for alpha_val in alpha_grid:
        u_exact = exact_solver(alpha_val).u
        u_pinn = pinn_solver(alpha_val).u
        diff = u_pinn - u_exact
        
        # Field relative L2
        norm_exact = np.linalg.norm(u_exact)
        rel_l2 = float(np.linalg.norm(diff) / (norm_exact + 1e-15))
        max_abs = float(np.max(np.abs(diff)))
        
        rel_l2_errors.append(rel_l2)
        max_abs_errors.append(max_abs)
        
        if obs_operator is not None:
            y_exact = obs_operator(u_exact)
            y_pinn = obs_operator(u_pinn)
            norm_y_exact = np.linalg.norm(y_exact)
            obs_err = float(np.linalg.norm(y_pinn - y_exact) / (norm_y_exact + 1e-15))
            obs_errors.append(obs_err)
            
    return {
        "alpha_grid": np.asarray(alpha_grid, dtype=float),
        "e_alpha_field": np.asarray(rel_l2_errors, dtype=float),
        "e_alpha_max_abs": np.asarray(max_abs_errors, dtype=float),
        "e_alpha_obs": np.asarray(obs_errors, dtype=float) if obs_operator is not None else np.zeros_like(alpha_grid)
    }


def compute_global_forward_metrics(
    e_alpha_dict: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Aggregates parameter-resolved errors into global summary statistics.
    """
    e_field = e_alpha_dict["e_alpha_field"]
    e_obs = e_alpha_dict["e_alpha_obs"]
    e_max_abs = e_alpha_dict["e_alpha_max_abs"]
    
    return {
        "mean_rel_l2": float(np.mean(e_field)),
        "median_rel_l2": float(np.median(e_field)),
        "p95_rel_l2": float(np.percentile(e_field, 95)),
        "max_rel_l2": float(np.max(e_field)),
        "mean_max_abs_error": float(np.mean(e_max_abs)),
        "mean_obs_error": float(np.mean(e_obs)),
        "median_obs_error": float(np.median(e_obs)),
        "p95_obs_error": float(np.percentile(e_obs, 95)),
        "max_obs_error": float(np.max(e_obs))
    }


def compute_weighted_forward_errors(
    alpha_grid: np.ndarray,
    e_alpha: np.ndarray,
    prior_density: np.ndarray,
    exact_posterior_density: np.ndarray
) -> Dict[str, float]:
    """
    Computes prior-weighted and posterior-weighted forward error diagnostics:
        E_global    = (1 / |Omega|) int e(alpha) d alpha
        E_prior     = int e(alpha) p(alpha) d alpha
        E_posterior = int e(alpha) pi_exact(alpha | y) d alpha
        RMS_posterior = sqrt( int e(alpha)^2 pi_exact(alpha | y) d alpha )
    """
    # Numerical integration using trapezoidal rule
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    
    # Normalize densities so numerical integral is 1.0
    p_norm = prior_density / (trap_fn(prior_density, alpha_grid) + 1e-15)
    pi_norm = exact_posterior_density / (trap_fn(exact_posterior_density, alpha_grid) + 1e-15)
    
    e_global = float(np.mean(e_alpha))
    e_prior = float(trap_fn(e_alpha * p_norm, alpha_grid))
    e_posterior = float(trap_fn(e_alpha * pi_norm, alpha_grid))
    rms_posterior = float(np.sqrt(trap_fn((e_alpha ** 2) * pi_norm, alpha_grid)))
    
    return {
        "E_global": e_global,
        "E_prior": e_prior,
        "E_posterior": e_posterior,
        "RMS_posterior": rms_posterior
    }


def compute_bayesian_posterior_metrics(
    samples_exact: np.ndarray,
    samples_pinn: np.ndarray,
    alpha_true: float = 0.5000
) -> Dict[str, Any]:
    """
    Computes rigorous stationary posterior discrepancy metrics:
    - Posterior means, variances, stds, 95% CIs
    - Posterior mean bias (Delta alpha), relative bias, bias in std units
    - Variance ratio Var(A) / Var(E)
    - Wasserstein-1 distance W1(pi_E, pi_A)
    - Kolmogorov-Smirnov statistic and p-value
    - Mean absolute posterior sample difference
    """
    mean_E = float(np.mean(samples_exact))
    std_E = float(np.std(samples_exact))
    var_E = float(np.var(samples_exact))
    ci_E = [float(np.percentile(samples_exact, 2.5)), float(np.percentile(samples_exact, 97.5))]
    
    mean_A = float(np.mean(samples_pinn))
    std_A = float(np.std(samples_pinn))
    var_A = float(np.var(samples_pinn))
    ci_A = [float(np.percentile(samples_pinn, 2.5)), float(np.percentile(samples_pinn, 97.5))]
    
    mean_bias = float(mean_A - mean_E)
    rel_bias = float(mean_bias / mean_E) if mean_E != 0 else 0.0
    bias_in_std_units = float(mean_bias / (std_E + 1e-15))
    var_ratio = float(var_A / (var_E + 1e-15))
    
    w1 = float(stats.wasserstein_distance(samples_exact, samples_pinn))
    ks_res = stats.ks_2samp(samples_exact, samples_pinn)
    
    # Coverage check
    covered_E = bool(ci_E[0] <= alpha_true <= ci_E[1])
    covered_A = bool(ci_A[0] <= alpha_true <= ci_A[1])
    
    return {
        "exact": {
            "mean": mean_E,
            "std": std_E,
            "var": var_E,
            "ci_95": ci_E,
            "ci_width": float(ci_E[1] - ci_E[0]),
            "covered_true": covered_E
        },
        "pinn": {
            "mean": mean_A,
            "std": std_A,
            "var": var_A,
            "ci_95": ci_A,
            "ci_width": float(ci_A[1] - ci_A[0]),
            "covered_true": covered_A
        },
        "discrepancy": {
            "mean_bias": mean_bias,
            "relative_mean_bias": rel_bias,
            "bias_in_std_units": bias_in_std_units,
            "variance_ratio": var_ratio,
            "wasserstein_1": w1,
            "ks_statistic": float(ks_res.statistic),
            "ks_p_value": float(ks_res.pvalue)
        }
    }


def compute_bayesian_fidelity_ratio(
    w1_research: float,
    w1_control: float
) -> float:
    """
    Computes the proposed Bayesian Fidelity Ratio (BFR):
        BFR = W1(Exact, PINN) / W1(Exact_1, Exact_2)
    where W1(Exact_1, Exact_2) is the empirical MCMC stochastic noise floor.
    """
    if w1_control <= 1e-15:
        return float("inf")
    return float(w1_research / w1_control)
