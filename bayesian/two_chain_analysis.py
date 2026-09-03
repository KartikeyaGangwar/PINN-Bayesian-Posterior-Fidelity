r"""
Two-Chain Statistical & Mathematical Post-Sampling Analysis Module (1D Heat Equation)
======================================================================================
Mathematical Reference:
    Robert & Casella (2004) "Monte Carlo Statistical Methods", Springer, Chapter 12
    Gelman et al. (2013) "Bayesian Data Analysis", 3rd Edition, CRC Press
    Stuart (2010) "Inverse problems: A Bayesian perspective", Acta Numerica

Answers Core Scientific Questions (Post-Sampling):
--------------------------------------------------
1. Trajectory distance profile: d_t = |\alpha_E^{(t)} - \alpha_A^{(t)}|
2. Evolution of d_t over iterations (initial, mean, max, final, growth rate)
3. Post-burn-in stationary posterior statistics for \alpha (Exact vs PINN)
4. Distributional discrepancies: Wasserstein-1 distance \mathcal{W}_1(\pi_E, \pi_A), Kolmogorov-Smirnov test.
"""

import numpy as np
from scipy import stats
import json
import os
from typing import Dict, Any, Tuple, Optional, Union, List

from .two_chain_sampler import TwoChainMCMCResult
from .diagnostics import compute_autocorrelation, compute_effective_sample_size, compute_credible_interval


# =============================================================================
# STATISTICAL ANALYSIS ENGINE
# =============================================================================

def analyze_two_chain_experiment(
    result_or_npz: Union[TwoChainMCMCResult, str, Dict[str, np.ndarray]],
    burn_in: int = 1000,
    diagnostic_thresholds: Optional[List[float]] = None
) -> Dict[str, Any]:
    r"""
    Executes full statistical post-sampling analysis on two-chain experiment data.
    Can consume either an in-memory TwoChainMCMCResult or saved NPZ file path.
    """
    if diagnostic_thresholds is None:
        diagnostic_thresholds = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

    # 1. Load Trajectory Data
    if isinstance(result_or_npz, TwoChainMCMCResult):
        alpha_E = result_or_npz.get_alpha_E_array()
        alpha_A = result_or_npz.get_alpha_A_array()
        distance = result_or_npz.get_distance_array()
        log_post_E = result_or_npz.get_log_posterior_E_array()
        log_post_A = result_or_npz.get_log_posterior_A_array()
        alpha_0 = result_or_npz.alpha_0
        n_total = len(result_or_npz)
        acc_E = result_or_npz.final_acceptance_rate_E
        acc_A = result_or_npz.final_acceptance_rate_A
        runtime = result_or_npz.runtime_seconds
        exp_type = result_or_npz.experiment_type
    elif isinstance(result_or_npz, str):
        with np.load(result_or_npz) as data:
            alpha_E = np.array(data["alpha_E"], dtype=float)
            alpha_A = np.array(data["alpha_A"], dtype=float)
            distance = np.array(data["distance"], dtype=float)
            log_post_E = np.array(data["log_posterior_E"], dtype=float)
            log_post_A = np.array(data["log_posterior_A"], dtype=float)
            alpha_0 = float(data["alpha_0"])
            n_total = len(alpha_E)
            acc_E = float(np.mean(data["accepted_E"])) if "accepted_E" in data else None
            acc_A = float(np.mean(data["accepted_A"])) if "accepted_A" in data else None
            runtime = float(data.get("runtime_seconds", 0.0))
            exp_type = str(data.get("experiment_type", "research"))
    else:
        raise ValueError("Unsupported input format for analyze_two_chain_experiment.")

    # Apply burn-in for stationary posterior comparisons
    b_idx = min(burn_in, max(0, n_total - 2))
    alpha_E_post = alpha_E[b_idx:]
    alpha_A_post = alpha_A[b_idx:]
    dist_post = distance[b_idx:]
    post_diff = np.abs(log_post_E - log_post_A)

    # 2. Distance Trajectory Diagnostics
    distance_stats = {
        "initial_distance": float(distance[0]),
        "mean_distance_full": float(np.mean(distance)),
        "max_distance_full": float(np.max(distance)),
        "final_distance": float(distance[-1]),
        "mean_distance_post_burnin": float(np.mean(dist_post)),
        "mean_log_posterior_diff": float(np.mean(post_diff)),
        "max_log_posterior_diff": float(np.max(post_diff))
    }

    # 3. Post-Hoc Diagnostic Threshold Sensitivity
    diagnostic_crossings = {}
    for eps in diagnostic_thresholds:
        exceed_idx = np.where(distance > eps)[0]
        t_first_crossing = int(exceed_idx[0]) if len(exceed_idx) > 0 else None

        sustained_step = None
        if len(exceed_idx) > 0:
            for idx in exceed_idx:
                if idx + 5 <= n_total and np.all(distance[idx:idx + 5] > eps):
                    sustained_step = int(idx)
                    break

        diagnostic_crossings[f"eps_{eps:.3f}"] = {
            "diagnostic_threshold": float(eps),
            "first_crossing_step": t_first_crossing,
            "sustained_crossing_step": sustained_step,
            "pre_crossing_steps": t_first_crossing if t_first_crossing is not None else n_total,
            "pre_crossing_percentage": float((t_first_crossing / n_total * 100) if t_first_crossing is not None else 100.0)
        }

    # 4. Divergence Growth Profile
    if len(distance) > 1:
        steps_arr = np.arange(len(distance))
        slope, intercept, r_value, p_value, std_err = stats.linregress(steps_arr, distance)
        divergence_profile = {
            "linear_slope_per_step": float(slope),
            "linear_r2": float(r_value**2),
            "overall_trend": "increasing" if slope > 1e-6 else ("decreasing" if slope < -1e-6 else "stable")
        }
    else:
        divergence_profile = {"linear_slope_per_step": 0.0, "linear_r2": 0.0, "overall_trend": "stable"}

    # 5. Stationary Marginal Posterior Statistics (Post Burn-In)
    a_E_low, a_E_high = compute_credible_interval(alpha_E_post)
    a_A_low, a_A_high = compute_credible_interval(alpha_A_post)

    marginal_stats = {
        "exact_chain": {
            "alpha": {
                "mean": float(np.mean(alpha_E_post)),
                "std": float(np.std(alpha_E_post)),
                "median": float(np.median(alpha_E_post)),
                "cred_95": [float(a_E_low), float(a_E_high)],
                "ess": compute_effective_sample_size(alpha_E_post)
            }
        },
        "approx_chain": {
            "alpha": {
                "mean": float(np.mean(alpha_A_post)),
                "std": float(np.std(alpha_A_post)),
                "median": float(np.median(alpha_A_post)),
                "cred_95": [float(a_A_low), float(a_A_high)],
                "ess": compute_effective_sample_size(alpha_A_post)
            }
        }
    }

    # 6. Distribution Discrepancy Metrics
    w1_dist = float(stats.wasserstein_distance(alpha_E_post, alpha_A_post))
    ks_stat, ks_pval = stats.ks_2samp(alpha_E_post, alpha_A_post)
    mean_bias = float(np.mean(alpha_A_post) - np.mean(alpha_E_post))
    std_ratio = float(np.std(alpha_A_post) / (np.std(alpha_E_post) + 1e-15))

    discrepancy_metrics = {
        "wasserstein_1_distance": w1_dist,
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_pval),
        "mean_bias": mean_bias,
        "std_ratio": std_ratio
    }

    analysis_report = {
        "metadata": {
            "experiment_type": exp_type,
            "n_total_samples": n_total,
            "burn_in": burn_in,
            "n_post_burnin_samples": len(alpha_E_post),
            "common_initial_alpha": float(alpha_0),
            "runtime_seconds": runtime,
            "acceptance_rate_E": acc_E,
            "acceptance_rate_A": acc_A
        },
        "distance_diagnostics": distance_stats,
        "divergence_profile": divergence_profile,
        "diagnostic_crossings": diagnostic_crossings,
        "marginal_posterior_statistics": marginal_stats,
        "distribution_discrepancies": discrepancy_metrics
    }

    return analysis_report
