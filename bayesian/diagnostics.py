r"""
MCMC Diagnostics and Visualization Module (1D Heat Equation)
============================================================
Mathematical Reference:
    Robert & Casella (2004) "Monte Carlo Statistical Methods", Springer, Chapter 12
    Gelman et al. (2013) "Bayesian Data Analysis", 3rd Edition, CRC Press

Supported Diagnostics:
- Acceptance Rate & Burn-In Discard
- Posterior Sample Statistics for \alpha (Mean, Std, Median, 95% Credible Intervals)
- Autocorrelation Function (ACF)
- Effective Sample Size (ESS)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Tuple, Optional
from .chain import MCMCChain


def compute_autocorrelation(samples: np.ndarray, max_lag: int = 100) -> np.ndarray:
    r"""Computes empirical autocorrelation function \hat{\rho}_k for lag k = 0, ..., K."""
    n = len(samples)
    if n <= 1:
        return np.array([1.0])
        
    mean = np.mean(samples)
    var = np.var(samples)
    if var == 0:
        return np.ones(min(max_lag + 1, n))
        
    norm_samples = samples - mean
    max_lag = min(max_lag, n - 1)
    
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    fft_vals = np.fft.fft(norm_samples, n=n_fft)
    autocov = np.fft.ifft(fft_vals * np.conj(fft_vals)).real[:max_lag + 1]
    autocov /= n
    
    return autocov / var


def compute_effective_sample_size(samples: np.ndarray, max_lag: int = 100) -> float:
    r"""Computes Effective Sample Size (ESS) based on integrated autocorrelation time."""
    n = len(samples)
    if n <= 1:
        return 1.0
        
    acf = compute_autocorrelation(samples, max_lag=max_lag)
    
    cutoff = 1
    while cutoff < len(acf) and acf[cutoff] > 0:
        cutoff += 1
        
    integrated_tau = 1.0 + 2.0 * np.sum(acf[1:cutoff])
    return float(n / max(integrated_tau, 1.0))


def compute_credible_interval(samples: np.ndarray, cred_level: float = 0.95) -> Tuple[float, float]:
    """Computes equal-tailed credible interval [lower, upper]."""
    alpha = 1.0 - cred_level
    lower = float(np.percentile(samples, 100 * (alpha / 2.0)))
    upper = float(np.percentile(samples, 100 * (1.0 - alpha / 2.0)))
    return lower, upper


def analyze_chain(chain: MCMCChain, burn_in: int = 2000) -> Dict[str, Any]:
    """Computes complete statistical diagnostics for an MCMC chain post burn-in."""
    alpha_samples = chain.get_alpha_array(burn_in=burn_in)
    log_post_samples = chain.get_log_posterior_array(burn_in=burn_in)
    
    a_low, a_high = compute_credible_interval(alpha_samples)
    
    return {
        "n_total": len(chain),
        "n_burn_in": burn_in,
        "n_post_burnin": len(alpha_samples),
        "acceptance_rate": chain.acceptance_rate,
        "alpha_stats": {
            "mean": float(np.mean(alpha_samples)),
            "std": float(np.std(alpha_samples)),
            "median": float(np.median(alpha_samples)),
            "cred_95": [float(a_low), float(a_high)],
            "ess": compute_effective_sample_size(alpha_samples),
            "autocorr_lag1": float(compute_autocorrelation(alpha_samples, max_lag=1)[1]) if len(alpha_samples) > 1 else 0.0
        },
        "log_posterior_stats": {
            "mean": float(np.mean(log_post_samples)),
            "std": float(np.std(log_post_samples)),
            "median": float(np.median(log_post_samples)),
            "max": float(np.max(log_post_samples)),
            "min": float(np.min(log_post_samples))
        }
    }


def plot_mcmc_diagnostics(
    chain: MCMCChain,
    burn_in: int = 2000,
    true_alpha: Optional[float] = None,
    output_dir: str = "results",
    dpi: int = 300
) -> Dict[str, str]:
    """Generates and saves standard single-chain MCMC diagnostic figures."""
    os.makedirs(output_dir, exist_ok=True)
    alpha_all = chain.get_alpha_array(burn_in=0)
    alpha_post = chain.get_alpha_array(burn_in=burn_in)
    iters = np.arange(len(alpha_all))

    # 1. Trace plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, alpha_all, color="#1f77b4", linewidth=1.0, alpha=0.85)
    ax.axvline(burn_in, color="red", linestyle="--", label=f"Burn-in ({burn_in})")
    if true_alpha is not None:
        ax.axhline(true_alpha, color="black", linestyle=":", label=f"True alpha={true_alpha:.3f}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("alpha")
    ax.set_title("MCMC Parameter Trace")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    p1 = os.path.join(output_dir, "mcmc_trace.png")
    fig.savefig(p1, dpi=dpi)
    plt.close()

    # 2. Posterior Histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(alpha_post, bins=30, density=True, color="#2ca02c", alpha=0.6, edgecolor="black")
    if true_alpha is not None:
        ax.axvline(true_alpha, color="black", linestyle="--", linewidth=1.5, label=f"True alpha={true_alpha:.3f}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Posterior Density")
    ax.set_title("Post-Burnin Posterior Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    p2 = os.path.join(output_dir, "mcmc_posterior_hist.png")
    fig.savefig(p2, dpi=dpi)
    plt.close()

    return {"trace": p1, "hist": p2}
