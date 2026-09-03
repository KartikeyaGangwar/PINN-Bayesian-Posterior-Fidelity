r"""
Prior-Derived Parameter Domain & Latin Hypercube Sampler Module (1D Heat Equation)
===================================================================================
Mathematical Formulation:
    For thermal diffusivity \alpha > 0 with prior \pi_{\text{prior}}(\alpha):
    1. Statistical Prior Bounds:
       \alpha_{\text{min}} = F_{\pi}^{-1}((1 - \gamma)/2), \quad \alpha_{\text{max}} = F_{\pi}^{-1}((1 + \gamma)/2)
    2. Physical Positivity Constraint:
       \alpha \ge \epsilon > 0
    3. Training Domain:
       \alpha \in [\max(\alpha_{\text{min}}, \epsilon), \alpha_{\text{max}}]
"""

import numpy as np
from scipy import stats
from scipy.stats import qmc
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional

from bayesian.prior import Prior, NormalPrior, LogNormalPrior, BaseParameterPrior


@dataclass
class PriorDomainConfig:
    """Configuration container for prior-derived parameter training domain."""
    credible_mass: float = 0.997
    enforce_physical_constraints: bool = True
    min_alpha_threshold: float = 1e-4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParameterDomainSummary:
    """Detailed metadata container for parameter bounds derivation."""
    parameter_name: str
    prior_type: str
    prior_params: Dict[str, float]
    credible_mass: float
    inv_cdf_lower: float
    inv_cdf_upper: float
    physical_min: float
    physical_max: float
    physical_clipping_applied: bool
    final_lower_bound: float
    final_upper_bound: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_parameter_bounds(
    prior: Prior,
    config: Optional[PriorDomainConfig] = None
) -> Tuple[float, float, ParameterDomainSummary]:
    """Computes statistical prior bounds for alpha and applies physical positivity."""
    if config is None:
        config = PriorDomainConfig()

    dist = prior.alpha_prior
    gamma = config.credible_mass
    alpha_tail = (1.0 - gamma) / 2.0

    if hasattr(dist, "dist") and hasattr(dist.dist, "ppf"):
        q_low = float(dist.dist.ppf(alpha_tail))
        q_high = float(dist.dist.ppf(1.0 - alpha_tail))
    else:
        q_low, q_high = 0.05, 2.0

    prior_type = dist.__class__.__name__
    prior_params = {}
    if isinstance(dist, NormalPrior):
        prior_params = {"mean": dist.mean, "std": dist.std}
    elif isinstance(dist, LogNormalPrior):
        prior_params = {"mu": dist.mu, "sigma": dist.sigma}

    phys_min = config.min_alpha_threshold
    phys_max = np.inf
    final_low = max(q_low, phys_min) if config.enforce_physical_constraints else q_low
    clipping = bool(q_low < phys_min and config.enforce_physical_constraints)
    final_high = q_high

    summary = ParameterDomainSummary(
        parameter_name="alpha",
        prior_type=prior_type,
        prior_params=prior_params,
        credible_mass=gamma,
        inv_cdf_lower=q_low,
        inv_cdf_upper=q_high,
        physical_min=phys_min,
        physical_max=phys_max,
        physical_clipping_applied=clipping,
        final_lower_bound=final_low,
        final_upper_bound=final_high
    )

    return final_low, final_high, summary


def sample_parameter_domain_lhs(
    prior: Prior,
    n_samples: int = 50,
    config: Optional[PriorDomainConfig] = None,
    seed: int = 42
) -> Tuple[np.ndarray, float, float, ParameterDomainSummary]:
    """
    Generates Latin Hypercube / Log-Uniform samples of alpha across the derived prior bounds.
    """
    a_low, a_high, summary = compute_parameter_bounds(prior, config)
    
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    u_samples = sampler.random(n=n_samples).flatten()
    
    # Log-uniform mapping for strictly positive alpha
    log_a_low = np.log(max(a_low, 1e-4))
    log_a_high = np.log(a_high)
    alpha_samples = np.exp(log_a_low + u_samples * (log_a_high - log_a_low))
    
    return alpha_samples, a_low, a_high, summary
