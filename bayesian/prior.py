r"""
Prior Probability Distribution Module (\pi_{\text{prior}}(\alpha))
===================================================================
Mathematical Formulation:
    For the 1D Parametric Heat Equation (u_t = \alpha u_{xx}), the unknown physical
    parameter is the thermal diffusivity \alpha > 0.

Physical Constraints:
    - \alpha > 0 is a strict physical and mathematical requirement.
    - \alpha = 0 eliminates thermal dissipation (u_t = 0).
    - \alpha < 0 corresponds to backward heat flow (ill-posed initial value problem).
    - Supported positive-support prior distributions: LogNormal, Gamma, Truncated Normal.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Union


class BaseParameterPrior:
    """Abstract Base Class for parameter prior distributions."""
    def log_pdf(self, val: float) -> float:
        raise NotImplementedError
        
    def pdf(self, val: float) -> float:
        log_p = self.log_pdf(val)
        return 0.0 if np.isneginf(log_p) else float(np.exp(log_p))

    def sample(self, size: int = 1) -> np.ndarray:
        raise NotImplementedError


class LogNormalPrior(BaseParameterPrior):
    r"""
    LogNormal Prior Distribution for strictly positive parameter \alpha:
        \ln \alpha \sim \mathcal{N}(\mu, \sigma^2)
    """
    def __init__(self, mu: float = 0.0, sigma: float = 0.5):
        self.mu = mu
        self.sigma = sigma
        self.dist = stats.lognorm(s=sigma, scale=np.exp(mu))

    def log_pdf(self, val: float) -> float:
        if val <= 0.0:
            return -np.inf
        return float(self.dist.logpdf(val))

    def sample(self, size: int = 1) -> np.ndarray:
        return self.dist.rvs(size=size)


class NormalPrior(BaseParameterPrior):
    r"""Gaussian (Normal) Prior Distribution: \alpha \sim \mathcal{N}(\mu, \sigma^2)."""
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        self.dist = stats.norm(loc=mean, scale=std)

    def log_pdf(self, val: float) -> float:
        return float(self.dist.logpdf(val))

    def sample(self, size: int = 1) -> np.ndarray:
        return self.dist.rvs(size=size)


class GammaPrior(BaseParameterPrior):
    r"""Gamma Prior Distribution: \alpha \sim \text{Gamma}(a, \text{scale}=1/b)."""
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self.dist = stats.gamma(a=alpha, scale=1.0 / beta)

    def log_pdf(self, val: float) -> float:
        if val <= 0.0:
            return -np.inf
        return float(self.dist.logpdf(val))

    def sample(self, size: int = 1) -> np.ndarray:
        return self.dist.rvs(size=size)


class UniformPrior(BaseParameterPrior):
    r"""Uniform Prior Distribution: \alpha \sim \mathcal{U}(a, b)."""
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        self.dist = stats.uniform(loc=low, scale=high - low)

    def log_pdf(self, val: float) -> float:
        if val < self.low or val > self.high:
            return -np.inf
        return float(self.dist.logpdf(val))

    def sample(self, size: int = 1) -> np.ndarray:
        return self.dist.rvs(size=size)


class TruncatedNormalPrior(BaseParameterPrior):
    r"""Truncated Normal Prior Distribution on [low, high]: \alpha \sim \mathcal{N}_{[a,b]}(\mu, \sigma^2)."""
    def __init__(self, mean: float, std: float, low: float = 0.0, high: float = np.inf):
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high
        a_scaled = (low - mean) / std
        b_scaled = (high - mean) / std
        self.dist = stats.truncnorm(a_scaled, b_scaled, loc=mean, scale=std)

    def log_pdf(self, val: float) -> float:
        if val < self.low or val > self.high:
            return -np.inf
        return float(self.dist.logpdf(val))

    def sample(self, size: int = 1) -> np.ndarray:
        return self.dist.rvs(size=size)


class Prior:
    r"""
    Prior Distribution \pi_{\text{prior}}(\alpha) over thermal diffusivity \alpha.
    Wraps a 1D BaseParameterPrior distribution (defaults to LogNormalPrior(mu=ln(0.5), sigma=0.5)).
    """
    def __init__(self, alpha_prior: Optional[BaseParameterPrior] = None):
        if alpha_prior is None:
            # Default prior: LogNormal centered around 0.5
            self.alpha_prior = LogNormalPrior(mu=float(np.log(0.5)), sigma=0.5)
        else:
            self.alpha_prior = alpha_prior

    def log_prior(self, alpha: Union[float, np.ndarray]) -> float:
        r"""Evaluates log prior density \ln \pi_{\text{prior}}(\alpha)."""
        if isinstance(alpha, (list, tuple, np.ndarray)):
            if len(alpha) == 1:
                val = float(alpha[0])
            else:
                val = float(alpha[0])
        else:
            val = float(alpha)
        return self.alpha_prior.log_pdf(val)

    def prior(self, alpha: Union[float, np.ndarray]) -> float:
        r"""Evaluates prior density \pi_{\text{prior}}(\alpha)."""
        log_p = self.log_prior(alpha)
        if np.isneginf(log_p):
            return 0.0
        return float(np.exp(log_p))

    def sample(self, size: int = 1) -> Union[float, np.ndarray]:
        """Draws sample(s) from the prior distribution."""
        s = self.alpha_prior.sample(size=size)
        if size == 1:
            return float(s[0]) if isinstance(s, np.ndarray) else float(s)
        return s

    def __call__(self, alpha: Union[float, np.ndarray]) -> float:
        return self.log_prior(alpha)
