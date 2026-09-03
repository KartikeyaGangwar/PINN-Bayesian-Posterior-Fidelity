r"""
Proposal Distribution Module for Metropolis-Hastings MCMC (1D Heat Equation)
=============================================================================
Mathematical Reference:
    Robert & Casella (2004) "Monte Carlo Statistical Methods", Springer, Chapter 7
    Cotter, Dashti, Stuart (2013) "MCMC Methods for Functions", SIAM UQ

Mathematical Formulation:
    Symmetric Gaussian Random Walk Proposal:
        \alpha' \sim q(\cdot \mid \alpha^{(k)}) = \mathcal{N}(\alpha^{(k)}, \sigma_{\text{prop}}^2)

    Symmetry Property:
        q(\alpha' \mid \alpha^{(k)}) = q(\alpha^{(k)} \mid \alpha')
        \implies \frac{q(\alpha^{(k)} \mid \alpha')}{q(\alpha' \mid \alpha^{(k)})} = 1
"""

import numpy as np
from typing import Optional, Union


class BaseProposal:
    r"""Abstract Base Class for MCMC proposal distributions q(\alpha' \mid \alpha)."""
    def propose(self, alpha_current: Union[float, np.ndarray]) -> float:
        raise NotImplementedError

    def log_q(self, alpha_from: Union[float, np.ndarray], alpha_to: Union[float, np.ndarray]) -> float:
        raise NotImplementedError

    @property
    def is_symmetric(self) -> bool:
        return False


class GaussianRandomWalkProposal(BaseProposal):
    r"""
    1D Symmetric Gaussian Random Walk Proposal:
        \alpha' = \alpha^{(k)} + \sigma_{\text{prop}} \cdot z, \quad z \sim \mathcal{N}(0, 1)
    """
    def __init__(self, scale: float = 0.05):
        r"""
        Inputs:
            scale: Proposal standard deviation \sigma_{\text{prop}} > 0.
        """
        if scale <= 0:
            raise ValueError(f"Proposal scale must be positive, got {scale}")
        self.scale = float(scale)

    def propose(self, alpha_current: Union[float, np.ndarray]) -> float:
        r"""Draws candidate proposal \alpha' \sim \mathcal{N}(\alpha^{(k)}, \sigma_{\text{prop}}^2)."""
        if isinstance(alpha_current, (list, tuple, np.ndarray)):
            alpha_val = float(alpha_current[0])
        else:
            alpha_val = float(alpha_current)

        z = np.random.randn()
        return float(alpha_val + self.scale * z)

    def log_q(self, alpha_from: Union[float, np.ndarray], alpha_to: Union[float, np.ndarray]) -> float:
        r"""Evaluates log proposal density \ln q(\alpha_{\text{to}} \mid \alpha_{\text{from}})."""
        a_from = float(alpha_from[0]) if isinstance(alpha_from, (list, tuple, np.ndarray)) else float(alpha_from)
        a_to = float(alpha_to[0]) if isinstance(alpha_to, (list, tuple, np.ndarray)) else float(alpha_to)

        diff = a_to - a_from
        return float(-0.5 * np.log(2.0 * np.pi * (self.scale ** 2)) - 0.5 * (diff / self.scale) ** 2)

    @property
    def is_symmetric(self) -> bool:
        """Random Walk Gaussian proposals are symmetric."""
        return True
