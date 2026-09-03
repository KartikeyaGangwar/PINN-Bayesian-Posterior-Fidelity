r"""
Gaussian Likelihood Module (p(\boldsymbol{y} \mid \boldsymbol{\theta}))
===================================================================
Mathematical Reference:
    Alexanderian (2021) "Optimal Experimental Design for Infinite-Dimensional Bayesian Inverse Problems"
    Stuart (2010) "Inverse problems: A Bayesian perspective", Acta Numerica

Mathematical Equations:
    Observation Model:
        \boldsymbol{y}_{\text{obs}} = \mathcal{H}(\mathcal{F}(\boldsymbol{\theta})) + \boldsymbol{\eta}, \quad \boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \sigma_{\text{noise}}^2 \mathbf{I}_M)
        
    Log Likelihood:
        \ln p(\boldsymbol{y}_{\text{obs}} \mid \boldsymbol{\theta}) = -\frac{M}{2} \ln(2\pi \sigma_{\text{noise}}^2) - \frac{1}{2 \sigma_{\text{noise}}^2} \|\boldsymbol{y}_{\text{obs}} - \boldsymbol{y}_{\text{pred}}\|_2^2
        
    Misfit Functional:
        \mathcal{J}_{\text{misfit}}(\boldsymbol{\theta}) = \frac{1}{2 \sigma_{\text{noise}}^2} \|\boldsymbol{y}_{\text{obs}} - \boldsymbol{y}_{\text{pred}}\|_2^2
        
    Negative Log Likelihood:
        -\ln p(\boldsymbol{y}_{\text{obs}} \mid \boldsymbol{\theta}) = \mathcal{J}_{\text{misfit}}(\boldsymbol{\theta}) + \frac{M}{2} \ln(2\pi \sigma_{\text{noise}}^2)

Description:
    Computes data likelihood and misfit functionals assuming independent and identically 
    distributed (i.i.d.) Gaussian measurement noise.
    Separated completely from Forward Solver, Observation Operator, and Prior.
"""

import numpy as np
from typing import Union, Tuple


class GaussianLikelihood:
    r"""
    Likelihood Model p(\boldsymbol{y} \mid \boldsymbol{\theta}) for i.i.d. Gaussian measurement noise.
    """
    def __init__(self, noise_std: float):
        r"""
        Inputs:
            noise_std: Measurement noise standard deviation \sigma_{\text{noise}}
        """
        self.noise_std = max(float(noise_std), 1e-12)
        self.variance = self.noise_std ** 2

    def misfit(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
        r"""
        Computes weighted data misfit functional \mathcal{J}_{\text{misfit}}(\boldsymbol{\theta}).
        
        Equation:
            \mathcal{J}_{\text{misfit}} = \frac{1}{2 \sigma_{\text{noise}}^2} \|\boldsymbol{y}_{\text{obs}} - \boldsymbol{y}_{\text{pred}}\|_2^2
        """
        diff = y_obs.flatten() - y_pred.flatten()
        return float(0.5 * np.sum(diff ** 2) / self.variance)

    def log_likelihood(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
        r"""
        Computes log likelihood \ln p(\boldsymbol{y}_{\text{obs}} \mid \boldsymbol{\theta}).
        
        Equation:
            \ln p(\boldsymbol{y} \mid \boldsymbol{\theta}) = -\frac{M}{2} \ln(2\pi \sigma_{\text{noise}}^2) - \mathcal{J}_{\text{misfit}}(\boldsymbol{\theta})
        """
        M = y_obs.size
        misfit_val = self.misfit(y_obs, y_pred)
        norm_const = 0.5 * M * np.log(2.0 * np.pi * self.variance)
        return float(-norm_const - misfit_val)

    def negative_log_likelihood(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
        r"""Computes negative log likelihood -\ln p(\boldsymbol{y}_{\text{obs}} \mid \boldsymbol{\theta})."""
        return -self.log_likelihood(y_obs, y_pred)

    def __call__(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
        return self.log_likelihood(y_obs, y_pred)
