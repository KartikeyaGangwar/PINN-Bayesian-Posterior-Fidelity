r"""
Unnormalized Posterior Probability Distribution Module (\pi(\alpha \mid \boldsymbol{y}))
========================================================================================
Mathematical Formulation:
    Unnormalized Bayes' Rule:
        \pi_{\text{post}}(\alpha \mid \boldsymbol{y}) \propto p(\boldsymbol{y} \mid \mathcal{F}(\alpha)) \cdot \pi_{\text{prior}}(\alpha)
        
    Unnormalized Log Posterior:
        \ln \pi_{\text{post}}(\alpha \mid \boldsymbol{y}) = \ln \pi_{\text{prior}}(\alpha) + \ln p(\boldsymbol{y} \mid \mathcal{F}(\alpha))
"""

import numpy as np
from typing import Tuple, Callable, Dict, Any, Union, Optional
from .prior import Prior
from .likelihood import GaussianLikelihood
from .observation_operator import ObservationOperator


class Posterior:
    r"""
    Unnormalized Posterior Distribution \pi(\alpha \mid \boldsymbol{y}).
    Combines Prior and Likelihood through unnormalized Bayes' rule.
    """
    def __init__(
        self,
        prior: Prior,
        likelihood: GaussianLikelihood,
        obs_operator: ObservationOperator
    ):
        self.prior = prior
        self.likelihood = likelihood
        self.obs_operator = obs_operator

    def evaluate_components(
        self, 
        alpha: Union[float, np.ndarray], 
        y_obs: np.ndarray, 
        forward_solver_fn: Callable
    ) -> Dict[str, float]:
        r"""
        Evaluates log-prior, log-likelihood, misfit, and unnormalized log-posterior.
        
        Inputs:
            alpha: Parameter candidate \alpha > 0
            y_obs: Observed data vector \boldsymbol{y}_{\text{obs}}
            forward_solver_fn: Function evaluating \mathcal{F}(\alpha) \mapsto u(x,t)
            
        Outputs:
            Dictionary containing log_prior, log_likelihood, misfit, and log_posterior.
        """
        # 1. Evaluate log prior
        log_pr = self.prior.log_prior(alpha)
        if np.isneginf(log_pr):
            return {
                "log_prior": -np.inf,
                "log_likelihood": -np.inf,
                "misfit": np.inf,
                "log_posterior": -np.inf
            }

        # 2. Evaluate forward model u = \mathcal{F}(\alpha)
        solver_out = forward_solver_fn(alpha)
            
        # Extract the solution field cleanly
        if hasattr(solver_out, 'u'):
            u_field = solver_out.u
        elif hasattr(solver_out, 'u_pinn') and solver_out.u_pinn is not None:
            u_field = solver_out.u_pinn
        elif hasattr(solver_out, 'u_exact'):
            u_field = solver_out.u_exact
        else:
            u_field = solver_out

        # 3. Extract simulated observations y_pred = \mathcal{H}(u)
        y_pred = self.obs_operator(u_field)

        # 4. Compute log likelihood and data misfit
        log_like = self.likelihood.log_likelihood(y_obs, y_pred)
        misfit_val = self.likelihood.misfit(y_obs, y_pred)
        log_post = log_pr + log_like

        return {
            "log_prior": float(log_pr),
            "log_likelihood": float(log_like),
            "misfit": float(misfit_val),
            "log_posterior": float(log_post)
        }

    def log_posterior(
        self, 
        alpha: Union[float, np.ndarray], 
        y_obs: np.ndarray, 
        forward_solver_fn: Callable
    ) -> float:
        r"""Evaluates unnormalized log posterior \ln \pi(\alpha \mid \boldsymbol{y})."""
        comp = self.evaluate_components(alpha, y_obs, forward_solver_fn)
        return comp["log_posterior"]

    def evaluate_grid(
        self,
        alpha_grid: np.ndarray,
        y_obs: np.ndarray,
        forward_solver_fn: Callable
    ) -> Dict[str, np.ndarray]:
        """
        Evaluates the unnormalized posterior on a dense 1D grid of alpha values.
        Returns normalized posterior density array, log posterior array, and misfit array.
        """
        n_pts = len(alpha_grid)
        log_posts = np.zeros(n_pts, dtype=np.float64)
        misfits = np.zeros(n_pts, dtype=np.float64)
        log_likes = np.zeros(n_pts, dtype=np.float64)
        log_priors = np.zeros(n_pts, dtype=np.float64)

        for i, a_v in enumerate(alpha_grid):
            comp = self.evaluate_components(a_v, y_obs, forward_solver_fn)
            log_posts[i] = comp["log_posterior"]
            misfits[i] = comp["misfit"]
            log_likes[i] = comp["log_likelihood"]
            log_priors[i] = comp["log_prior"]

        # Compute normalized posterior density via trapezoidal integration
        max_lp = np.max(log_posts[~np.isneginf(log_posts)]) if np.any(~np.isneginf(log_posts)) else 0.0
        unnorm_pdf = np.exp(log_posts - max_lp)
        integral = np.trapezoid(unnorm_pdf, alpha_grid) if hasattr(np, 'trapezoid') else np.trapz(unnorm_pdf, alpha_grid)
        norm_pdf = unnorm_pdf / (integral + 1e-15)

        return {
            "alpha_grid": alpha_grid,
            "log_posterior": log_posts,
            "normalized_pdf": norm_pdf,
            "misfit": misfits,
            "log_likelihood": log_likes,
            "log_prior": log_priors
        }

    def __call__(
        self, 
        alpha: Union[float, np.ndarray], 
        y_obs: np.ndarray, 
        forward_solver_fn: Callable
    ) -> float:
        return self.log_posterior(alpha, y_obs, forward_solver_fn)
