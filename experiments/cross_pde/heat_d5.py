r"""
5D Multi-Mode Parametric Diffusion Benchmark (d=5)
===================================================
Defines the 5-dimensional multi-mode parametric thermal diffusion benchmark:
    u_t = \sum_{k=1}^5 c_k(\theta) \phi_k''(x),  (x, t) \in [0, 1] \times [0, 1]
where:
    \phi_k(x) = (1/k) \sin(k \pi x)
    c_k(\theta) = \theta_1 + 0.5 * \theta_k,  k = 1, ..., 5
    \theta = (\theta_1, \theta_2, \theta_3, \theta_4, \theta_5) \in \Theta \subset \mathbb{R}^5

Analytical Solution:
    u_exact(x, t; \theta) = \sum_{k=1}^5 (1/k) \sin(k \pi x) \exp(-(k \pi)^2 c_k(\theta) t)

Metric:
    100-direction Sliced Wasserstein-1 distance (SW1) and coordinate marginal W1.
"""

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional, Callable


@dataclass
class HeatD5Config:
    name: str = "heat_d5"
    display_name: str = "5D Multi-Mode Diffusion"
    dimension: int = 5
    true_param: np.ndarray = None
    param_bounds: np.ndarray = None
    noise_std: float = 0.010
    n_sensors: int = 40

    def __post_init__(self):
        if self.true_param is None:
            self.true_param = np.array([0.50, 0.08, -0.05, 0.04, -0.02], dtype=np.float64)
        if self.param_bounds is None:
            self.param_bounds = np.array([
                [0.20, 1.00],
                [-0.15, 0.15],
                [-0.15, 0.15],
                [-0.10, 0.10],
                [-0.10, 0.10]
            ], dtype=np.float64)


class HeatD5MultiModeBenchmark:
    """
    Benchmark suite for 5D Parametric Multi-Mode Thermal Diffusion.
    """
    def __init__(self, config: Optional[HeatD5Config] = None):
        self.config = config if config is not None else HeatD5Config()
        self.d = self.config.dimension
        self.bounds = self.config.param_bounds
        self.true_param = self.config.true_param
        self.prior_means = np.mean(self.bounds, axis=1)
        self.prior_stds = (self.bounds[:, 1] - self.bounds[:, 0]) / 4.0

    def modal_decay_coefficients(self, theta: np.ndarray) -> np.ndarray:
        """Computes c_k(theta) = theta_1 + 0.5 * theta_k for k=1..5."""
        c = np.zeros(5, dtype=np.float64)
        for k in range(5):
            c[k] = theta[0] + 0.5 * theta[k]
        return c

    def exact_solution(self, x: np.ndarray, t: np.ndarray, theta: np.ndarray) -> np.ndarray:
        r"""
        Computes analytical exact solution u(x, t; theta) to machine precision:
            u(x, t; theta) = \sum_{k=1}^5 (1/k) \sin(k \pi x) \exp(-(k \pi)^2 c_k(\theta) t)
        """
        x_arr = np.asarray(x, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        u = np.zeros_like(x_arr * t_arr, dtype=np.float64)
        c = self.modal_decay_coefficients(theta)
        for k in range(1, 6):
            decay = np.exp(-((k * np.pi) ** 2) * c[k - 1] * t_arr)
            spatial_mode = (1.0 / k) * np.sin(k * np.pi * x_arr)
            u += spatial_mode * decay
        return u

    def exact_solution_torch(self, inputs_grad: torch.Tensor) -> torch.Tensor:
        r"""
        Computes analytical exact solution u(x, t; theta) on PyTorch tensors preserving autograd graph:
            u(x, t; theta) = \sum_{k=1}^5 (1/k) \sin(k \pi x) \exp(-(k \pi)^2 c_k(\theta) t)
        """
        x_col = inputs_grad[:, 0:1]
        t_col = inputs_grad[:, 1:2]
        theta_cols = inputs_grad[:, 2:7]
        u = torch.zeros_like(x_col)
        for k in range(1, 6):
            c_k = theta_cols[:, 0:1] + 0.5 * theta_cols[:, (k - 1):k]
            decay_k = torch.exp(-((k * np.pi) ** 2) * c_k * t_col)
            phi_k = (1.0 / k) * torch.sin(k * np.pi * x_col)
            u = u + phi_k * decay_k
        return u

    def spatial_modes(self, x: np.ndarray) -> np.ndarray:
        """Returns array of shape [5, len(x)] containing each phi_k(x)."""
        x_arr = np.asarray(x, dtype=np.float64)
        modes = np.zeros((5, len(x_arr)), dtype=np.float64)
        for k in range(1, 6):
            modes[k - 1] = (1.0 / k) * np.sin(k * np.pi * x_arr)
        return modes

    def compute_pde_residual(self, u_pred: torch.Tensor, inputs_grad: torch.Tensor) -> torch.Tensor:
        r"""
        Computes autograd PDE residual:
            R = u_t - \sum_{k=1}^5 c_k(\theta) \phi_k''(x) \exp(-(k\pi)^2 c_k(\theta) t)
        """
        grads = torch.autograd.grad(u_pred, inputs_grad, torch.ones_like(u_pred), create_graph=True)[0]
        dudt = grads[:, 1:2]
        x_col = inputs_grad[:, 0:1]
        t_col = inputs_grad[:, 1:2]
        theta_cols = inputs_grad[:, 2:7]

        # Analytical diffusion operator sum_k c_k phi_k''(x)
        diff_sum = torch.zeros_like(dudt)
        for k in range(1, 6):
            c_k = theta_cols[:, 0:1] + 0.5 * theta_cols[:, (k - 1):k]
            decay_k = torch.exp(-((k * np.pi) ** 2) * c_k * t_col)
            d2phi_k = - (k * (np.pi ** 2)) * torch.sin(k * np.pi * x_col)
            diff_sum = diff_sum + c_k * d2phi_k * decay_k

        return dudt - diff_sum

    def log_prior(self, theta: np.ndarray) -> float:
        """Evaluates truncated Gaussian log-prior over compact box Theta."""
        for i in range(self.d):
            if theta[i] < self.bounds[i, 0] or theta[i] > self.bounds[i, 1]:
                return -np.inf
        return float(np.sum(stats.norm.logpdf(theta, loc=self.prior_means, scale=self.prior_stds)))

    def log_likelihood(self, theta: np.ndarray, y_obs: np.ndarray, sensors: np.ndarray) -> float:
        """Evaluates Gaussian log-likelihood given sparse sensor observations."""
        for i in range(self.d):
            if theta[i] < self.bounds[i, 0] or theta[i] > self.bounds[i, 1]:
                return -np.inf
        preds = np.array([self.exact_solution(sensors[m, 0], sensors[m, 1], theta) for m in range(len(sensors))])
        norm_const = 0.5 * len(sensors) * np.log(2.0 * np.pi * (self.config.noise_std ** 2))
        misfit = 0.5 * np.sum((preds - y_obs) ** 2) / (self.config.noise_std ** 2)
        return float(-norm_const - misfit)

    def sample_posterior(
        self,
        log_lik_fn: Optional[Callable[[np.ndarray], float]] = None,
        y_obs: Optional[np.ndarray] = None,
        sensors: Optional[np.ndarray] = None,
        n_samples: int = 15000,
        burnin: int = 3000,
        seed: int = 42
    ) -> np.ndarray:
        """
        Executes Metropolis-Hastings MCMC sampling in R^5.
        If log_lik_fn is provided (e.g. evaluating a neural network surrogate),
        it uses log_lik_fn(theta) + log_prior(theta).
        Otherwise, evaluates the exact reference likelihood via self.log_likelihood.
        """
        if log_lik_fn is None:
            if y_obs is None or sensors is None:
                raise ValueError("y_obs and sensors must be provided when log_lik_fn is None.")
            eval_lik = lambda th: self.log_likelihood(th, y_obs, sensors)
        else:
            eval_lik = log_lik_fn

        rng = np.random.default_rng(seed)
        samples = np.zeros((n_samples, self.d), dtype=np.float64)
        curr = self.true_param + rng.normal(0.0, 0.01, size=self.d)
        curr_lp = eval_lik(curr) + self.log_prior(curr)
        prop_std = np.array([0.015, 0.012, 0.012, 0.010, 0.010], dtype=np.float64)

        for i in range(n_samples):
            prop = curr + rng.normal(0.0, prop_std)
            prop_lp = eval_lik(prop) + self.log_prior(prop)
            if np.log(rng.uniform(0.0, 1.0) + 1e-300) < (prop_lp - curr_lp):
                curr = prop
                curr_lp = prop_lp
            samples[i] = curr
        return samples[burnin:]

    def sample_reference_posterior(
        self,
        y_obs: np.ndarray,
        sensors: np.ndarray,
        n_samples: int = 15000,
        burnin: int = 3000,
        seed: int = 42
    ) -> np.ndarray:
        """Executes reference MCMC sampling of exact posterior in R^5."""
        return self.sample_posterior(
            log_lik_fn=None,
            y_obs=y_obs,
            sensors=sensors,
            n_samples=n_samples,
            burnin=burnin,
            seed=seed
        )

    @staticmethod
    def compute_sliced_wasserstein_1(
        samples_a: np.ndarray,
        samples_b: np.ndarray,
        n_projections: int = 100,
        seed: int = 42
    ) -> Tuple[float, List[float]]:
        r"""
        Computes 100-direction Sliced Wasserstein-1 distance (SW1) between two empirical distributions:
            SW1 = (1/L) \sum_{l=1}^L W1(v_l^T theta_A, v_l^T theta_B)
        where v_l are drawn uniformly from the unit hypersphere S^{d-1}.
        """
        d = samples_a.shape[1]
        rng = np.random.default_rng(seed)
        proj_dirs = rng.normal(0.0, 1.0, size=(n_projections, d))
        proj_dirs /= np.linalg.norm(proj_dirs, axis=1, keepdims=True)

        sw1_vals = []
        for v in proj_dirs:
            proj_a = np.sort(samples_a @ v)
            proj_b = np.sort(samples_b @ v)
            w1 = stats.wasserstein_distance(proj_a, proj_b)
            sw1_vals.append(float(w1))
        return float(np.mean(sw1_vals)), sw1_vals

    @staticmethod
    def compute_marginal_wasserstein_1(
        samples_a: np.ndarray,
        samples_b: np.ndarray
    ) -> Tuple[float, List[float]]:
        """Computes coordinate-wise 1D Wasserstein-1 distances across all d dimensions."""
        d = samples_a.shape[1]
        w1_dims = []
        for j in range(d):
            w1_j = stats.wasserstein_distance(np.sort(samples_a[:, j]), np.sort(samples_b[:, j]))
            w1_dims.append(float(w1_j))
        return float(np.mean(w1_dims)), w1_dims
