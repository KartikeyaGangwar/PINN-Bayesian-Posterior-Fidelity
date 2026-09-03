"""
Unit tests for research metrics module (experiments/metrics.py)
"""

import unittest
import numpy as np
from experiments.metrics import (
    compute_global_forward_metrics,
    evaluate_parameter_resolved_error,
    compute_weighted_forward_errors,
    compute_bayesian_posterior_metrics,
    compute_bayesian_fidelity_ratio
)


class TestResearchMetrics(unittest.TestCase):
    def test_global_forward_metrics(self):
        e_dict = {
            "alpha_grid": np.array([0.2, 0.5, 0.8]),
            "e_alpha_field": np.array([0.02, 0.03, 0.04]),
            "e_alpha_max_abs": np.array([0.01, 0.02, 0.03]),
            "e_alpha_obs": np.array([0.015, 0.025, 0.035])
        }
        res = compute_global_forward_metrics(e_dict)
        self.assertAlmostEqual(res["mean_rel_l2"], 0.03)
        self.assertAlmostEqual(res["median_rel_l2"], 0.03)
        self.assertAlmostEqual(res["max_rel_l2"], 0.04)

    def test_weighted_forward_errors(self):
        alphas = np.linspace(0.1, 1.0, 100)
        e_alpha = 0.02 + 0.01 * alphas
        prior = np.ones_like(alphas) / (1.0 - 0.1)
        # Gaussian-like posterior centered at 0.5
        post = np.exp(-0.5 * ((alphas - 0.5) / 0.05) ** 2)
        
        res = compute_weighted_forward_errors(alphas, e_alpha, prior, post)
        self.assertIn("E_global", res)
        self.assertIn("E_prior", res)
        self.assertIn("E_posterior", res)
        self.assertIn("RMS_posterior", res)
        self.assertGreater(res["E_posterior"], 0.0)

    def test_bayesian_posterior_metrics(self):
        s_exact = np.random.normal(0.50, 0.01, 1000)
        s_pinn = np.random.normal(0.502, 0.01, 1000)
        
        res = compute_bayesian_posterior_metrics(s_exact, s_pinn, alpha_true=0.50)
        self.assertIn("exact", res)
        self.assertIn("pinn", res)
        self.assertIn("discrepancy", res)
        self.assertGreater(res["discrepancy"]["wasserstein_1"], 0.0)

    def test_bayesian_fidelity_ratio(self):
        bfr = compute_bayesian_fidelity_ratio(w1_research=0.002, w1_control=0.0005)
        self.assertAlmostEqual(bfr, 4.0)


if __name__ == "__main__":
    unittest.main()
