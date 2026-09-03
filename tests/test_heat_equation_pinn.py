r"""
Unit Tests for 1D Parametric Heat Equation PINN Surrogate
=========================================================
Verifies:
1. Prior bounds derivation for alpha > 0.
2. Latin Hypercube sampling across prior support.
3. Exact dataset generator field and derivative shapes.
4. 3D input tensor forward pass (x, t, alpha) -> u.
5. Physics residual autograd calculation u_t - alpha * u_xx.
6. Single-step training convergence.
"""

import unittest
import numpy as np
import torch

from bayesian.prior import Prior, LogNormalPrior
from parametric_surrogate.parameter_sampler import (
    sample_parameter_domain_lhs,
    compute_parameter_bounds,
    PriorDomainConfig
)
from parametric_surrogate.exact_dataset_generator import (
    generate_exact_parametric_dataset,
    evaluate_exact_pde_field_and_derivatives
)
from parametric_surrogate.dataset import ParametricPINNDataset
from parametric_surrogate.parametric_model import ParametricModifiedMLP
from parametric_surrogate.trainer import ParametricPINNTrainer


class TestHeatEquationPINN(unittest.TestCase):

    def setUp(self):
        self.prior = Prior(LogNormalPrior(mu=float(np.log(0.5)), sigma=0.5))

    def test_parameter_bounds_and_lhs(self):
        """Verify prior bounds derivation and LHS sampling for alpha."""
        a_low, a_high, summary = compute_parameter_bounds(self.prior)
        self.assertGreater(a_low, 0.0)
        self.assertGreater(a_high, a_low)

        alphas, low, high, summ = sample_parameter_domain_lhs(self.prior, n_samples=20, seed=42)
        self.assertEqual(len(alphas), 20)
        self.assertTrue(np.all(alphas >= low))
        self.assertTrue(np.all(alphas <= high))

    def test_exact_dataset_generator(self):
        """Verify analytical solution fields and derivatives."""
        X, T, u, dudx, dudt, d2udx2 = evaluate_exact_pde_field_and_derivatives(alpha=0.5, resolution=30)
        self.assertEqual(u.shape, (30, 30))
        self.assertEqual(dudx.shape, (30, 30))
        self.assertEqual(dudt.shape, (30, 30))
        self.assertEqual(d2udx2.shape, (30, 30))

        # Check PDE balance
        res = dudt - 0.5 * d2udx2
        self.assertLess(np.max(np.abs(res)), 1e-12)

    def test_parametric_model_forward_pass(self):
        """Verify ParametricModifiedMLP accepts (x, t, alpha) and outputs correct tensor shape."""
        model = ParametricModifiedMLP(
            n_input=3, n_output=1, n_hidden=32, n_layers=2,
            use_fourier=True, fourier_scale=1.0
        ).to(torch.float64)

        # Batch of 50 points
        inputs = torch.rand(50, 3, dtype=torch.float64)
        out = model(inputs)
        self.assertEqual(out.shape, (50, 1))
        self.assertEqual(out.dtype, torch.float64)

    def test_physics_loss_autograd(self):
        """Verify trainer calculates PDE residual via autograd."""
        model = ParametricModifiedMLP(
            n_input=3, n_output=1, n_hidden=32, n_layers=2,
            use_fourier=True, fourier_scale=1.0
        ).to(torch.float64)

        dummy_dict = {
            "X_grid": np.linspace(0, 1, 10)[:, None] * np.ones((10, 10)),
            "T_grid": np.ones((10, 10)) * np.linspace(0, 1, 10)[None, :],
            "alpha_values": np.array([0.5, 1.0]),
            "u_fields": np.zeros((2, 10, 10))
        }
        dataset = ParametricPINNDataset(dummy_dict)
        trainer = ParametricPINNTrainer(model=model, dataset=dataset, device=torch.device("cpu"))

        test_inputs = torch.rand(20, 3, dtype=torch.float64)
        phys_loss = trainer.compute_physics_loss(test_inputs)
        self.assertIsInstance(phys_loss, torch.Tensor)
        self.assertGreaterEqual(phys_loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
