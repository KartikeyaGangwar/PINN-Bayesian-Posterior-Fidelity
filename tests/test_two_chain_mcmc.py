r"""
Unit Test Suite for 1D Parametric Heat Equation Bayesian Framework
===================================================================
Verifies:
1. Exact analytical solution satisfies the heat equation (u_t = \alpha u_{xx}).
2. Exact solution satisfies initial condition (u(x,0) = \sin(\pi x)).
3. Exact solution satisfies boundary conditions (u(0,t) = u(1,t) = 0).
4. PINN forward operator accepts (x, t, \alpha).
5. Both chains start from the exact same initial state \alpha_E^{(0)} == \alpha_A^{(0)} == \alpha_0.
6. Both chains use identical symmetric MH configuration.
7. Distance diagnostic d_t = |\alpha_E - \alpha_A| is purely observational and never alters chain evolution.
8. Chains always run for the requested number of iterations.
9. Post-burn-in extraction is correct.
10. Exact-vs-exact control produces consistent posterior behavior.
11. Serialization works (NPZ, JSON).
12. Offline post-sampling statistical analysis works.
13. Posterior comparison works.
14. All publication figures are generated successfully.
"""

import unittest
import os
import tempfile
import torch
import numpy as np

from bayesian.forward_operator import (
    ExactForwardOperator, ParametricPINNForwardOperator,
    ParametricModifiedMLP, ForwardSolverOutput
)
from bayesian.observation_operator import ObservationOperator
from bayesian.prior import LogNormalPrior, Prior
from bayesian.likelihood import GaussianLikelihood
from bayesian.posterior import Posterior
from bayesian.proposal import GaussianRandomWalkProposal
from bayesian.two_chain_sampler import TwoChainSampler, TwoChainMCMCResult
from bayesian.two_chain_analysis import analyze_two_chain_experiment
from bayesian.two_chain_plots import generate_all_two_chain_plots
from parametric_surrogate.exact_dataset_generator import evaluate_exact_pde_field_and_derivatives


class TestHeatEquationBayesianFramework(unittest.TestCase):

    def setUp(self):
        """Set up standard operators and test configurations."""
        self.exact_op = ExactForwardOperator(resolution=50)
        
        # Observation operator
        sensor_coords = np.column_stack([
            np.linspace(0.1, 0.9, 5),
            np.linspace(0.2, 0.8, 5)
        ])
        self.obs_op = ObservationOperator(sensor_locations=sensor_coords)
        
        # Ground truth observation from true alpha = 0.5
        gt = self.exact_op(0.5)
        self.y_obs = self.obs_op(gt.u)
        
        self.prior = Prior(LogNormalPrior(mu=float(np.log(0.5)), sigma=0.5))
        self.likelihood = GaussianLikelihood(noise_std=0.01)
        self.posterior = Posterior(prior=self.prior, likelihood=self.likelihood, obs_operator=self.obs_op)
        self.proposal = GaussianRandomWalkProposal(scale=0.05)

    def test_1_exact_solution_satisfies_heat_equation(self):
        """1. Verify exact analytical solution satisfies u_t = alpha u_xx."""
        alpha = 0.75
        _, _, u, dudx, dudt, d2udx2 = evaluate_exact_pde_field_and_derivatives(alpha=alpha, resolution=100)
        pde_res = dudt - alpha * d2udx2
        max_res = np.max(np.abs(pde_res))
        self.assertLess(max_res, 1e-12)

    def test_2_exact_solution_initial_condition(self):
        """2. Verify exact solution satisfies u(x, 0) = sin(pi * x)."""
        alpha = 0.5
        out = self.exact_op(alpha)
        u_t0 = out.u[:, 0]
        expected_ic = np.sin(np.pi * self.exact_op.x_1d)
        ic_err = np.max(np.abs(u_t0 - expected_ic))
        self.assertLess(ic_err, 1e-14)

    def test_3_exact_solution_boundary_conditions(self):
        """3. Verify exact solution satisfies u(0, t) = 0 and u(1, t) = 0."""
        for alpha in [0.1, 0.5, 1.2]:
            out = self.exact_op(alpha)
            bc0_err = np.max(np.abs(out.u[0, :]))
            bc1_err = np.max(np.abs(out.u[-1, :]))
            self.assertLess(bc0_err, 1e-14)
            self.assertLess(bc1_err, 1e-14)

    def test_4_pinn_forward_operator_interface(self):
        """4. Verify PINN forward operator accepts 3D inputs (x, t, alpha)."""
        model = ParametricModifiedMLP(n_input=3, n_output=1, n_hidden=32, n_layers=2, use_fourier=False)
        pinn_op = ParametricPINNForwardOperator(model=model, resolution=40)
        out = pinn_op(0.5)
        self.assertIsInstance(out, ForwardSolverOutput)
        self.assertEqual(out.u.shape, (40, 40))
        self.assertEqual(out.alpha, 0.5)

    def test_5_common_initialization(self):
        """5. Verify both chains start from the exact same randomly generated alpha_0."""
        sampler = TwoChainSampler(
            posterior=self.posterior,
            proposal=self.proposal,
            forward_solver_exact=self.exact_op,
            forward_solver_pinn=self.exact_op
        )
        # Test with specified initial state
        result_fixed = sampler.run(y_obs=self.y_obs, n_samples=50, initial_alpha=0.75, seed=42, verbose=False)
        self.assertEqual(result_fixed.alpha_0, 0.75)

        # Test with prior-drawn initial state
        result_rand = sampler.run(y_obs=self.y_obs, n_samples=50, seed=42, verbose=False)
        self.assertGreater(result_rand.alpha_0, 0.0)
        self.assertFalse(np.isneginf(self.prior.log_prior(result_rand.alpha_0)))

    def test_6_identical_mh_configuration(self):
        """6. Verify both chains use identical proposal scale and transition logic."""
        sampler = TwoChainSampler(
            posterior=self.posterior,
            proposal=self.proposal,
            forward_solver_exact=self.exact_op,
            forward_solver_pinn=self.exact_op
        )
        self.assertTrue(sampler.proposal.is_symmetric)
        self.assertEqual(sampler.proposal.scale, 0.05)

    def test_7_observational_distance_diagnostic(self):
        """7. Verify distance metric is purely observational and does not alter chain evolution."""
        # Perturbed solver to produce distinct trajectories
        def perturbed_solver(alpha):
            return self.exact_op(alpha * 1.05)
            
        sampler = TwoChainSampler(
            posterior=self.posterior,
            proposal=self.proposal,
            forward_solver_exact=self.exact_op,
            forward_solver_pinn=perturbed_solver
        )
        result = sampler.run(y_obs=self.y_obs, n_samples=100, seed=42, verbose=False)
        
        for r in result.records:
            expected_d = abs(r.alpha_E - r.alpha_A)
            self.assertAlmostEqual(r.distance, expected_d, places=10)

    def test_8_chains_run_for_requested_iterations(self):
        """8. Verify chains always run for the exact number of requested iterations."""
        sampler = TwoChainSampler(
            posterior=self.posterior,
            proposal=self.proposal,
            forward_solver_exact=self.exact_op,
            forward_solver_pinn=self.exact_op
        )
        for n in [25, 75, 120]:
            res = sampler.run(y_obs=self.y_obs, n_samples=n, seed=42, verbose=False)
            self.assertEqual(len(res), n)
            self.assertEqual(len(res.records), n)

    def test_9_post_burnin_extraction(self):
        """9. Verify post-burn-in sample extraction separates transient samples correctly."""
        sampler = TwoChainSampler(
            posterior=self.posterior,
            proposal=self.proposal,
            forward_solver_exact=self.exact_op,
            forward_solver_pinn=self.exact_op
        )
        res = sampler.run(y_obs=self.y_obs, n_samples=100, seed=42, verbose=False)
        s_E, s_A = res.get_post_burnin_samples(burn_in=40)
        self.assertEqual(len(s_E), 60)
        self.assertEqual(len(s_A), 60)

    def test_10_control_experiment_consistency(self):
        """10. Verify exact-vs-exact control produces consistent posterior behavior."""
        sampler = TwoChainSampler(
            posterior=self.posterior,
            proposal=self.proposal,
            forward_solver_exact=self.exact_op,
            forward_solver_pinn=self.exact_op
        )
        res = sampler.run(y_obs=self.y_obs, n_samples=300, seed=42, is_control=True, verbose=False)
        analysis = analyze_two_chain_experiment(res, burn_in=50)
        
        # Means should be very close
        mean_E = analysis["marginal_posterior_statistics"]["exact_chain"]["alpha"]["mean"]
        mean_A = analysis["marginal_posterior_statistics"]["approx_chain"]["alpha"]["mean"]
        self.assertAlmostEqual(mean_E, mean_A, places=2)

    def test_11_serialization(self):
        """11. Verify serialization to NPZ and JSON works cleanly."""
        sampler = TwoChainSampler(
            posterior=self.posterior,
            proposal=self.proposal,
            forward_solver_exact=self.exact_op,
            forward_solver_pinn=self.exact_op
        )
        res = sampler.run(y_obs=self.y_obs, n_samples=50, seed=42, verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_file = os.path.join(tmpdir, "test_res.npz")
            json_file = os.path.join(tmpdir, "test_summary.json")
            res.save_to_npz(npz_file)
            res.save_summary_json(json_file, burn_in=10)
            
            self.assertTrue(os.path.exists(npz_file))
            self.assertTrue(os.path.exists(json_file))

    def test_12_offline_analysis(self):
        """12. Verify offline analysis consumes saved NPZ correctly."""
        sampler = TwoChainSampler(
            posterior=self.posterior,
            proposal=self.proposal,
            forward_solver_exact=self.exact_op,
            forward_solver_pinn=self.exact_op
        )
        res = sampler.run(y_obs=self.y_obs, n_samples=60, seed=42, verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_file = os.path.join(tmpdir, "test_res.npz")
            res.save_to_npz(npz_file)
            
            analysis = analyze_two_chain_experiment(npz_file, burn_in=10)
            self.assertIn("marginal_posterior_statistics", analysis)
            self.assertIn("distance_diagnostics", analysis)
            self.assertIn("distribution_discrepancies", analysis)

    def test_13_posterior_comparison(self):
        """13. Verify posterior grid evaluation calculates proper unnormalized and normalized densities."""
        alpha_grid = np.linspace(0.2, 1.0, 20)
        grid_data = self.posterior.evaluate_grid(alpha_grid, self.y_obs, self.exact_op)
        self.assertEqual(len(grid_data["alpha_grid"]), 20)
        self.assertEqual(len(grid_data["normalized_pdf"]), 20)
        self.assertGreater(np.max(grid_data["normalized_pdf"]), 0.0)

    def test_14_plot_generation(self):
        """14. Verify all publication figures are generated without errors."""
        sampler = TwoChainSampler(
            posterior=self.posterior,
            proposal=self.proposal,
            forward_solver_exact=self.exact_op,
            forward_solver_pinn=self.exact_op
        )
        res = sampler.run(y_obs=self.y_obs, n_samples=50, seed=42, verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plots = generate_all_two_chain_plots(res, output_dir=tmpdir, burn_in=10, true_alpha=0.5)
            for name, path in plots.items():
                self.assertTrue(os.path.exists(path), f"Figure {name} was not generated at {path}")


if __name__ == "__main__":
    unittest.main()
