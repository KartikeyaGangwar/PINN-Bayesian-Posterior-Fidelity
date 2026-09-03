r"""
Parallel Two-Chain Metropolis-Hastings Sampler Module (1D Heat Equation)
========================================================================
Mathematical Reference:
    Metropolis, Rosenbluth, Rosenbluth, Teller, Teller (1953) J. Chem. Phys.
    Hastings (1970) Biometrika
    Robert & Casella (2004) "Monte Carlo Statistical Methods", Springer, Chapter 7
    Stuart (2010) "Inverse problems: A Bayesian perspective", Acta Numerica

Core Scientific Experiment:
---------------------------
- Target Posteriors:
    * Exact Target: \pi_E(\alpha \mid \boldsymbol{y}) \propto p(\boldsymbol{y} \mid \mathcal{F}_{\text{exact}}(\alpha)) \cdot \pi_{\text{prior}}(\alpha)
    * Approximate Target: \pi_A(\alpha \mid \boldsymbol{y}) \propto p(\boldsymbol{y} \mid \mathcal{F}_{\text{PINN}}(\alpha)) \cdot \pi_{\text{prior}}(\alpha)
- Common Random Initial State:
    \alpha_E^{(0)} = \alpha_A^{(0)} = \alpha_0 \sim \pi_{\text{prior}}(\alpha)
- Identical MH Transition Algorithm:
    Both chains use identical symmetric proposal distribution q, proposal scale, acceptance rule,
    prior, likelihood formulation, and iteration count.
- Observational Diagnostic Metric:
    d_t = |\alpha_E^{(t)} - \alpha_A^{(t)}|
    (Recorded purely as an observational metric; does NOT control or alter chain sampling)
"""

import numpy as np
import time
import math
import json
import csv
import os
from dataclasses import dataclass, asdict
from typing import Tuple, Callable, Optional, Dict, Any, List, Union
from datetime import datetime

from .posterior import Posterior
from .proposal import BaseProposal, GaussianRandomWalkProposal


# =============================================================================
# ITERATION RECORD CONTAINER
# =============================================================================

@dataclass
class TwoChainIterationRecord:
    r"""
    Container storing exact and approximate chain states and observational metrics at iteration t.
    """
    iteration: int
    alpha_E: float
    alpha_A: float
    accepted_E: bool
    accepted_A: bool
    log_prior_E: float
    log_prior_A: float
    log_likelihood_E: float
    log_likelihood_A: float
    log_posterior_E: float
    log_posterior_A: float
    acceptance_prob_E: float
    acceptance_prob_A: float
    random_num_E: float
    random_num_A: float
    running_acc_rate_E: float
    running_acc_rate_A: float
    distance: float                  # Observational distance d_t = |\alpha_E - \alpha_A|
    log_posterior_diff: float        # |\ln \pi_E(\alpha_E) - \ln \pi_A(\alpha_A)|
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# EXPERIMENT RESULT CONTAINER
# =============================================================================

@dataclass
class TwoChainMCMCResult:
    r"""
    Complete result container for the parallel two-chain observational experiment.
    Preserves full trajectory history and diagnostic arrays.
    """
    alpha_0: float
    records: List[TwoChainIterationRecord]
    n_samples: int
    runtime_seconds: float
    proposal_scale: float
    noise_std: float
    seed: int
    experiment_type: str = "research"  # 'research' (Exact vs PINN) or 'control' (Exact vs Exact)

    def __len__(self) -> int:
        return len(self.records)

    # 1. Trajectory Extraction Helpers
    def get_alpha_E_array(self) -> np.ndarray:
        return np.array([r.alpha_E for r in self.records], dtype=np.float64)

    def get_alpha_A_array(self) -> np.ndarray:
        return np.array([r.alpha_A for r in self.records], dtype=np.float64)

    def get_distance_array(self) -> np.ndarray:
        return np.array([r.distance for r in self.records], dtype=np.float64)

    def get_log_posterior_E_array(self) -> np.ndarray:
        return np.array([r.log_posterior_E for r in self.records], dtype=np.float64)

    def get_log_posterior_A_array(self) -> np.ndarray:
        return np.array([r.log_posterior_A for r in self.records], dtype=np.float64)

    def get_log_likelihood_E_array(self) -> np.ndarray:
        return np.array([r.log_likelihood_E for r in self.records], dtype=np.float64)

    def get_log_likelihood_A_array(self) -> np.ndarray:
        return np.array([r.log_likelihood_A for r in self.records], dtype=np.float64)

    def get_accepted_E_array(self) -> np.ndarray:
        return np.array([r.accepted_E for r in self.records], dtype=bool)

    def get_accepted_A_array(self) -> np.ndarray:
        return np.array([r.accepted_A for r in self.records], dtype=bool)

    # 2. Acceptance Rates
    @property
    def final_acceptance_rate_E(self) -> float:
        if not self.records:
            return 0.0
        return self.records[-1].running_acc_rate_E

    @property
    def final_acceptance_rate_A(self) -> float:
        if not self.records:
            return 0.0
        return self.records[-1].running_acc_rate_A

    # 3. Post-Burn-in Stationary Extraction
    def get_post_burnin_samples(self, burn_in: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Returns stationary post-burn-in sample arrays (S_E, S_A)."""
        idx = min(burn_in, max(0, len(self.records) - 1))
        a_E = self.get_alpha_E_array()[idx:]
        a_A = self.get_alpha_A_array()[idx:]
        return a_E, a_A

    # 4. Serialization Helpers
    def save_to_npz(self, filepath: str):
        """Serializes numerical trajectories to a compressed NPZ archive."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            alpha_E=self.get_alpha_E_array(),
            alpha_A=self.get_alpha_A_array(),
            distance=self.get_distance_array(),
            log_posterior_E=self.get_log_posterior_E_array(),
            log_posterior_A=self.get_log_posterior_A_array(),
            log_likelihood_E=self.get_log_likelihood_E_array(),
            log_likelihood_A=self.get_log_likelihood_A_array(),
            accepted_E=self.get_accepted_E_array(),
            accepted_A=self.get_accepted_A_array(),
            alpha_0=self.alpha_0,
            n_samples=self.n_samples,
            runtime_seconds=self.runtime_seconds,
            proposal_scale=self.proposal_scale,
            noise_std=self.noise_std,
            seed=self.seed,
            experiment_type=self.experiment_type
        )

    def save_summary_json(self, filepath: str, burn_in: int = 1000):
        """Exports statistical summary to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        a_E_post, a_A_post = self.get_post_burnin_samples(burn_in)
        dist_all = self.get_distance_array()
        dist_post = dist_all[min(burn_in, len(dist_all) - 1):]

        summary = {
            "experiment_type": self.experiment_type,
            "n_samples": self.n_samples,
            "burn_in": burn_in,
            "runtime_seconds": self.runtime_seconds,
            "alpha_0": float(self.alpha_0),
            "proposal_scale": self.proposal_scale,
            "noise_std": self.noise_std,
            "acceptance_rate_E": float(self.final_acceptance_rate_E),
            "acceptance_rate_A": float(self.final_acceptance_rate_A),
            "mean_distance_full": float(np.mean(dist_all)),
            "max_distance_full": float(np.max(dist_all)),
            "final_distance": float(dist_all[-1]),
            "mean_distance_post_burnin": float(np.mean(dist_post)),
            "posterior_E": {
                "mean_alpha": float(np.mean(a_E_post)),
                "std_alpha": float(np.std(a_E_post)),
                "median_alpha": float(np.median(a_E_post)),
                "ci_95": [float(np.percentile(a_E_post, 2.5)), float(np.percentile(a_E_post, 97.5))]
            },
            "posterior_A": {
                "mean_alpha": float(np.mean(a_A_post)),
                "std_alpha": float(np.std(a_A_post)),
                "median_alpha": float(np.median(a_A_post)),
                "ci_95": [float(np.percentile(a_A_post, 2.5)), float(np.percentile(a_A_post, 97.5))]
            }
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)


# =============================================================================
# PARALLEL TWO-CHAIN SAMPLER
# =============================================================================

class TwoChainSampler:
    r"""
    Parallel Two-Chain Metropolis-Hastings Sampler.
    Executes Chain E (Exact Solver) and Chain A (PINN Solver) simultaneously.
    """
    def __init__(
        self,
        posterior: Posterior,
        proposal: Optional[BaseProposal] = None,
        forward_solver_exact: Optional[Callable] = None,
        forward_solver_pinn: Optional[Callable] = None
    ):
        self.posterior = posterior
        self.proposal = proposal if proposal is not None else GaussianRandomWalkProposal(scale=0.05)
        self.forward_solver_exact = forward_solver_exact
        self.forward_solver_pinn = forward_solver_pinn

    def run(
        self,
        y_obs: np.ndarray,
        n_samples: int = 5000,
        initial_alpha: Optional[float] = None,
        seed: Optional[int] = 42,
        is_control: bool = False,
        verbose: bool = True,
        print_interval: int = 1000
    ) -> TwoChainMCMCResult:
        r"""
        Runs the parallel two-chain observational experiment.
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. Common Random Initialization from Prior
        if initial_alpha is not None:
            alpha_0 = float(initial_alpha)
        else:
            alpha_0 = float(self.posterior.prior.sample())
            # Ensure valid support
            while alpha_0 <= 0 or np.isneginf(self.posterior.prior.log_prior(alpha_0)):
                alpha_0 = float(self.posterior.prior.sample())

        alpha_E = float(alpha_0)
        alpha_A = float(alpha_0)

        # Select forward solver for Chain A (Exact if control, PINN if research)
        solver_A = self.forward_solver_exact if is_control else self.forward_solver_pinn
        if solver_A is None:
            solver_A = self.forward_solver_exact

        # 2. Initial State Evaluation
        comp_E = self.posterior.evaluate_components(alpha_E, y_obs, self.forward_solver_exact)
        comp_A = self.posterior.evaluate_components(alpha_A, y_obs, solver_A)

        records: List[TwoChainIterationRecord] = []
        n_accepted_E = 0
        n_accepted_A = 0

        t_start = time.time()
        experiment_type = "control (Exact vs Exact)" if is_control else "research (Exact vs PINN)"

        if verbose:
            print("=" * 80)
            print(f"PARALLEL TWO-CHAIN METROPOLIS-HASTINGS: [{experiment_type.upper()}]")
            print("=" * 80)
            print(f"  Iterations      : {n_samples:,}")
            print(f"  Common Start a0 : {alpha_0:.6f}")
            print(f"  Proposal Scale  : {self.proposal.scale:.4f}")
            print(f"  Measurement Dim : {y_obs.size} sensors")
            print("=" * 80)

        for t in range(1, n_samples + 1):
            # -------------------------------------------------------------
            # STEP A: EXACT CHAIN (Chain E)
            # -------------------------------------------------------------
            alpha_prop_E = self.proposal.propose(alpha_E)
            
            if alpha_prop_E <= 0.0:
                acc_prob_E = 0.0
                comp_prop_E = {"log_prior": -np.inf, "log_likelihood": -np.inf, "log_posterior": -np.inf}
            else:
                comp_prop_E = self.posterior.evaluate_components(alpha_prop_E, y_obs, self.forward_solver_exact)
                if np.isneginf(comp_prop_E["log_posterior"]):
                    acc_prob_E = 0.0
                else:
                    log_alpha_ratio_E = comp_prop_E["log_posterior"] - comp_E["log_posterior"]
                    if not self.proposal.is_symmetric:
                        log_alpha_ratio_E += self.proposal.log_q(alpha_prop_E, alpha_E) - self.proposal.log_q(alpha_E, alpha_prop_E)
                    acc_prob_E = min(1.0, math.exp(min(0.0, log_alpha_ratio_E)))

            u_rand_E = np.random.uniform(0.0, 1.0)
            accepted_E = bool(u_rand_E < acc_prob_E)

            if accepted_E:
                alpha_E = alpha_prop_E
                comp_E = comp_prop_E
                n_accepted_E += 1

            # -------------------------------------------------------------
            # STEP B: APPROXIMATE / SECOND CHAIN (Chain A)
            # -------------------------------------------------------------
            alpha_prop_A = self.proposal.propose(alpha_A)

            if alpha_prop_A <= 0.0:
                acc_prob_A = 0.0
                comp_prop_A = {"log_prior": -np.inf, "log_likelihood": -np.inf, "log_posterior": -np.inf}
            else:
                comp_prop_A = self.posterior.evaluate_components(alpha_prop_A, y_obs, solver_A)
                if np.isneginf(comp_prop_A["log_posterior"]):
                    acc_prob_A = 0.0
                else:
                    log_alpha_ratio_A = comp_prop_A["log_posterior"] - comp_A["log_posterior"]
                    if not self.proposal.is_symmetric:
                        log_alpha_ratio_A += self.proposal.log_q(alpha_prop_A, alpha_A) - self.proposal.log_q(alpha_A, alpha_prop_A)
                    acc_prob_A = min(1.0, math.exp(min(0.0, log_alpha_ratio_A)))

            u_rand_A = np.random.uniform(0.0, 1.0)
            accepted_A = bool(u_rand_A < acc_prob_A)

            if accepted_A:
                alpha_A = alpha_prop_A
                comp_A = comp_prop_A
                n_accepted_A += 1

            # -------------------------------------------------------------
            # STEP C: OBSERVATIONAL DIAGNOSTICS
            # -------------------------------------------------------------
            dist_t = float(abs(alpha_E - alpha_A))
            lp_diff = float(abs(comp_E["log_posterior"] - comp_A["log_posterior"]))

            record = TwoChainIterationRecord(
                iteration=t,
                alpha_E=float(alpha_E),
                alpha_A=float(alpha_A),
                accepted_E=accepted_E,
                accepted_A=accepted_A,
                log_prior_E=float(comp_E["log_prior"]),
                log_prior_A=float(comp_A["log_prior"]),
                log_likelihood_E=float(comp_E["log_likelihood"]),
                log_likelihood_A=float(comp_A["log_likelihood"]),
                log_posterior_E=float(comp_E["log_posterior"]),
                log_posterior_A=float(comp_A["log_posterior"]),
                acceptance_prob_E=float(acc_prob_E),
                acceptance_prob_A=float(acc_prob_A),
                random_num_E=float(u_rand_E),
                random_num_A=float(u_rand_A),
                running_acc_rate_E=float(n_accepted_E / t),
                running_acc_rate_A=float(n_accepted_A / t),
                distance=dist_t,
                log_posterior_diff=lp_diff
            )
            records.append(record)

            if verbose and (t % print_interval == 0 or t == n_samples):
                print(
                    f"  Iter {t:6d}/{n_samples} | "
                    f"a_E: {alpha_E:.4f} | a_A: {alpha_A:.4f} | "
                    f"d_t: {dist_t:.4e} | "
                    f"Acc_E: {n_accepted_E / t:.1%} | Acc_A: {n_accepted_A / t:.1%}",
                    flush=True
                )

        runtime = time.time() - t_start
        if verbose:
            print("-" * 80)
            print(f"  [COMPLETED] Runtime: {runtime:.2f}s | Final d_t: {records[-1].distance:.4e}")
            print("=" * 80)

        return TwoChainMCMCResult(
            alpha_0=alpha_0,
            records=records,
            n_samples=n_samples,
            runtime_seconds=runtime,
            proposal_scale=self.proposal.scale,
            noise_std=self.posterior.likelihood.noise_std,
            seed=seed if seed is not None else 0,
            experiment_type=experiment_type
        )
