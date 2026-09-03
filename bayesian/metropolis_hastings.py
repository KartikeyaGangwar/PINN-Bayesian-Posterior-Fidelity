r"""
Metropolis-Hastings MCMC Sampler Module (1D Heat Equation)
==========================================================
Mathematical Reference:
    Metropolis, Rosenbluth, Rosenbluth, Teller, Teller (1953) J. Chem. Phys.
    Hastings (1970) Biometrika
    Robert & Casella (2004) "Monte Carlo Statistical Methods", Springer, Chapter 7
"""

import numpy as np
import time
import math
from typing import Tuple, Callable, Optional, Dict, Any, Union
from datetime import datetime

from .posterior import Posterior
from .proposal import BaseProposal
from .chain import MCMCChain, ChainRecord


class MetropolisHastingsSampler:
    r"""
    Single-Chain Metropolis-Hastings MCMC Sampler for scalar parameter \alpha.
    """
    def __init__(
        self,
        posterior: Posterior,
        proposal: BaseProposal
    ):
        self.posterior = posterior
        self.proposal = proposal

    def sample(
        self,
        alpha_init: Union[float, np.ndarray],
        y_obs: np.ndarray,
        forward_solver_fn: Callable,
        n_samples: int = 10000,
        log_interval: int = 1000,
        seed: Optional[int] = 42
    ) -> MCMCChain:
        r"""
        Executes Metropolis-Hastings MCMC sampling loop.
        """
        if seed is not None:
            np.random.seed(seed)

        chain = MCMCChain()
        start_time = time.time()

        alpha_curr = float(alpha_init[0]) if isinstance(alpha_init, (list, tuple, np.ndarray)) else float(alpha_init)
        curr_comps = self.posterior.evaluate_components(alpha_curr, y_obs, forward_solver_fn)
        
        if np.isneginf(curr_comps["log_posterior"]):
            raise ValueError(f"Initial state alpha={alpha_curr} has -inf log posterior! Choose valid initial guess.")

        n_accepted = 0

        # Record initial step
        rec0 = ChainRecord(
            iteration=0,
            alpha=alpha_curr,
            accepted=True,
            log_prior=curr_comps["log_prior"],
            log_likelihood=curr_comps["log_likelihood"],
            log_posterior=curr_comps["log_posterior"],
            acceptance_prob=1.0,
            random_num=0.0,
            running_acceptance_rate=1.0
        )
        chain.append(rec0)

        for k in range(1, n_samples + 1):
            alpha_prop = self.proposal.propose(alpha_curr)

            if alpha_prop <= 0.0:
                acc_prob = 0.0
                prop_comps = {"log_prior": -np.inf, "log_likelihood": -np.inf, "log_posterior": -np.inf}
            else:
                prop_comps = self.posterior.evaluate_components(alpha_prop, y_obs, forward_solver_fn)
                if np.isneginf(prop_comps["log_posterior"]):
                    acc_prob = 0.0
                else:
                    log_alpha_ratio = prop_comps["log_posterior"] - curr_comps["log_posterior"]
                    if not self.proposal.is_symmetric:
                        log_alpha_ratio += self.proposal.log_q(alpha_prop, alpha_curr) - self.proposal.log_q(alpha_curr, alpha_prop)
                    acc_prob = min(1.0, math.exp(min(0.0, log_alpha_ratio)))

            u = np.random.uniform(0.0, 1.0)
            accepted = bool(u < acc_prob)

            if accepted:
                alpha_curr = alpha_prop
                curr_comps = prop_comps
                n_accepted += 1

            record = ChainRecord(
                iteration=k,
                alpha=alpha_curr,
                accepted=accepted,
                log_prior=curr_comps["log_prior"],
                log_likelihood=curr_comps["log_likelihood"],
                log_posterior=curr_comps["log_posterior"],
                acceptance_prob=acc_prob,
                random_num=u,
                running_acceptance_rate=n_accepted / k
            )
            chain.append(record)

            if k % log_interval == 0 or k == n_samples:
                print(
                    f"  Iter {k:6d}/{n_samples} | alpha: {alpha_curr:.4f} | "
                    f"Acc: {n_accepted / k:.1%} | log_post: {curr_comps['log_posterior']:.2f}",
                    flush=True
                )

        return chain
