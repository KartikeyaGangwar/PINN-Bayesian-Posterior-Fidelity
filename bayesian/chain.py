r"""
MCMC Chain Dataclass and Container Module (1D Heat Equation)
============================================================
Description:
    Encapsulates iteration records and trajectory history for a Markov Chain Monte Carlo sampler.
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Union
from datetime import datetime
import numpy as np


@dataclass
class ChainRecord:
    r"""
    Dataclass holding all state variables and diagnostics for a single MCMC step.
    
    Fields:
        iteration: Current chain iteration k
        alpha: Parameter scalar \alpha^{(k)}
        accepted: Boolean flag indicating whether the proposal was accepted
        log_prior: Log prior density \ln \pi_{\text{prior}}(\alpha^{(k)})
        log_likelihood: Log likelihood \ln p(\boldsymbol{y} \mid \alpha^{(k)})
        log_posterior: Unnormalized log posterior \ln \pi_{\text{post}}(\alpha^{(k)} \mid \boldsymbol{y})
        acceptance_prob: Calculated acceptance probability \alpha(\alpha^{(k-1)}, \alpha')
        random_num: Drawn uniform random number u \sim \mathcal{U}(0,1)
        running_acceptance_rate: Cumulative acceptance ratio up to iteration k
        timestamp: ISO timestamp of evaluation
    """
    iteration: int
    alpha: float
    accepted: bool
    log_prior: float
    log_likelihood: float
    log_posterior: float
    acceptance_prob: float
    random_num: float
    running_acceptance_rate: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MCMCChain:
    """
    Container class for managing MCMC trajectory history and sample extractions.
    """
    def __init__(self):
        self.records: List[ChainRecord] = []

    def append(self, record: ChainRecord):
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def get_alpha_array(self, burn_in: int = 0) -> np.ndarray:
        """Returns array of sampled alpha values after burn-in."""
        return np.array([r.alpha for r in self.records[burn_in:]], dtype=np.float64)

    def get_log_posterior_array(self, burn_in: int = 0) -> np.ndarray:
        """Returns array of unnormalized log posterior values after burn-in."""
        return np.array([r.log_posterior for r in self.records[burn_in:]], dtype=np.float64)

    def get_log_likelihood_array(self, burn_in: int = 0) -> np.ndarray:
        """Returns array of log likelihood values after burn-in."""
        return np.array([r.log_likelihood for r in self.records[burn_in:]], dtype=np.float64)

    @property
    def acceptance_rate(self) -> float:
        """Cumulative acceptance rate across all recorded steps."""
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.accepted) / len(self.records)
