r"""
Bayesian Package Initialization (1D Heat Equation)
===================================================
Implements the Parallel Two-Chain Bayesian Inverse Problem Framework:
- Forward Operators (Exact Analytical & Parametric PINN Surrogate)
- Observation Operator H(u)
- Flexible Prior Framework p(\alpha)
- Gaussian Likelihood p(y \mid \alpha)
- Unnormalized Posterior p(\alpha \mid y)
- Single and Parallel Two-Chain Metropolis-Hastings MCMC Samplers
- Observational Distance Diagnostic d_t = |\alpha_E^{(t)} - \alpha_A^{(t)}|
- Post-Sampling Statistical Analysis & Diagnostics
- Dynamic Visualizations and Plots
"""

from .forward_operator import (
    BaseForwardOperator,
    ExactForwardOperator,
    ParametricPINNForwardOperator,
    ParametricModifiedMLP,
    ParametricFourierFeatures,
    ForwardSolverOutput
)
from .observation_operator import ObservationOperator
from .prior import (
    BaseParameterPrior,
    Prior,
    NormalPrior,
    LogNormalPrior,
    GammaPrior,
    UniformPrior,
    TruncatedNormalPrior
)
from .likelihood import GaussianLikelihood
from .posterior import Posterior
from .proposal import BaseProposal, GaussianRandomWalkProposal
from .chain import ChainRecord, MCMCChain
from .metropolis_hastings import MetropolisHastingsSampler
from .two_chain_sampler import (
    TwoChainIterationRecord,
    TwoChainMCMCResult,
    TwoChainSampler
)
from .two_chain_analysis import analyze_two_chain_experiment
from .two_chain_plots import (
    plot_alpha_trajectories,
    plot_distance_trajectory,
    plot_posterior_comparison,
    plot_master_dashboard,
    generate_all_two_chain_plots
)
from .animation import generate_two_chain_animation
from .diagnostics import (
    analyze_chain,
    compute_autocorrelation,
    compute_effective_sample_size,
    compute_credible_interval,
    plot_mcmc_diagnostics
)
from .sampler_results import MCMCResult, export_mcmc_results

__all__ = [
    "BaseForwardOperator",
    "ExactForwardOperator",
    "ParametricPINNForwardOperator",
    "ParametricModifiedMLP",
    "ParametricFourierFeatures",
    "ForwardSolverOutput",
    "ObservationOperator",
    "BaseParameterPrior",
    "Prior",
    "NormalPrior",
    "LogNormalPrior",
    "GammaPrior",
    "UniformPrior",
    "TruncatedNormalPrior",
    "GaussianLikelihood",
    "Posterior",
    "BaseProposal",
    "GaussianRandomWalkProposal",
    "ChainRecord",
    "MCMCChain",
    "MetropolisHastingsSampler",
    "TwoChainIterationRecord",
    "TwoChainMCMCResult",
    "TwoChainSampler",
    "analyze_two_chain_experiment",
    "plot_alpha_trajectories",
    "plot_distance_trajectory",
    "plot_posterior_comparison",
    "plot_master_dashboard",
    "generate_all_two_chain_plots",
    "generate_two_chain_animation",
    "analyze_chain",
    "compute_autocorrelation",
    "compute_effective_sample_size",
    "compute_credible_interval",
    "plot_mcmc_diagnostics",
    "MCMCResult",
    "export_mcmc_results"
]
