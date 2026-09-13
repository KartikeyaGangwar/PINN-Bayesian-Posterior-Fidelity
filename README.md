# Parameter-Space Error Localization and Posterior Fidelity of Physics-Informed Neural Network Forward Surrogates in Bayesian Inverse Problems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official open-source research repository and computational reproducibility suite for the research article:  
**"Parameter-Space Error Localization and Posterior Fidelity of Physics-Informed Neural Network Forward Surrogates in Bayesian Inverse Problems"**

**Author:** Kartikey Singh  
**Affiliation:** University of Delhi, Delhi 110007, India  
**E-mail:** `kartikeysingh525@protonmail.com`  
**ORCID:** [0009-0009-1973-7532](https://orcid.org/0009-0009-1973-7532)  
**Repository:** [https://github.com/KartikeyaGangwar/PINN-Bayesian-Posterior-Fidelity](https://github.com/KartikeyaGangwar/PINN-Bayesian-Posterior-Fidelity)

---

## Scientific Overview and Theoretical Foundations

In Bayesian inverse problems governed by Partial Differential Equations (PDEs), forward surrogate models—such as Physics-Informed Neural Networks (PINNs)—are frequently deployed to accelerate Markov Chain Monte Carlo (MCMC) sampling. Conventional validation practices rely almost exclusively on **uniform global forward-field error metrics** ($E_{\mathrm{global}}$), such as relative $L_2$ errors averaged uniformly across spatial, temporal, and parameter domains.

This repository provides a unified theoretical, algorithmic, and empirical framework demonstrating that:

1. **Forward Error Filtering**: Surrogate approximation error enters the Bayesian posterior strictly through sparse sensor projections $\Delta \mathcal{G}(\theta)$ and the likelihood functional $\Delta \log \mathcal{L}(\theta)$, which exponentially reweights parameter space according to observation consistency.
2. **Posterior Stability Bounds**: Under approximate forward operators on compact parameter spaces $\Theta \subset \mathbb{R}^d$, quantitative Total Variation ($d_{\mathrm{TV}}$) and Wasserstein-1 ($\mathcal{W}_1$) discrepancy bounds are proved via optimal transport maximal coupling and centered Kantorovich-Rubinstein duality:
   $$\mathcal{W}_1(\pi, \widehat{\pi}) \le \mathrm{diam}(\Theta) \cdot d_{\mathrm{TV}}(\pi, \widehat{\pi}) \le \frac{\mathrm{diam}(\Theta)}{2}\left[\exp(2R\varepsilon + \varepsilon^2) - 1\right] = \mathcal{O}(\sigma_{\mathrm{noise}}^{-2})$$
3. **Parameter-Space Error Localization**: Errors situated inside the high-posterior-density bulk interval $\Omega_{\mathrm{bulk}}$ govern posterior distortion, whereas identical errors in unvisited prior tails produce zero detectable discrepancy at continuous quadrature precision (distortion ratios $> 10^2$ to $> 10^{13}$).
4. **Diagnostic Predictive Superiority in Diffusion-Dominated Dynamics**: Integrated log-likelihood perturbation $\|\Delta \log \mathcal{L}\|_{L_1}$ and posterior-weighted forward error $E_{\mathrm{posterior}}$ provide statistically superior predictors of posterior discrepancy over conventional global error ($p_{\mathrm{Bonf}} < 10^{-12}$) in parabolic diffusion and transport-diffusion systems.
5. **Observational Projection Crossover in Multi-Dimensional Domains**: In higher dimensions ($d=5$ parametric multi-mode diffusion and 2D Incompressible Navier-Stokes with 80 velocity channels), sparse sensor distributions introduce observational projection variance, causing domain-integrated field errors to statistically significantly outperform sparse-sensor likelihood diagnostics ($p < 0.015$).
6. **Bayesian Fidelity Ratio (BFR)**: A dimensionless MCMC sampling noise-floor calibration framework ($\mathrm{BFR}_{99} \approx 1.63 - 2.28$) establishing an empirical threshold below which apparent surrogate bias cannot be distinguished from finite-sample Monte Carlo stochasticity.

```
┌─────────────────────────────────────────────────────────┐
│              FORWARD SURROGATE ERROR                    │
│   e(θ) = || u_PINN(·; θ) - u_exact(·; θ) ||_L2(Ω)       │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼  [Observation Operator H]
┌─────────────────────────────────────────────────────────┐
│             SENSOR-PROJECTED SURROGATE ERROR            │
│   ΔG(θ) = H[F_PINN(θ)] - H[F_exact(θ)]                 │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼  [Measurement Noise σ]
┌─────────────────────────────────────────────────────────┐
│             LOG-LIKELIHOOD PERTURBATION                 │
│   Δ log L(θ) = -1/(2σ²) [ ||y - G_PINN||² - ||y - G||² ]│
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼  [Bayesian Posterior Reweighting]
┌─────────────────────────────────────────────────────────┐
│             POSTERIOR WASSERSTEIN DISCREPANCY           │
│   W₁(π, π̂) = ∫ | F_exact(θ) - F_PINN(θ) | dθ            │
└─────────────────────────────────────────────────────────┘
```

---

## Benchmark Systems and Empirical Findings

The empirical validation suite evaluates a comprehensive cross-dynamical benchmark suite ($N=280$ total trained surrogates across six physical dynamical classes):

| Benchmark PDE System | Physical Mechanism | Exact Parameter $\theta^*$ | Effect Size $\Delta r$ | Williams Test ($p$-value) | Key Finding |
|---|---|:---:|:---:|:---:|:---:|
| **1D Heat Equation** | Linear Parabolic Diffusion | $\alpha^* = 0.50$ | $\mathbf{+0.0971}$ | $t = +10.12 \ (p = 1.95 \times 10^{-13})$ | Likelihood dominates ($r = 0.972$) |
| **1D Wave Equation** | Hyperbolic Wave | $c^* = 1.00$ | $+0.0052$ | $t = +0.27 \ (p = 1.0000, \text{NS})$ | Phase-shift sensitivity |
| **1D Advection-Diffusion** | Directional Transport | $v^* = 1.00$ | $\mathbf{+0.0902}$ | $t = +13.83 \ (p < 10^{-14})$ | Strong likelihood coupling ($r = 0.975$) |
| **1D Viscous Burgers** | Nonlinear Shock Wave | $\nu^* = 0.05$ | $+0.0035$ | $t = +0.15 \ (p = 1.0000, \text{NS})$ | Convective-diffusive attenuation |
| **2D Navier-Stokes** | Incompressible Fluid Flow | $\nu^* = 0.05$ | $-0.1067$ | $t = -3.86 \ (p = 0.00127)$ | **Spatial crossover**: Field error ($r=0.848$) beats sparse likelihood |
| **5D Multi-Mode Diffusion** | Multi-Parameter Dissipation | $\boldsymbol{\theta}^* = (0.5, 0.08, -0.05, 0.04, -0.02)$ | $-0.2964$ | $t = -2.71 \ (p = 0.0147)$ | **Dimensional crossover**: Field error ($r=0.829$) beats sparse likelihood ($r=0.532$) |

*All benchmark problems feature closed-form analytical reference solutions verified to machine precision ($< 10^{-15}$ residual) to isolate surrogate error from reference numerical discretization error.*

---

## Diagnostic Validation Protocol for Neural Surrogates

For researchers and practitioners deploying neural network forward surrogates in Bayesian inverse problems:

1. **Avoid Sole Reliance on Global Forward Norms**: Global $L_2$ error ($E_{\mathrm{global}}$) averages error uniformly over uninformative parameter space and correlates poorly with localized posterior distortion.
2. **Execute Rapid Pilot-Informed Posterior Weighting**: Run a fast preliminary inversion (e.g., maximum a posteriori optimization or short pilot MCMC chains, execution time $< 0.1\,\mathrm{s}$) to identify the estimated high-density bulk interval $\widehat{\Omega}_{\mathrm{bulk}}$, and evaluate posterior-weighted error $E_{\mathrm{posterior}}$ or integrated likelihood perturbation $\|\Delta \log \mathcal{L}\|_{L_1}$ (validation overhead $< 0.02\,\mathrm{s}$, $< 0.2\%$ of surrogate training time).
3. **Calibrate Against Empirical MCMC Sampling Noise**: When comparing surrogate posteriors using MCMC, run paired Exact-vs-Exact control chains to compute the baseline sampling noise floor $\overline{\mathcal{W}}_1^{\mathrm{ctrl}}$. Ensure surrogate discrepancy satisfies $\mathrm{BFR} > \mathrm{BFR}_{99} \approx 1.63 - 2.28$ to prevent false discoveries of surrogate bias.

---

## Repository Architecture

```
PINN-Bayesian-Posterior-Fidelity/
├── bayesian/                           # Bayesian MCMC inference framework
│   ├── likelihood.py                   # Gaussian log-likelihood & misfit functionals
│   ├── metropolis_hastings.py          # Metropolis-Hastings MCMC sampling engine
│   ├── prior.py                        # Prior distributions (Uniform, Gaussian, Truncated)
│   ├── proposal.py                     # Random walk proposal generators
│   ├── forward_operator.py             # Exact & PINN surrogate forward evaluation interfaces
│   ├── observation_operator.py         # Space-time sensor projection operators
│   ├── posterior.py                    # Posterior extraction & moment analysis
│   ├── two_chain_sampler.py            # Coupled two-chain (Exact-vs-PINN, Exact-vs-Exact) sampler
│   ├── two_chain_analysis.py           # Observational distance diagnostics & W1 computations
│   ├── two_chain_plots.py              # Visualizations for paired MCMC chain comparisons
│   └── diagnostics.py                  # MCMC convergence diagnostics (ESS, autocorrelation)
├── experiments/                        # Experimental runners & benchmark definitions
│   ├── cross_pde/                      # Cross-PDE benchmarking suite (N=240 ensemble)
│   │   ├── pde_definitions.py          # Analytical benchmark PDEs & sensor layouts
│   │   ├── trainer.py                  # Parametric PINN trainer (Adam + L-BFGS)
│   │   ├── runner.py                   # Cross-PDE batch execution engine
│   │   ├── plots.py                    # Publication figure generation utilities
│   │   ├── run_cross_pde_benchmark.py  # Master 240-model cross-dynamical benchmark runner
│   │   ├── run_statistical_ancova_and_d5.py # Categorical ANCOVA, multiplicity corrections, & 5D analysis
│   │   ├── navier_stokes_2d.py         # 2D Navier-Stokes Taylor-Green analytical solver
│   │   ├── run_navier_stokes_2d_campaign.py # Navier-Stokes 20-model training campaign
│   │   ├── plot_navier_stokes_2d.py    # Figure 26 generator (streamlines, continuous crossover)
│   │   ├── heat_d5.py                  # 5D multi-mode diffusion benchmark analytical solver
│   │   ├── run_heat_d5_campaign.py     # 5D multi-mode diffusion 20-model training campaign
│   │   ├── plot_heat_d5.py             # Figure 27 generator (space-time field, crossover, marginals)
│   │   ├── pure_pinn_ablation.py       # Differential equation physics-loss ablation
│   │   └── generate_all_publication_vector_figures.py # Master publication vector graphics generator (9 figures)
│   └── metrics.py                      # E_global, E_posterior, W1, SW1, and BFR implementations
├── results/                            # Numerical datasets & validation summaries
│   ├── cross_pde_n60/                  # Datasets for 240 cross-dynamical surrogate models
│   │   ├── all_240models_raw.csv       # Raw metrics for all 240 trained surrogates
│   │   ├── categorical_ancova_results.csv # 12-test ANCOVA multiplicity table
│   │   ├── heat_d5_summary.json        # 5D high-dimensional validation dataset
│   │   └── bfr_sensitivity_analysis.json # BFR calibration sensitivity sweep
│   ├── navier_stokes_2d/               # 2D Navier-Stokes multi-dimensional benchmark
│   │   ├── ns2d_20models_raw.csv       # Raw metrics for 20 Navier-Stokes models
│   │   ├── ns2d_summary.json           # Correlation and Williams test summary
│   │   ├── ns2d_localization_sweep.csv # 2D fluid flow error localization sweep
│   │   └── checkpoints/                # Serialized PyTorch model checkpoints (.pt)
│   ├── heat_d5/                        # 5D multi-mode parametric diffusion benchmark
│   │   ├── heat_d5_20models_raw.csv    # Raw metrics for 20 5D diffusion models
│   │   ├── heat_d5_summary.json        # Correlation and Williams test summary (d=5)
│   │   └── checkpoints/                # Serialized PyTorch model checkpoints (.pt)
│   ├── sensitivity_analysis/           # Robustness and sensitivity evaluation datasets
│   │   ├── noise_sweep/                # Continuous observation noise modulation
│   │   ├── mcmc_noise_floor/           # Dimensionless BFR sampling calibration
│   │   ├── sensor_geometry/            # Sensor layout resolution sweep
│   │   ├── error_localization/         # Parameter error localization sweep
│   │   └── non_oracle_validation/      # Budget-constrained validation analysis
│   └── ablation/                       # Methodological ablation studies
│       └── pure_pinn_ablation/         # Pure physics loss vs supervised comparison
└── tests/                              # Automated test suite
    ├── test_cross_pde_suite.py         # Analytical PDE residual verifications
    ├── test_experiments_suite.py       # Baseline and sweep verification tests
    ├── test_heat_equation_pinn.py      # Autograd physics loss & architecture tests
    ├── test_navier_stokes_2d.py        # 2D Navier-Stokes autograd residual & W1 tests
    ├── test_heat_d5.py                 # 5D multi-mode diffusion autograd residual & SW1 tests
    ├── test_research_metrics.py        # Error metrics & Wasserstein implementations
    ├── test_statistical_methodology.py # Williams formula, ANCOVA, OT coupling proofs
    └── test_two_chain_mcmc.py          # MCMC stochastic control & consistency tests
```

---

## Installation and Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/KartikeyaGangwar/PINN-Bayesian-Posterior-Fidelity.git
cd PINN-Bayesian-Posterior-Fidelity
pip install -r requirements.txt
```

### Running Tests

Execute the automated test suite:

```bash
pytest tests/
```

---

## Reproducing Computational Experiments

1. **Cross-PDE Benchmark ($N=240$ Models)**:
   ```bash
   python experiments/cross_pde/run_cross_pde_benchmark.py
   ```

2. **Categorical ANCOVA and Multiplicity Corrections**:
   ```bash
   python experiments/cross_pde/run_statistical_ancova_and_d5.py
   ```

3. **2D Incompressible Navier-Stokes Benchmark**:
   ```bash
   python experiments/cross_pde/run_navier_stokes_2d_campaign.py
   ```

4. **5D Multi-Mode Parametric Diffusion Benchmark**:
   ```bash
   python experiments/cross_pde/run_heat_d5_campaign.py
   ```

5. **Generate Publication Figures**:
   ```bash
   python experiments/cross_pde/generate_all_publication_vector_figures.py
   ```

---

## Manuscript and Computational Reproducibility

The research article associated with this computational study has been prepared for submission to the ***Journal of Computational Physics* (JCP)** (Elsevier). All experimental scripts, data files, trained model checkpoints, and statistical routines in this repository allow complete independent verification and reproduction of every result and figure reported in the paper.

---

## Citation

If you utilize this methodology, diagnostic validation protocol, or codebase in your research, please cite:

```bibtex
@article{singh2026parameter,
  title={Parameter-Space Error Localization and Posterior Fidelity of Physics-Informed Neural Network Forward Surrogates in Bayesian Inverse Problems},
  author={Singh, Kartikey},
  journal={Journal of Computational Physics},
  year={2026},
  note={Under review. Preprint available at https://github.com/KartikeyaGangwar/PINN-Bayesian-Posterior-Fidelity}
}
```

---

## License

This project is open-source and licensed under the [MIT License](LICENSE).
