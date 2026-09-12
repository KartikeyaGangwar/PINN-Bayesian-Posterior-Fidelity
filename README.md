# Parameter-Space Error Localization and Posterior Fidelity of Physics-Informed Neural Network Forward Surrogates in Bayesian Inverse Problems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Test Suite](https://img.shields.io/badge/pytest-49%2F49%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Submission Ready](https://img.shields.io/badge/Status-Submission%20Ready-success.svg)](#)

Official open-source research repository and computational reproducibility suite for the research article:  
**"Parameter-Space Error Localization and Posterior Fidelity of Physics-Informed Neural Network Forward Surrogates in Bayesian Inverse Problems"**

**Author:** Kartikey Singh  
**Affiliation:** University of Delhi, Delhi 110007, India  
**E-mail:** `kartikeysingh525@protonmail.com`  
**ORCID:** [0009-0009-1973-7532](https://orcid.org/0009-0009-1973-7532)  
**Repository:** [https://github.com/KartikeyaGangwar/PINN-Bayesian-Posterior-Fidelity](https://github.com/KartikeyaGangwar/PINN-Bayesian-Posterior-Fidelity)

---

## Scientific Overview and Theoretical Foundations

In Bayesian inverse problems governed by Partial Differential Equations (PDEs), forward surrogate models—such as Physics-Informed Neural Networks (PINNs)—are deployed to accelerate Markov Chain Monte Carlo (MCMC) sampling. Conventional validation practices rely almost exclusively on **uniform global forward-field error metrics** ($E_{\mathrm{global}}$), such as relative $L_2$ errors averaged uniformly across spatial, temporal, and parameter domains.

This repository provides a unified theoretical, algorithmic, and empirical framework demonstrating that:
1. **Forward error is filtered through the observation operator and likelihood functional**: Surrogate approximation error enters the Bayesian posterior strictly through sparse sensor projections $\Delta \mathcal{G}(\theta)$ and the likelihood functional $\Delta \log \mathcal{L}(\theta)$, which exponentially reweights parameter space according to observation consistency.
2. **Mathematical Posterior Stability Bounds**: Under approximate forward operators on compact parameter spaces $\Theta \subset \mathbb{R}^d$, quantitative Total Variation ($d_{\mathrm{TV}}$) and Wasserstein-1 ($\mathcal{W}_1$) discrepancy bounds are proved via optimal transport maximal coupling and centered Kantorovich-Rubinstein duality:
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

## Benchmark Systems and Empirical Results

The empirical validation suite evaluates a comprehensive cross-dynamical benchmark suite ($N=260$ total trained surrogates across five physical dynamical classes):

| Benchmark PDE System | Physical Mechanism | Exact Parameter $\theta^*$ | Effect Size $\Delta r$ | Williams Test ($p$-value) | Key Finding |
|---|---|:---:|:---:|:---:|:---:|
| **1D Heat Equation** | Linear Parabolic Diffusion | $\alpha^* = 0.50$ | $\mathbf{+0.0971}$ | $t = +10.12 \ (p = 1.95 \times 10^{-13})$ | Likelihood dominates ($r = 0.972$) |
| **1D Wave Equation** | Hyperbolic Wave | $c^* = 1.00$ | $+0.0052$ | $t = +0.27 \ (p = 1.0000, \text{NS})$ | Phase-shift sensitivity |
| **1D Advection-Diffusion** | Directional Transport | $v^* = 1.00$ | $\mathbf{+0.0902}$ | $t = +13.83 \ (p < 10^{-14})$ | Strong likelihood coupling ($r = 0.975$) |
| **1D Viscous Burgers** | Nonlinear Shock Wave | $\nu^* = 0.05$ | $+0.0035$ | $t = +0.15 \ (p = 1.0000, \text{NS})$ | Convective-diffusive attenuation |
| **2D Navier-Stokes** | Incompressible Fluid Flow | $\nu^* = 0.05$ | $-0.1067$ | $t = -3.86 \ (p = 0.00127)$ | **Spatial crossover**: Field error ($r=0.848$) beats sparse likelihood |

*All benchmark problems feature closed-form analytical reference solutions verified to machine precision ($< 10^{-15}$ residual) to isolate surrogate error from reference numerical discretization error.*

---

## Practical Three-Step Validation Protocol

For practitioners deploying neural network forward surrogates in Bayesian inverse problems:
1. **Avoid Sole Reliance on Global Forward Norms**: Global $L_2$ error ($E_{\mathrm{global}}$) averages error uniformly over uninformative parameter space and correlates poorly with localized posterior distortion.
2. **Execute Rapid Pilot-Informed Posterior Weighting**: Run a fast preliminary inversion (e.g., maximum a posteriori optimization or short pilot MCMC chains, execution time $< 0.1\,\mathrm{s}$) to identify the estimated high-density bulk interval $\widehat{\Omega}_{\mathrm{bulk}}$, and evaluate posterior-weighted error $E_{\mathrm{posterior}}$ or integrated likelihood perturbation $\|\Delta \log \mathcal{L}\|_{L_1}$ (validation overhead $< 0.02\,\mathrm{s}$, $< 0.2\%$ of surrogate training time).
3. **Calibrate Against Empirical MCMC Sampling Noise**: When comparing surrogate posteriors using MCMC, run paired Exact-vs-Exact control chains to compute the baseline sampling noise floor $\overline{\mathcal{W}}_1^{\mathrm{ctrl}}$. Ensure surrogate discrepancy satisfies $\mathrm{BFR} > \mathrm{BFR}_{99} \approx 1.63 - 2.28$ to prevent false discoveries of surrogate bias.

---

## Repository Structure

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
│   │   ├── run_comprehensive_jcp_hardening.py # Master 240-model reproducibility runner
│   │   ├── run_audit_and_d5_stress_test.py    # ANCOVA, Multiple testing, & d=5 stress test
│   │   ├── navier_stokes_2d.py         # 2D Navier-Stokes Taylor-Green analytical solver
│   │   ├── run_navier_stokes_2d_campaign.py   # Navier-Stokes 20-model training campaign
│   │   ├── pure_pinn_ablation.py       # Differential equation physics-loss ablation
│   │   └── generate_all_publication_vector_figures.py # Master publication vector graphics generator
│   └── metrics.py                      # E_global, E_posterior, W1, and BFR implementations
├── paper/                              # LaTeX manuscript source, figures, & proofs
│   ├── manuscript_cmame.tex            # Master Elsevier CMAME format manuscript (59 pages)
│   ├── manuscript.tex                  # Standard article format manuscript (39 pages)
│   ├── CMAME_Cover_Letter.tex          # Formal editor cover letter
│   ├── CMAME_Highlights.tex            # Elsevier research highlights
│   ├── CMAME_Suggested_Reviewers.md    # Independent reviewer recommendations
│   ├── title_abstract.tex              # Title, author metadata, abstract, & keywords
│   ├── sec_01_introduction.tex         # Literature review & research positioning
│   ├── sec_02_problem_formulation.tex  # Problem setup & theoretical stability bounds
│   ├── sec_03_diagnostic_framework.tex # Error metrics & BFR calibration framework
│   ├── sec_04_benchmark_problems.tex   # Benchmark PDE regimes & experimental design
│   ├── sec_05_results.tex              # Comprehensive empirical results & ANCOVA
│   ├── sec_06_discussion.tex           # Mechanistic interpretation & validation protocol
│   ├── sec_07_conclusion.tex           # Summary of findings & data availability
│   ├── appendix_proofs.tex             # Complete mathematical proofs (Theorems 2.1, 2.3)
│   └── figures/                        # High-resolution vector PDF publication figures
├── results/                            # Numerical datasets & validation summaries
│   ├── cross_pde_n60/                  # Complete datasets for all 240 surrogate models
│   │   ├── all_240models_raw.csv       # Raw metrics for all 240 trained surrogates
│   │   ├── categorical_ancova_results.csv # 12-test ANCOVA multiplicity table
│   │   ├── heat_d5_summary.json        # 5D high-dimensional stress test dataset
│   │   └── bfr_sensitivity_analysis.json # BFR calibration sensitivity sweep
│   └── navier_stokes_2d/               # 2D Navier-Stokes multi-dimensional benchmark
│       ├── ns2d_20models_raw.csv       # Raw metrics for 20 Navier-Stokes models
│       ├── ns2d_summary.json           # Correlation and Williams test summary
│       └── ns2d_localization_sweep.csv # 2D fluid flow error localization sweep
└── tests/                              # Automated unit test suite (49 test functions)
    ├── test_audit_and_statistical_rigor.py # Williams formula, ANCOVA, OT coupling proofs
    ├── test_cross_pde_suite.py         # Analytical PDE residual verifications
    ├── test_heat_equation_pinn.py      # Autograd physics loss & architecture tests
    ├── test_navier_stokes_2d.py        # 2D Navier-Stokes autograd residual & W1 tests
    ├── test_research_metrics.py        # Error metrics & Wasserstein implementations
    └── test_two_chain_mcmc.py          # MCMC stochastic control & consistency tests
```

---

## Installation and Quickstart

### 1. Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/KartikeyaGangwar/PINN-Bayesian-Posterior-Fidelity.git
cd PINN-Bayesian-Posterior-Fidelity
pip install -r requirements.txt
```

*Requirements: Python 3.10+, PyTorch 2.0+, NumPy, SciPy, Matplotlib, Pandas, Statsmodels, PyTest.*

---

### 2. Run Automated Verification Suite (49 / 49 Passing)

Verify mathematical residuals, metrics, statistical formulas, and MCMC samplers:

```bash
python -m pytest tests/ -v
```

Expected output:
```
============================= 49 passed in 24.2s =============================
```

---

### 3. Reproduce Full Cross-PDE Benchmark ($N=240$ Models)

Train the balanced 240-model surrogate ensemble across all four PDE regimes, compute continuous quadrature Wasserstein distances, evaluate likelihood perturbations, and generate all publication figures:

```bash
python experiments/cross_pde/run_comprehensive_jcp_hardening.py
```

---

### 4. Run Categorical ANCOVA and $d=5$ High-Dimensional Stress Test

Execute the fixed-effects Categorical ANCOVA, within-tier subgroup analysis, family-wise multiple testing corrections, and the 5-dimensional parametric thermal diffusion stress test:

```bash
python experiments/cross_pde/run_audit_and_d5_stress_test.py
```

---

### 5. Reproduce 2D Navier-Stokes Incompressible Benchmark

Execute the multi-dimensional fluid dynamics validation campaign and evaluate spatial projection crossover:

```bash
python experiments/cross_pde/run_navier_stokes_2d_campaign.py
```

---

### 6. Compile LaTeX Manuscripts

Compile the Elsevier CMAME formatted manuscript (59 pages):

```bash
cd paper
pdflatex -interaction=nonstopmode manuscript_cmame.tex
bibtex manuscript_cmame
pdflatex -interaction=nonstopmode manuscript_cmame.tex
pdflatex -interaction=nonstopmode manuscript_cmame.tex
```

Or compile the standard article format manuscript (39 pages):

```bash
pdflatex -interaction=nonstopmode manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
```

---

## Citation

If you utilize this methodology, diagnostic validation protocol, or codebase in your research, please cite:

```bibtex
@article{singh2026parameter,
  title={Parameter-Space Error Localization and Posterior Fidelity of Physics-Informed Neural Network Forward Surrogates in Bayesian Inverse Problems},
  author={Singh, Kartikey},
  journal={Computer Methods in Applied Mechanics and Engineering},
  year={2026},
  note={Under review. Preprint available at https://github.com/KartikeyaGangwar/PINN-Bayesian-Posterior-Fidelity}
}
```

---

## License

This project is open-source and licensed under the [MIT License](LICENSE).
