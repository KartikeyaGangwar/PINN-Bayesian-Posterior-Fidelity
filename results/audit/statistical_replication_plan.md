# Statistical Replication & Publication Hardening Plan

**Audit Target**: Bayesian PINN Fidelity & Parameter-Resolved Error Localization Framework  
**Date**: August 2026  
**Auditor**: Lead Scientific Auditor

---

## 1. Prioritized Experimental Matrix for Top-Tier Journal Submission

To transition this research program from a solid computational study to an unassailable top-tier journal publication (e.g. *Journal of Computational Physics*, *SIAM/ASA Journal on Uncertainty Quantification*), we define the prioritized minimum experiment set.

---

### Priority 1: Multi-Seed Surrogate Ensemble Replication (High Priority / Moderate Compute)
- **Goal**: Replicate Phase II (Accuracy Sweep) across $K = 10$ independent training random seeds per accuracy level ($6 \times 10 = 60$ surrogate models).
- **Outcome**:
  - Eliminates single-seed training variability.
  - Generates empirical 95% bootstrap confidence bands for $\mathcal{W}_1(\pi_E, \pi_A)$ vs forward relative $L_2$ error.
  - Directly tests whether the non-monotonicity observed at high accuracy is a robust feature of parameter-space error localization.
- **Estimated Compute**: $\approx 45$ minutes on single GPU.

---

### Priority 2: Analytical Gold-Standard Quadrature Benchmarking (High Priority / Low Compute)
- **Goal**: Augment empirical MCMC comparisons with exact 1D numerical quadrature on dense grids ($N_\alpha = 10,000$).
- **Outcome**:
  - Isolates surrogate-induced likelihood distortion $\Delta \log L(\alpha)$ from MCMC finite-sample Monte Carlo error.
  - Computes exact continuous Wasserstein-1 distances $\mathcal{W}_1^{\text{quad}}$, exact Kullback-Leibler divergences $D_{\text{KL}}(\pi_{\text{exact}} \parallel \pi_{\text{PINN}})$, and Jensen-Shannon divergences.
- **Estimated Compute**: $\approx 2$ minutes on CPU/GPU.

---

### Priority 3: Noise Floor Calibration & Multi-Pair BFR Estimation (High Priority / Low Compute)
- **Goal**: Define the denominator of the proposed Bayesian Fidelity Ratio using an ensemble of $P = 50$ paired Exact-vs-Exact control chains:
  $$\overline{\mathcal{W}}_1^{\text{control}} = \frac{1}{P} \sum_{p=1}^P \mathcal{W}_1(\pi_{E1}^{(p)}, \pi_{E2}^{(p)})$$
- **Outcome**:
  - Provides a statistically rigid denominator with $< 4\%$ standard error.
  - Allows computation of two-sample permutation $p$-values to formally test $H_0: \pi_{\text{PINN}} = \pi_{\text{exact}}$.
- **Estimated Compute**: $\approx 3$ minutes on GPU.

---

### Priority 4: 2D Spatial Extension (Moderate Priority / High Compute)
- **Goal**: Test the parameter-resolved error localization framework on a 2D Heat / Advection-Diffusion equation with 2 unknown parameters $(\alpha, \beta)$.
- **Outcome**:
  - Demonstrates that error localization dominance generalizes to multi-parameter posterior manifolds.
- **Estimated Compute**: $\approx 2$ hours on GPU.
