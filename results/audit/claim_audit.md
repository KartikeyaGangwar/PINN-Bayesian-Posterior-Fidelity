# Scientific Claim Audit: Categorization & Verification

**Audit Target**: Bayesian PINN Fidelity & Parameter-Resolved Error Localization Framework  
**Date**: August 2026  
**Auditor**: Lead Scientific Auditor

---

## 1. Classification Taxonomy
- **Category A (Directly Demonstrated)**: Supported by reproducible numerical experiments, exact continuous mathematical checks, and statistical tests.
- **Category B (Supported but Requires Qualification)**: Empirically supported under specific benchmark settings, but requires explicit statement of boundary conditions, observation noise, or budget limits.
- **Category C (Plausible Hypothesis)**: Mechanistically coherent, but currently validated only on 1D parametric heat equation; generalization to complex PDE systems is an open hypothesis.
- **Category D (Not Demonstrated)**: Claims unsupported by the numerical data or conflating distinct statistical concepts.
- **Category E (Contradicted by Evidence)**: Explicitly refuted by numerical or analytical evidence.

---

## 2. Granular Audit of Major Scientific Statements

### Statement 1: "Global forward-field error alone cannot adequately predict or guarantee Bayesian posterior fidelity."
- **Audit Classification**: **CATEGORY A (Directly Demonstrated)**
- **Evidence**:
  1. In Phase II, non-monotonicity is observed: Level 5 ($4.29\%$ global error) achieved lower posterior discrepancy ($\mathcal{W}_1 = 1.29 \times 10^{-3}$) than Level 6 ($2.46\%$ global error $\implies \mathcal{W}_1 = 2.11 \times 10^{-3}$).
  2. In Phase III, PINN-A had $57.18\%$ global error (concentrated in the prior tails) yet achieved $\mathcal{W}_1 = 1.23 \times 10^{-3}$, outperforming models with significantly lower global error but errors in the posterior support.
  3. Continuous mathematical quadrature confirms that error outside the posterior support is weighted by $\pi_{\text{exact}}(\alpha \mid y) \approx 0$, contributing zero to likelihood misfit.

---

### Statement 2: "Surrogate error in the posterior support region dominates Bayesian posterior distortion."
- **Audit Classification**: **CATEGORY A (Directly Demonstrated)**
- **Evidence**:
  1. Causal perturbation tests show that a 5.0 log-likelihood perturbation inside the posterior support $[0.495, 0.505]$ induces $\mathcal{W}_1 = 2.97 \times 10^{-3}$, whereas the exact same perturbation in the prior tails $[1.20, 1.40]$ produces $\mathcal{W}_1 = 0.0000$.
  2. $E_{\text{posterior}} = \int e(\alpha)\pi_{\text{exact}}(\alpha \mid y)d\alpha$ directly reflects the active error experienced by the Markov chain at stationarity.

---

### Statement 3: "Observational noise modulates sensitivity to surrogate error: high noise masks surrogate error, while low noise amplifies it."
- **Audit Classification**: **CATEGORY B (Supported but Requires Qualification)**
- **Qualification Required**:
  - "Masking" does NOT mean surrogate forward error disappears or that surrogate bias becomes zero.
  - Rather, at high observational noise ($\sigma_{\text{noise}} = 0.10$), the posterior standard deviation widens ($\sigma_{\text{post}} \approx 0.057$), making a fixed surrogate shift ($\approx 0.002$) small relative to posterior variance ($\approx 0.04\sigma_{\text{post}}$), driving $\text{BFR} \to 0.95$.
  - At ultra-low noise ($\sigma_{\text{noise}} = 0.001$), posterior standard deviation contracts to $\sigma_{\text{post}} \approx 0.00053$, so the same surrogate shift represents $4.4\sigma_{\text{post}}$, causing $\text{BFR} = 23.38$.

---

### Statement 4: "Posterior concentration increases sensitivity to structured surrogate errors."
- **Audit Classification**: **CATEGORY A (Directly Demonstrated)**
- **Evidence**:
  - Across sensor sweeps $M \in \{10, 20, 40, 80, 160, 320\}$, the correlation between posterior concentration index ($1/\sigma_{\text{post}}$) and BFR is $r = +0.9573$.
  - Higher sensor density sharpens the likelihood gradient, amplifying surrogate model discrepancies in observation space into larger relative posterior shifts.

---

### Statement 5: "Adaptive targeted collocation improves Bayesian fidelity per unit computational cost."
- **Audit Classification**: **CATEGORY B (Supported but Requires Qualification)**
- **Qualification Required**:
  - The $88.5\%$ reduction in BFR observed in Phase IX compares Stage 0 (coarse pilot PINN, 15s budget) to Stage 1 (targeted refinement around pilot posterior).
  - While this demonstrates dramatic efficiency gains over uniform global allocation, the exact compute budget, number of training epochs, and optimization steps must always be reported transparently.

---

### Statement 6: "A Bayesian Fidelity Ratio $\text{BFR} > 1$ represents a statistically significant surrogate distortion."
- **Audit Classification**: **CATEGORY B (Supported but Requires Qualification)**
- **Qualification Required**:
  - $\text{BFR} = \frac{\mathcal{W}_1(\pi_E, \pi_A)}{\mathcal{W}_1(\pi_{E1}, \pi_{E2})}$ is a ratio comparing surrogate discrepancy against an empirical realization of MCMC stochastic noise.
  - Because the denominator has finite Monte Carlo variance ($\text{CV} \approx 31\%$, 90% CI $[3.71 \times 10^{-4}, 8.87 \times 10^{-4}]$), an empirical $\text{BFR}$ of $1.1 - 1.3$ cannot be called "statistically significant distortion" without a formal permutation/bootstrap confidence interval.
  - For publication, values $\text{BFR} > 3.0$ (such as baseline $\text{BFR} = 4.86$ or low-noise $\text{BFR} = 23.4$) are well beyond the 99th percentile of MCMC noise, whereas values in $[0.8, 1.5]$ should be interpreted as within stochastic parity.

---

### Statement 7: "Two paired MCMC chains with exact vs PINN solvers should follow identical paths and then diverge."
- **Audit Classification**: **CATEGORY E (Contradicted by Evidence / Rejected)**
- **Status**: **REJECTED (Correctly classified in codebase)**.
- **Explanation**: MCMC trajectories are stochastic Markov processes. Pathwise trajectory divergence is governed by independent random walk draws, not solver fidelity. The central scientific object is the stationary distribution $\pi(\alpha \mid y)$, not trajectory alignment.
