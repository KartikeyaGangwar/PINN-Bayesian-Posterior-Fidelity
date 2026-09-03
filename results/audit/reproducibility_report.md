# Scientific Audit: Reproducibility & Environment Report

**Audit Target**: Parameter-Space Error Localization and Posterior Fidelity of PINN Forward Surrogates in Bayesian Inverse Problems  
**Date**: August 2026  
**Auditor**: Lead Scientific Auditor

---

## 1. Verified Software & Hardware Environment

| Component | Specification | Verified Status |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (AMD64) | Tested & Compatible |
| **Python Version** | Python 3.13.2 | Tested & Compatible |
| **PyTorch** | PyTorch 2.6.0+cu124 (CUDA 12.4 enabled) | GPU Acceleration Active |
| **NumPy** | NumPy 2.2.3 | Validated |
| **SciPy** | SciPy 1.15.2 | Validated |
| **Pandas** | Pandas 2.2.3 | Validated |
| **Matplotlib** | Matplotlib 3.10.0 | Validated |
| **PyTest** | PyTest 9.1.1 | 42/42 Passing Tests |
| **Primary GPU** | NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) | Active (cuBLAS / autograd) |

---

## 2. Deterministic Reproducibility Protocol

A new researcher can clone this repository and reproduce all findings through the following step-by-step procedure:

### Step 1: Install Dependencies
```bash
pip install torch numpy scipy pandas matplotlib pytest
```

### Step 2: Run Full Automated Verification Suite
```bash
python -m pytest tests/ -v
# Verified Result: 42 passed in ~13.5 seconds
```

### Step 3: Train Master Canonical PINN Surrogate
```bash
python parametric_surrogate/trainer.py
# Produces: results/heat_equation_pinn.pth (~130 seconds)
```

### Step 4: Run Baseline Two-Chain Inference
```bash
python run_two_chain_standalone.py
# Produces: results/research_two_chain_results.npz and results/two_chain_summary.json
```

### Step 5: Execute Experimental Research Suite
```bash
# Phase I: 10-Realization Baseline Robustness
python experiments/phase1_baseline.py

# Phase II: 6-Level Surrogate Accuracy Sweep
python experiments/phase2_accuracy_sweep.py

# Phase III & IV: Error Localization & Weighted Diagnostics
python experiments/phase3_error_localization.py

# Phase V: Observation Noise Sweep
python experiments/phase5_noise_sweep.py

# Phase VI & VII: Sensor Density & Posterior Concentration
python experiments/phase6_sensor_sweep.py

# Phase VIII & IX: Posterior-Aware Training & Multi-Stage Adaptive Refinement
python experiments/phase8_posterior_aware.py
```

### Step 6: Generate All 12 Publication Figures
```bash
python experiments/plots.py
# Outputs 12 PNG figures to results/research_figures/
```

---

## 3. Benchmarked Computational Runtimes

| Phase / Script | Wall-Clock Time | Primary Compute Load | Output Artifact |
| :--- | :--- | :--- | :--- |
| **Unit Tests** | 12.3 s | CPU / GPU forward passes | `tests/` passing report |
| **Master PINN Training** | 132.5 s | GPU Adam (500 ep) + L-BFGS (35 it) | `results/heat_equation_pinn.pth` |
| **Phase I (10 Runs)** | 28.4 s | 20 MCMC Chains (5000 it) | `results/phase1_baseline/` |
| **Phase II (6 Models)** | 485.2 s | 6 Train Regimes + 6 MCMC Chains | `results/phase2_accuracy_sweep/` |
| **Phase III/IV (3 Models)** | 398.1 s | 3 Targeted Train + 3 MCMC Chains | `results/phase3_error_localization/` |
| **Phase V (30 Runs)** | 184.6 s | 60 MCMC Chains across noise levels | `results/phase5_noise_sweep/` |
| **Phase VI/VII (20 Runs)** | 142.3 s | 40 MCMC Chains across sensor levels| `results/phase6_sensor_sweep/` |
| **Phase VIII/IX (Adaptive)**| 312.8 s | Pilot + Refinement Train + MCMC | `results/phase8_posterior_aware/` |
| **Figure Generation** | 4.8 s | Matplotlib 300 DPI rendering | `results/research_figures/` (12 PNGs) |
