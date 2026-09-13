import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from experiments.cross_pde.pde_definitions import get_pde_benchmark, generate_space_time_sensors

def generate_control_samples(
    P: int = 50,
    n_samples: int = 10000,
    burnin: int = 2000,
    proposal_std: float = 0.012,
    output_csv: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate P=50 Exact-vs-Exact MCMC control chain pairs to establish
    the empirical Wasserstein-1 sampling noise floor distribution.
    """
    pde = get_pde_benchmark("heat")
    cfg = pde.config
    sensors = generate_space_time_sensors(n_sensors=40, seed=42)

    y_clean = np.array([pde.exact_solution(sensors[m, 0], sensors[m, 1], cfg.true_param) for m in range(len(sensors))])
    rng_obs = np.random.default_rng(100)
    y_obs = y_clean + rng_obs.normal(0.0, cfg.noise_std, size=len(sensors))

    def log_likelihood(alpha_val):
        if alpha_val <= 0.05 or alpha_val >= 2.5:
            return -np.inf
        preds = np.array([pde.exact_solution(sensors[m, 0], sensors[m, 1], alpha_val) for m in range(len(sensors))])
        return -0.5 * np.sum((preds - y_obs)**2) / (cfg.noise_std**2)

    def log_prior(alpha_val):
        if alpha_val <= 0.05 or alpha_val >= 2.5:
            return -np.inf
        return stats.lognorm.logpdf(alpha_val, s=cfg.prior_sigma, scale=np.exp(cfg.prior_mu))

    def run_mcmc_chain(n_s, b_in, p_std, seed):
        rng = np.random.default_rng(seed)
        samples = np.zeros(n_s)
        curr = cfg.true_param + rng.normal(0.0, 0.05)
        curr_lp = log_likelihood(curr) + log_prior(curr)

        for i in range(n_s):
            prop = curr + rng.normal(0.0, p_std)
            prop_lp = log_likelihood(prop) + log_prior(prop)
            if np.log(rng.uniform(0.0, 1.0) + 1e-300) < (prop_lp - curr_lp):
                curr = prop
                curr_lp = prop_lp
            samples[i] = curr
        return samples[b_in:]

    print(f"Generating P={P} Exact-vs-Exact MCMC control pairs (N={n_samples}, burnin={burnin})...", flush=True)
    w1_ctrls = []
    for p in range(P):
        c1 = run_mcmc_chain(n_samples, burnin, proposal_std, seed=1000 + 2*p)
        c2 = run_mcmc_chain(n_samples, burnin, proposal_std, seed=1000 + 2*p + 1)
        c1_s = np.sort(c1)
        c2_s = np.sort(c2)
        w1_val = float(stats.wasserstein_distance(c1_s, c2_s))
        w1_ctrls.append(w1_val)

    mean_w1 = float(np.mean(w1_ctrls))
    std_w1 = float(np.std(w1_ctrls, ddof=1))
    p95 = float(np.percentile(w1_ctrls, 95))
    p99 = float(np.percentile(w1_ctrls, 99))
    print(f"Completed P={P}: Mean={mean_w1:.4e}, Std={std_w1:.4e}, P95={p95:.4e}, P99={p99:.4e}", flush=True)

    df = pd.DataFrame({
        "pair_idx": np.arange(1, P + 1),
        "w1_ctrl": w1_ctrls
    })

    if output_csv is None:
        output_csv = repo_root / "results" / "cross_pde_n60" / "exact_control_distances_p50.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved exact control distances to: {output_csv}", flush=True)
    return df

if __name__ == "__main__":
    generate_control_samples()
