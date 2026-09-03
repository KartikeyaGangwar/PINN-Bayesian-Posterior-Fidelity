import os
import sys
import json
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import statsmodels.api as sm
from statsmodels.formula.api import ols

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.cross_pde.pde_definitions import get_pde_benchmark, generate_space_time_sensors
from experiments.cross_pde.run_comprehensive_jcp_hardening import williams_test
from parametric_surrogate.parametric_model import ParametricModifiedMLP

output_dir = os.path.join(repo_root, "results", "cross_pde_n60")
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------
# Part 1: Within-Tier Breakdown & Categorical ANCOVA
# -------------------------------------------------------------
print("=======================================================")
print("PART 1: WITHIN-TIER CORRELATION & CATEGORICAL ANCOVA")
print("=======================================================")

df_raw = pd.read_csv(os.path.join(output_dir, "all_240models_raw.csv"))

ancova_results = []
within_tier_summary = []

for pde_name in ["heat", "wave", "advection_diffusion", "burgers"]:
    sub = df_raw[df_raw["pde"] == pde_name].copy()
    sub["tier_cat"] = sub["level"].astype("category")
    
    for diag in ["e_global", "e_posterior", "l1_delta_loglik"]:
        formula = f"w1 ~ C(tier_cat) + {diag}"
        model = ols(formula, data=sub).fit()
        t_stat = model.tvalues[diag]
        p_val = model.pvalues[diag]
        r2_full = model.rsquared
        model_null = ols("w1 ~ C(tier_cat)", data=sub).fit()
        r2_null = model_null.rsquared
        df_resid = model.df_resid
        cat_partial_r = np.sign(t_stat) * np.sqrt((t_stat**2) / (t_stat**2 + df_resid))
        
        ancova_results.append({
            "pde": pde_name,
            "diagnostic": diag,
            "t_stat": float(t_stat),
            "p_val": float(p_val),
            "categorical_partial_r": float(cat_partial_r),
            "incremental_R2": float(r2_full - r2_null),
            "df_resid": int(df_resid)
        })
    
    for tier_name, grp in sub.groupby("level"):
        r_eg = stats.pearsonr(grp["e_global"], grp["w1"])[0]
        r_ep = stats.pearsonr(grp["e_posterior"], grp["w1"])[0]
        r_lik = stats.pearsonr(grp["l1_delta_loglik"], grp["w1"])[0]
        within_tier_summary.append({
            "pde": pde_name,
            "tier": tier_name,
            "n": len(grp),
            "r_eg": float(r_eg),
            "r_ep": float(r_ep),
            "r_lik": float(r_lik),
            "ep_beats_eg": bool(r_ep > r_eg),
            "lik_beats_eg": bool(r_lik > r_eg)
        })

df_ancova = pd.DataFrame(ancova_results)
df_within = pd.DataFrame(within_tier_summary)

print("\nCategorical ANCOVA (Controlling for Tier as Categorical Factor):")
print(df_ancova.to_string(index=False))

print("\nWithin-Tier Diagnostics Win Rate:")
for pde_name in ["heat", "wave", "advection_diffusion", "burgers"]:
    sub_w = df_within[df_within["pde"] == pde_name]
    ep_wins = sub_w["ep_beats_eg"].sum()
    lik_wins = sub_w["lik_beats_eg"].sum()
    print(f"  {pde_name.upper():20s}: Ep > Eg in {ep_wins}/6 tiers ({ep_wins/6*100:.1f}%), Lik > Eg in {lik_wins}/6 tiers ({lik_wins/6*100:.1f}%)")

total_ep_wins = df_within["ep_beats_eg"].sum()
total_lik_wins = df_within["lik_beats_eg"].sum()
print(f"\nAcross All 24 Tiers: Ep > Eg in {total_ep_wins}/24 ({total_ep_wins/24*100:.1f}%), Lik > Eg in {total_lik_wins}/24 ({total_lik_wins/24*100:.1f}%)")

df_ancova.to_csv(os.path.join(output_dir, "categorical_ancova_results.csv"), index=False)
df_within.to_csv(os.path.join(output_dir, "within_tier_correlations.csv"), index=False)

# -------------------------------------------------------------
# Part 2: Multiple Comparison Correction
# -------------------------------------------------------------
print("\n=======================================================")
print("PART 2: MULTIPLE COMPARISON CORRECTION ON WILLIAMS TESTS")
print("=======================================================")

with open(os.path.join(output_dir, "cross_pde_n60_meta_summary.json"), "r") as f:
    meta_summaries = json.load(f)

hypotheses = []
for p in meta_summaries:
    hypotheses.append({
        "pde": p["pde"],
        "comparison": "E_posterior vs E_global",
        "t_stat": p["williams_ep_eg_t"],
        "p_raw": p["williams_ep_eg_p"]
    })
    hypotheses.append({
        "pde": p["pde"],
        "comparison": "Likelihood vs E_global",
        "t_stat": p["williams_lik_eg_t"],
        "p_raw": p["williams_lik_eg_p"]
    })

df_hyp = pd.DataFrame(hypotheses)
m = len(df_hyp)
df_hyp["p_bonferroni"] = np.minimum(1.0, df_hyp["p_raw"] * m)

df_hyp = df_hyp.sort_values("p_raw").reset_index(drop=True)
df_hyp["rank"] = np.arange(1, m + 1)
df_hyp["p_fdr_bh"] = np.minimum(1.0, df_hyp["p_raw"] * m / df_hyp["rank"])
for i in range(m - 2, -1, -1):
    df_hyp.loc[i, "p_fdr_bh"] = min(df_hyp.loc[i, "p_fdr_bh"], df_hyp.loc[i + 1, "p_fdr_bh"])

df_hyp["sig_bonferroni_05"] = df_hyp["p_bonferroni"] < 0.05
df_hyp["sig_fdr_05"] = df_hyp["p_fdr_bh"] < 0.05

print(df_hyp.to_string(index=False))
df_hyp.to_csv(os.path.join(output_dir, "multiple_testing_correction.csv"), index=False)

# -------------------------------------------------------------
# Part 3: d=5 Parametric Heat Equation Stress Test
# -------------------------------------------------------------
print("\n=======================================================")
print("PART 3: HIGHER-DIMENSIONAL (d=5) PARAMETRIC STRESS TEST")
print("=======================================================")

theta_true = np.array([0.50, 0.08, -0.05, 0.04, -0.02])
d = 5
param_bounds = np.array([
    [0.20, 1.00],
    [-0.15, 0.15],
    [-0.15, 0.15],
    [-0.10, 0.10],
    [-0.10, 0.10]
])

def exact_solve_d5(x, t, theta):
    u = np.zeros_like(x * t, dtype=np.float64)
    for k in range(1, 6):
        c_k = theta[0] + theta[k-1] * 0.5
        decay = np.exp(- (k * np.pi)**2 * c_k * t)
        mode = (1.0 / k) * np.sin(k * np.pi * x)
        u += mode * decay
    return u

sensors_d5 = generate_space_time_sensors(n_sensors=40, seed=42)
sigma_noise_d5 = 0.010
y_clean_d5 = np.array([exact_solve_d5(sensors_d5[m, 0], sensors_d5[m, 1], theta_true) for m in range(len(sensors_d5))])
rng_d5 = np.random.default_rng(2026)
y_obs_d5 = y_clean_d5 + rng_d5.normal(0.0, sigma_noise_d5, size=len(sensors_d5))

prior_means = np.mean(param_bounds, axis=1)
prior_stds = (param_bounds[:, 1] - param_bounds[:, 0]) / 4.0

def log_likelihood_d5(theta):
    for i in range(d):
        if theta[i] < param_bounds[i, 0] or theta[i] > param_bounds[i, 1]:
            return -np.inf
    preds = np.array([exact_solve_d5(sensors_d5[m, 0], sensors_d5[m, 1], theta) for m in range(len(sensors_d5))])
    return -0.5 * np.sum((preds - y_obs_d5)**2) / (sigma_noise_d5**2)

def log_prior_d5(theta):
    for i in range(d):
        if theta[i] < param_bounds[i, 0] or theta[i] > param_bounds[i, 1]:
            return -np.inf
    return np.sum(stats.norm.logpdf(theta, loc=prior_means, scale=prior_stds))

print("Running reference MCMC (15,000 steps) for exact d=5 posterior...", flush=True)

def run_mcmc_d5(log_lik_fn, n_samples=15000, burnin=3000, seed=42):
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, d))
    curr = theta_true + rng.normal(0.0, 0.01, size=d)
    curr_lp = log_lik_fn(curr) + log_prior_d5(curr)
    prop_std = np.array([0.015, 0.012, 0.012, 0.010, 0.010])
    
    for i in range(n_samples):
        prop = curr + rng.normal(0.0, prop_std)
        prop_lp = log_lik_fn(prop) + log_prior_d5(prop)
        if np.log(rng.uniform(0.0, 1.0) + 1e-300) < (prop_lp - curr_lp):
            curr = prop
            curr_lp = prop_lp
        samples[i] = curr
    return samples[burnin:]

exact_samples_d5 = run_mcmc_d5(log_likelihood_d5, seed=42)
print(f"Exact MCMC posterior sampled: shape={exact_samples_d5.shape}, mean={np.mean(exact_samples_d5, axis=0)}")

print("Training N=20 PINN forward surrogates in d=5 (4 tiers x 5 seeds)...", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_x, n_t = 20, 20
x_g = np.linspace(0.0, 1.0, n_x)
t_g = np.linspace(0.0, 1.0, n_t)
X_m, T_m = np.meshgrid(x_g, t_g, indexing="ij")

rng_lhs = np.random.default_rng(100)
n_train_theta = 40
lhs_thetas = rng_lhs.uniform(param_bounds[:, 0], param_bounds[:, 1], size=(n_train_theta, d))

inputs_list_d5 = []
targets_list_d5 = []
for th in lhs_thetas:
    u_ex = exact_solve_d5(X_m, T_m, th)
    coords = np.column_stack([
        X_m.flatten(),
        T_m.flatten(),
        np.tile(th, (X_m.size, 1))
    ])
    inputs_list_d5.append(coords)
    targets_list_d5.append(u_ex.flatten()[:, None])

all_inputs_d5 = torch.tensor(np.vstack(inputs_list_d5), dtype=torch.float64)
all_targets_d5 = torch.tensor(np.vstack(targets_list_d5), dtype=torch.float64)

tiers_d5 = [
    {"name": "L1", "epochs": 40, "lbfgs": 0},
    {"name": "L2", "epochs": 120, "lbfgs": 10},
    {"name": "L3", "epochs": 250, "lbfgs": 25},
    {"name": "L4", "epochs": 450, "lbfgs": 40}
]
seeds_d5 = [101, 102, 103, 104, 105]

test_thetas_global = rng_lhs.uniform(param_bounds[:, 0], param_bounds[:, 1], size=(50, d))
test_thetas_posterior = exact_samples_d5[rng_lhs.choice(len(exact_samples_d5), size=50, replace=False)]

d5_results = []

for lvl_idx, lvl in enumerate(tiers_d5):
    for s_idx, s in enumerate(seeds_d5):
        torch.manual_seed(s * 10 + lvl_idx)
        np.random.seed(s * 10 + lvl_idx)
        
        model = ParametricModifiedMLP(
            n_input=7, n_output=1, n_hidden=64, n_layers=4, use_fourier=False
        ).to(device).to(torch.float64)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        dataset = torch.utils.data.TensorDataset(all_inputs_d5.to(device), all_targets_d5.to(device))
        loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
        
        model.train()
        for ep in range(lvl["epochs"]):
            for bx, by in loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = torch.mean((pred - by)**2)
                loss.backward()
                optimizer.step()
                
        if lvl["lbfgs"] > 0:
            lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lvl["lbfgs"], line_search_fn="strong_wolfe")
            def closure():
                lbfgs.zero_grad()
                pred = model(all_inputs_d5.to(device))
                loss = torch.mean((pred - all_targets_d5.to(device))**2)
                loss.backward()
                return loss
            lbfgs.step(closure)
            
        model.eval()
        
        eg_errs = []
        with torch.no_grad():
            for th in test_thetas_global:
                coords = np.column_stack([X_m.flatten(), T_m.flatten(), np.tile(th, (X_m.size, 1))])
                t_in = torch.tensor(coords, dtype=torch.float64, device=device)
                pred_f = model(t_in).cpu().numpy().reshape(X_m.shape)
                ex_f = exact_solve_d5(X_m, T_m, th)
                rel_err = np.linalg.norm(pred_f - ex_f) / (np.linalg.norm(ex_f) + 1e-15)
                eg_errs.append(rel_err)
        e_global = float(np.mean(eg_errs))
        
        ep_errs = []
        with torch.no_grad():
            for th in test_thetas_posterior:
                coords = np.column_stack([X_m.flatten(), T_m.flatten(), np.tile(th, (X_m.size, 1))])
                t_in = torch.tensor(coords, dtype=torch.float64, device=device)
                pred_f = model(t_in).cpu().numpy().reshape(X_m.shape)
                ex_f = exact_solve_d5(X_m, T_m, th)
                rel_err = np.linalg.norm(pred_f - ex_f) / (np.linalg.norm(ex_f) + 1e-15)
                ep_errs.append(rel_err)
        e_posterior = float(np.mean(ep_errs))
        
        lik_errs = []
        with torch.no_grad():
            for th in test_thetas_posterior:
                s_in = np.column_stack([sensors_d5[:, 0], sensors_d5[:, 1], np.tile(th, (len(sensors_d5), 1))])
                t_in = torch.tensor(s_in, dtype=torch.float64, device=device)
                p_sens = model(t_in).cpu().numpy().flatten()
                
                ex_lik = log_likelihood_d5(th)
                misfit_p = 0.5 * np.sum((p_sens - y_obs_d5)**2) / (sigma_noise_d5**2)
                norm_const = 0.5 * len(sensors_d5) * np.log(2.0 * np.pi * (sigma_noise_d5**2))
                p_lik = -norm_const - misfit_p
                lik_errs.append(abs(p_lik - ex_lik))
        l1_delta_loglik = float(np.mean(lik_errs))
        
        def pinn_log_likelihood(theta):
            for i in range(d):
                if theta[i] < param_bounds[i, 0] or theta[i] > param_bounds[i, 1]:
                    return -np.inf
            s_in = np.column_stack([sensors_d5[:, 0], sensors_d5[:, 1], np.tile(theta, (len(sensors_d5), 1))])
            with torch.no_grad():
                t_in = torch.tensor(s_in, dtype=torch.float64, device=device)
                p_sens = model(t_in).cpu().numpy().flatten()
            misfit_p = 0.5 * np.sum((p_sens - y_obs_d5)**2) / (sigma_noise_d5**2)
            norm_const = 0.5 * len(sensors_d5) * np.log(2.0 * np.pi * (sigma_noise_d5**2))
            return -norm_const - misfit_p
        
        pinn_samples = run_mcmc_d5(pinn_log_likelihood, n_samples=10000, burnin=2000, seed=s)
        
        w1_dims = []
        for dim in range(d):
            w1_d = stats.wasserstein_distance(np.sort(pinn_samples[:, dim]), np.sort(exact_samples_d5[:, dim]))
            w1_dims.append(w1_d)
        w1_mean = float(np.mean(w1_dims))
        
        rec = {
            "pde": "heat_d5",
            "tier": lvl["name"],
            "seed": s,
            "e_global": e_global,
            "e_posterior": e_posterior,
            "l1_delta_loglik": l1_delta_loglik,
            "w1_mean": w1_mean,
            "w1_dim0": w1_dims[0],
            "w1_dim1": w1_dims[1],
            "w1_dim2": w1_dims[2],
            "w1_dim3": w1_dims[3],
            "w1_dim4": w1_dims[4]
        }
        d5_results.append(rec)
        print(f"  Model d=5 Tier={lvl['name']}, Seed={s} -> Eg={e_global:.4f}, Ep={e_posterior:.4f}, Lik={l1_delta_loglik:.4f}, W1={w1_mean:.6f}", flush=True)

df_d5 = pd.DataFrame(d5_results)
df_d5.to_csv(os.path.join(output_dir, "heat_d5_stress_test_results.csv"), index=False)

w1_d5 = df_d5["w1_mean"].values
eg_d5 = df_d5["e_global"].values
ep_d5 = df_d5["e_posterior"].values
lik_d5 = df_d5["l1_delta_loglik"].values

r_eg_d5 = float(np.corrcoef(eg_d5, w1_d5)[0, 1])
r_ep_d5 = float(np.corrcoef(ep_d5, w1_d5)[0, 1])
r_lik_d5 = float(np.corrcoef(lik_d5, w1_d5)[0, 1])
r_collin_d5 = float(np.corrcoef(eg_d5, ep_d5)[0, 1])
r_collin_lik_d5 = float(np.corrcoef(eg_d5, lik_d5)[0, 1])

t_williams_ep_d5, p_williams_ep_d5 = williams_test(r_ep_d5, r_eg_d5, r_collin_d5, len(w1_d5))
t_williams_lik_d5, p_williams_lik_d5 = williams_test(r_lik_d5, r_eg_d5, r_collin_lik_d5, len(w1_d5))

print("\n[d=5 HIGHER-DIMENSIONAL STRESS TEST SUMMARY]")
print(f"  N = {len(w1_d5)} models (4 convergence tiers x 5 random seeds)")
print(f"  Pearson r(E_global, W1)       = {r_eg_d5:.4f}")
print(f"  Pearson r(E_posterior, W1)    = {r_ep_d5:.4f} (Williams vs Eg: t={t_williams_ep_d5:.2f}, p={p_williams_ep_d5:.4e})")
print(f"  Pearson r(Likelihood, W1)     = {r_lik_d5:.4f} (Williams vs Eg: t={t_williams_lik_d5:.2f}, p={p_williams_lik_d5:.4e})")

d5_summary = {
    "n_models": len(w1_d5),
    "dimension": 5,
    "r_global": r_eg_d5,
    "r_posterior": r_ep_d5,
    "r_loglik": r_lik_d5,
    "williams_ep_eg_t": t_williams_ep_d5,
    "williams_ep_eg_p": p_williams_ep_d5,
    "williams_lik_eg_t": t_williams_lik_d5,
    "williams_lik_eg_p": p_williams_lik_d5
}

with open(os.path.join(output_dir, "heat_d5_summary.json"), "w") as f:
    json.dump(d5_summary, f, indent=2)

print("\nAUDIT COMPLETE.")
