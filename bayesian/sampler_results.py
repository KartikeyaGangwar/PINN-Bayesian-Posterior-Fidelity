r"""
MCMC Sampler Results & Export Module (1D Heat Equation)
=======================================================
Description:
    Exports single-chain MCMC trajectories, summaries, and post-burnin arrays.
"""

import numpy as np
import os
import json
import csv
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from datetime import datetime

from .chain import MCMCChain
from .diagnostics import analyze_chain, plot_mcmc_diagnostics


@dataclass
class MCMCResult:
    """Dataclass holding single-chain MCMC summary results and metadata."""
    n_samples: int
    burn_in: int
    acceptance_rate: float
    alpha_mean: float
    alpha_std: float
    alpha_median: float
    alpha_cred_95: list
    alpha_ess: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def export_mcmc_results(
    chain: MCMCChain,
    burn_in: int = 2000,
    directory: str = "results",
    true_alpha: Optional[float] = None,
    seed: Optional[int] = 42
) -> MCMCResult:
    """
    Exports single-chain MCMC data files (CSV, JSON, NPZ) and generates diagnostic plots.
    """
    os.makedirs(directory, exist_ok=True)
    diag = analyze_chain(chain, burn_in=burn_in)
    now_str = datetime.now().isoformat()

    # 1. Export chain.csv
    chain_csv_path = os.path.join(directory, "chain.csv")
    with open(chain_csv_path, "w", newline="") as f:
        fieldnames = list(chain.records[0].to_dict().keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in chain.records:
            writer.writerow(rec.to_dict())

    # 2. Export posterior_samples.npz
    alpha_samples = chain.get_alpha_array(burn_in=burn_in)
    log_post_samples = chain.get_log_posterior_array(burn_in=burn_in)

    npz_path = os.path.join(directory, "posterior_samples.npz")
    np.savez_compressed(
        npz_path,
        alpha_samples=alpha_samples,
        log_posterior_samples=log_post_samples,
        burn_in=burn_in,
        total_samples=len(chain)
    )

    # 3. Construct MCMCResult & export summary.json
    a_stats = diag["alpha_stats"]
    result = MCMCResult(
        n_samples=len(chain),
        burn_in=burn_in,
        acceptance_rate=float(chain.acceptance_rate),
        alpha_mean=float(a_stats["mean"]),
        alpha_std=float(a_stats["std"]),
        alpha_median=float(a_stats["median"]),
        alpha_cred_95=list(a_stats["cred_95"]),
        alpha_ess=float(a_stats["ess"]),
        timestamp=now_str
    )

    summary_path = os.path.join(directory, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=4)

    # 4. Generate plots
    plot_mcmc_diagnostics(chain, burn_in=burn_in, true_alpha=true_alpha, output_dir=directory)

    return result
