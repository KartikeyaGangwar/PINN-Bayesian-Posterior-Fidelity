"""
Cross-PDE Benchmark Execution Entry Point
=========================================
Executes the full 240-model cross-dynamical surrogate training and Bayesian evaluation
across Heat, Wave, Advection-Diffusion, and Viscous Burgers benchmark systems.
"""

import os
import sys
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.cross_pde.run_comprehensive_jcp_hardening import (
    run_symmetric_n60_campaign,
    run_multi_magnitude_localization,
    run_bfr_sensitivity_analysis,
    run_computational_cost_benchmark,
    generate_publication_figures_n60,
)

if __name__ == "__main__":
    t_start = time.time()
    print("Starting 240-model Cross-PDE Computational Benchmark...", flush=True)
    summaries, raw_df, traces = run_symmetric_n60_campaign()
    df_sweep = run_multi_magnitude_localization()
    df_bfr = run_bfr_sensitivity_analysis()
    cost_data = run_computational_cost_benchmark()
    generate_publication_figures_n60()
    print(f"\n=======================================================", flush=True)
    print(f"CROSS-PDE EXPERIMENTAL CAMPAIGN COMPLETED IN {time.time() - t_start:.2f}s", flush=True)
    print(f"=======================================================", flush=True)
