"""
Statistical ANCOVA and Higher-Dimensional Validation Entry Point
================================================================
Executes fixed-effects Categorical ANCOVA, multiplicity corrections,
and 5-dimensional parametric thermal diffusion validation.
"""

import os
import sys
import runpy

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

if __name__ == "__main__":
    runpy.run_module("experiments.cross_pde.run_audit_and_d5_stress_test", run_name="__main__")
