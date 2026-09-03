"""
Unit and integration tests for experiments/ research suite
"""

import unittest
import os
import json
import numpy as np
import pandas as pd


class TestResearchProgram(unittest.TestCase):
    def test_phase1_artifacts_exist_and_valid(self):
        json_path = os.path.join("results", "phase1_baseline", "phase1_baseline_results.json")
        csv_path = os.path.join("results", "phase1_baseline", "phase1_baseline_summary.csv")
        npz_path = os.path.join("results", "phase1_baseline", "phase1_baseline_realizations.npz")
        
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(npz_path))
        
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 10)
        self.assertGreater(df["bfr"].mean(), 1.0)
        self.assertLess(df["w1_research"].mean(), 0.01)

    def test_phase2_accuracy_sweep_artifacts(self):
        csv_path = os.path.join("results", "phase2_accuracy_sweep", "phase2_accuracy_sweep_summary.csv")
        self.assertTrue(os.path.exists(csv_path))
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 6)
        # Check that Level 1 has high error and Level 6 has low error
        self.assertGreater(df.iloc[0]["mean_rel_l2_pct"], 50.0)
        self.assertLess(df.iloc[-1]["mean_rel_l2_pct"], 5.0)

    def test_phase3_error_localization_artifacts(self):
        json_path = os.path.join("results", "phase3_error_localization", "phase3_localization_results.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path) as f:
            data = json.load(f)
        self.assertIn("experiments", data)
        self.assertIn("curves", data)
        self.assertEqual(len(data["experiments"]), 3)

    def test_phase5_noise_sweep_artifacts(self):
        csv_path = os.path.join("results", "phase5_noise_sweep", "phase5_noise_sweep_summary.csv")
        self.assertTrue(os.path.exists(csv_path))
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 6)
        # Noise modulation: low noise has high BFR, high noise has low BFR
        self.assertGreater(df.iloc[0]["mean_bfr"], df.iloc[-1]["mean_bfr"])

    def test_phase6_sensor_sweep_artifacts(self):
        csv_path = os.path.join("results", "phase6_sensor_sweep", "phase6_sensor_sweep_summary.csv")
        self.assertTrue(os.path.exists(csv_path))
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 4)
        # Higher sensor count M -> lower posterior std
        self.assertGreater(df.iloc[0]["mean_exact_std"], df.iloc[-1]["mean_exact_std"])

    def test_all_12_publication_figures_exist(self):
        for i in range(1, 13):
            # Check for pattern fig{i}_*.png
            matching = [f for f in os.listdir("results/research_figures") if f.startswith(f"fig{i}_")]
            self.assertEqual(len(matching), 1, f"Missing figure {i}")


if __name__ == "__main__":
    unittest.main()
