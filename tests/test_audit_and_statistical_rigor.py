import pytest
import numpy as np
import scipy.stats as stats
import pandas as pd


def williams_test(r13, r23, r12, n):
    """
    Williams (1959) test for comparing two dependent correlations r13 and r23 sharing variable 3.
    """
    det_R = 1.0 - r12**2 - r13**2 - r23**2 + 2.0 * r12 * r13 * r23
    r_bar = 0.5 * (r13 + r23)
    num = (r13 - r23) * np.sqrt((n - 1) * (1.0 + r12))
    denom_sq = (2.0 * (n - 1) / (n - 3)) * det_R + (r_bar**2) * ((1.0 - r12)**3)
    if denom_sq <= 0:
        return 0.0, 1.0
    t_stat = num / np.sqrt(denom_sq)
    df = n - 3
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df))
    return float(t_stat), float(p_val)


def bonferroni_correction(p_values):
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]


def benjamini_hochberg_fdr(p_values):
    m = len(p_values)
    p_arr = np.array(p_values)
    order = np.argsort(p_arr)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    q_vals = p_arr * m / ranks
    sorted_q = np.minimum.accumulate(q_vals[order][::-1])[::-1]
    final_q = np.empty_like(p_arr)
    final_q[order] = np.minimum(1.0, sorted_q)
    return final_q.tolist()


class TestStatisticalRigor:
    def test_williams_formula_known_values(self):
        t, p = williams_test(0.8878, 0.8747, 0.9987, 60)
        assert t > 4.5
        assert p < 1e-4

    def test_williams_null_symmetry(self):
        t, p = williams_test(0.75, 0.75, 0.80, 50)
        assert np.isclose(t, 0.0)
        assert np.isclose(p, 1.0)

    def test_williams_d5_negative_crossover(self):
        # In d=5, r(Likelihood, SW1) = 0.5325, r(E_global, SW1) = 0.8289, r12 = 0.6861, N=20
        # Williams test yields negative t = -2.7147, p = 0.0147
        # Confirms that likelihood is statistically significantly INFERIOR to global error in d=5.
        t, p = williams_test(0.5325269, 0.8288521, 0.686105, 20)
        assert t < -2.5
        assert np.isclose(t, -2.7147, atol=1e-3)
        assert np.isclose(p, 0.0147, atol=1e-3)

    def test_bonferroni_fdr_properties(self):
        p_raw = [0.0001, 0.001, 0.01, 0.05, 0.50]
        p_bonf = bonferroni_correction(p_raw)
        p_fdr = benjamini_hochberg_fdr(p_raw)
        for pb, pf in zip(p_bonf, p_fdr):
            assert pb >= pf - 1e-12
        assert p_bonf[0] == 0.0005

    def test_binomial_win_rate_significance(self):
        k, n = 20, 24
        # One-sided test for directional hypothesis (greater predictive power)
        res_greater = stats.binomtest(k, n, p=0.5, alternative='greater')
        assert np.isclose(res_greater.pvalue, 0.000772, atol=1e-5)
        
        # Two-sided test
        res_two_sided = stats.binomtest(k, n, p=0.5, alternative='two-sided')
        assert res_two_sided.pvalue < 0.005
        assert np.isclose(res_two_sided.pvalue, 0.001544, atol=1e-5)

        # 17 wins out of 24 trials
        res17 = stats.binomtest(17, n, p=0.5, alternative='greater')
        assert res17.pvalue < 0.05
        assert np.isclose(res17.pvalue, 0.031957, atol=1e-5)

    def test_optimal_coupling_inequality_on_synthetic_distributions(self):
        np.random.seed(42)
        grid = np.linspace(0.1, 2.0, 100)
        diam = grid[-1] - grid[0]

        for _ in range(20):
            p1 = np.random.dirichlet(np.ones(100))
            p2 = np.random.dirichlet(np.ones(100))
            
            d_tv = 0.5 * np.sum(np.abs(p1 - p2))
            w1 = stats.wasserstein_distance(grid, grid, u_weights=p1, v_weights=p2)
            assert w1 <= diam * d_tv + 1e-10

    def test_categorical_ancova_structure(self):
        n_samples = 60
        n_tiers = 6
        n_diagnostic = 1
        df_resid = n_samples - n_tiers - n_diagnostic
        assert df_resid == 53
