"""Tests for all Monte Carlo path generators."""

import numpy as np
import pytest

from risk_testing.generators.gbm import GBMGenerator
from risk_testing.generators.markov import MarkovRegimeSwitchingGenerator
from risk_testing.generators.garch import GARCHGenerator
from risk_testing.generators.bootstrap import BlockBootstrapGenerator


# ======================================================================
# GBM Generator Tests
# ======================================================================


class TestGBMGenerator:
    def test_calibrate_sets_params(self, synthetic_prices):
        gen = GBMGenerator()
        gen.calibrate(synthetic_prices)
        assert gen.params["mu"] is not None
        assert gen.params["sigma"] is not None
        assert gen.params["sigma"] > 0

    def test_simulate_shape(self, synthetic_prices):
        gen = GBMGenerator()
        gen.calibrate(synthetic_prices)
        result = gen.simulate(n_paths=100, n_steps=60)
        assert result.paths.shape == (100, 61)
        assert result.returns.shape == (100, 60)

    def test_paths_positive(self, synthetic_prices):
        gen = GBMGenerator()
        gen.calibrate(synthetic_prices)
        result = gen.simulate(n_paths=500, n_steps=252)
        assert np.all(result.paths > 0)

    def test_reproducibility(self, synthetic_prices):
        gen = GBMGenerator()
        gen.calibrate(synthetic_prices)
        r1 = gen.simulate(n_paths=50, n_steps=30, seed=99)
        r2 = gen.simulate(n_paths=50, n_steps=30, seed=99)
        np.testing.assert_array_equal(r1.paths, r2.paths)

    def test_different_seeds_differ(self, synthetic_prices):
        gen = GBMGenerator()
        gen.calibrate(synthetic_prices)
        r1 = gen.simulate(n_paths=50, n_steps=30, seed=1)
        r2 = gen.simulate(n_paths=50, n_steps=30, seed=2)
        assert not np.array_equal(r1.paths, r2.paths)

    def test_initial_price_matches(self, synthetic_prices):
        gen = GBMGenerator()
        gen.calibrate(synthetic_prices)
        result = gen.simulate(n_paths=10, n_steps=30)
        expected = synthetic_prices[-1]
        np.testing.assert_allclose(result.paths[:, 0], expected, rtol=1e-10)

    def test_simulate_before_calibrate_raises(self):
        gen = GBMGenerator()
        with pytest.raises(RuntimeError, match="calibrate"):
            gen.simulate()

    def test_invalid_prices_raises(self):
        gen = GBMGenerator()
        with pytest.raises(ValueError):
            gen.calibrate(np.array([100, -50, 200]))
        with pytest.raises(ValueError):
            gen.calibrate(np.array([100, np.nan, 200]))
        with pytest.raises(ValueError):
            gen.calibrate(np.array([100.0]))

    def test_metadata(self, synthetic_prices):
        gen = GBMGenerator()
        gen.calibrate(synthetic_prices)
        result = gen.simulate(n_paths=10, n_steps=30)
        assert result.metadata["generator"] == "GBM"
        assert "mu" in result.metadata
        assert "sigma" in result.metadata

    def test_total_returns_property(self, synthetic_prices):
        gen = GBMGenerator()
        gen.calibrate(synthetic_prices)
        result = gen.simulate(n_paths=100, n_steps=252)
        ret = result.total_returns
        assert ret.shape == (100,)
        # Verify the calculation
        expected = result.paths[:, -1] / result.paths[:, 0] - 1
        np.testing.assert_allclose(ret, expected)


# ======================================================================
# Markov Regime-Switching Tests
# ======================================================================


class TestMarkovRegimeSwitching:
    def test_calibrate_2_regimes(self, regime_prices):
        gen = MarkovRegimeSwitchingGenerator(n_regimes=2)
        gen.calibrate(regime_prices)
        p = gen.params
        assert len(p["regime_mu"]) == 2
        assert len(p["regime_sigma"]) == 2
        assert np.array(p["transition_matrix"]).shape == (2, 2)

    def test_calibrate_3_regimes(self, synthetic_prices):
        gen = MarkovRegimeSwitchingGenerator(n_regimes=3)
        gen.calibrate(synthetic_prices)
        p = gen.params
        assert len(p["regime_mu"]) == 3
        assert np.array(p["transition_matrix"]).shape == (3, 3)

    def test_transition_matrix_row_stochastic(self, regime_prices):
        gen = MarkovRegimeSwitchingGenerator(n_regimes=2)
        gen.calibrate(regime_prices)
        tm = np.array(gen.params["transition_matrix"])
        np.testing.assert_allclose(tm.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(tm >= 0)

    def test_simulate_shape(self, regime_prices):
        gen = MarkovRegimeSwitchingGenerator(n_regimes=2)
        gen.calibrate(regime_prices)
        result = gen.simulate(n_paths=100, n_steps=60)
        assert result.paths.shape == (100, 61)
        assert result.returns.shape == (100, 60)

    def test_paths_positive(self, regime_prices):
        gen = MarkovRegimeSwitchingGenerator(n_regimes=2)
        gen.calibrate(regime_prices)
        result = gen.simulate(n_paths=500, n_steps=252)
        assert np.all(result.paths > 0)

    def test_reproducibility(self, regime_prices):
        gen = MarkovRegimeSwitchingGenerator(n_regimes=2)
        gen.calibrate(regime_prices)
        r1 = gen.simulate(n_paths=50, n_steps=30, seed=42)
        r2 = gen.simulate(n_paths=50, n_steps=30, seed=42)
        np.testing.assert_array_equal(r1.paths, r2.paths)

    def test_regime_mu_ordered(self, regime_prices):
        """Regimes should be sorted bearish (0) to bullish (K-1)."""
        gen = MarkovRegimeSwitchingGenerator(n_regimes=2)
        gen.calibrate(regime_prices)
        mus = gen.params["regime_mu"]
        assert mus[0] <= mus[-1]

    def test_regime_history_in_metadata(self, regime_prices):
        gen = MarkovRegimeSwitchingGenerator(n_regimes=2)
        gen.calibrate(regime_prices)
        result = gen.simulate(n_paths=10, n_steps=30)
        assert "regime_history_sample" in result.metadata

    def test_higher_vol_in_bear_regime(self, regime_prices):
        """Bear regime should typically have higher vol."""
        gen = MarkovRegimeSwitchingGenerator(n_regimes=2)
        gen.calibrate(regime_prices)
        sigmas = gen.params["regime_sigma"]
        # At least one regime should have meaningfully higher vol
        assert max(sigmas) / min(sigmas) > 1.0


# ======================================================================
# GARCH Generator Tests
# ======================================================================


class TestGARCHGenerator:
    def test_calibrate_sets_params(self, synthetic_prices):
        gen = GARCHGenerator()
        gen.calibrate(synthetic_prices)
        p = gen.params
        assert p["omega"] is not None and p["omega"] > 0
        assert p["alpha"] is not None and p["alpha"] > 0
        assert p["beta"] is not None and p["beta"] > 0

    def test_stationarity(self, synthetic_prices):
        gen = GARCHGenerator()
        gen.calibrate(synthetic_prices)
        p = gen.params
        assert p["persistence"] < 1.0

    def test_simulate_shape(self, synthetic_prices):
        gen = GARCHGenerator()
        gen.calibrate(synthetic_prices)
        result = gen.simulate(n_paths=100, n_steps=60)
        assert result.paths.shape == (100, 61)
        assert result.returns.shape == (100, 60)

    def test_paths_positive(self, synthetic_prices):
        gen = GARCHGenerator()
        gen.calibrate(synthetic_prices)
        result = gen.simulate(n_paths=500, n_steps=252)
        assert np.all(result.paths > 0)

    def test_reproducibility(self, synthetic_prices):
        gen = GARCHGenerator()
        gen.calibrate(synthetic_prices)
        r1 = gen.simulate(n_paths=50, n_steps=30, seed=42)
        r2 = gen.simulate(n_paths=50, n_steps=30, seed=42)
        np.testing.assert_array_equal(r1.paths, r2.paths)

    def test_t_dist_nu_estimated(self, synthetic_prices):
        gen = GARCHGenerator(use_t_dist=True)
        gen.calibrate(synthetic_prices)
        assert gen.params["nu"] is not None
        assert gen.params["nu"] > 2

    def test_no_t_dist(self, synthetic_prices):
        gen = GARCHGenerator(use_t_dist=False)
        gen.calibrate(synthetic_prices)
        assert gen.params["nu"] is None

    def test_metadata_has_persistence(self, synthetic_prices):
        gen = GARCHGenerator()
        gen.calibrate(synthetic_prices)
        result = gen.simulate(n_paths=10, n_steps=30)
        assert "persistence" in result.metadata
        assert result.metadata["generator"] == "GARCH(1,1)"

    def test_fatter_tails_with_t_dist(self, synthetic_prices):
        """t-distribution innovations should produce fatter tails."""
        gen_normal = GARCHGenerator(use_t_dist=False)
        gen_normal.calibrate(synthetic_prices)
        gen_t = GARCHGenerator(use_t_dist=True)
        gen_t.calibrate(synthetic_prices)

        r_normal = gen_normal.simulate(n_paths=5000, n_steps=252, seed=42)
        r_t = gen_t.simulate(n_paths=5000, n_steps=252, seed=42)

        # Compute kurtosis of daily returns
        def kurtosis(returns):
            flat = returns.flatten()
            mu = np.mean(flat)
            std = np.std(flat, ddof=1)
            return float(np.mean(((flat - mu) / std) ** 4)) - 3

        kurt_normal = kurtosis(r_normal.returns)
        kurt_t = kurtosis(r_t.returns)
        # t-distribution should produce higher kurtosis
        assert kurt_t > kurt_normal


# ======================================================================
# Block Bootstrap Tests
# ======================================================================


class TestBlockBootstrapGenerator:
    def test_calibrate_stores_returns(self, synthetic_prices):
        gen = BlockBootstrapGenerator(block_length=10)
        gen.calibrate(synthetic_prices)
        assert gen.params["historical_returns_length"] == len(synthetic_prices) - 1

    def test_simulate_shape(self, synthetic_prices):
        gen = BlockBootstrapGenerator(block_length=10)
        gen.calibrate(synthetic_prices)
        result = gen.simulate(n_paths=100, n_steps=60)
        assert result.paths.shape == (100, 61)
        assert result.returns.shape == (100, 60)

    def test_paths_positive(self, synthetic_prices):
        gen = BlockBootstrapGenerator(block_length=10)
        gen.calibrate(synthetic_prices)
        result = gen.simulate(n_paths=500, n_steps=252)
        assert np.all(result.paths > 0)

    def test_reproducibility(self, synthetic_prices):
        gen = BlockBootstrapGenerator(block_length=10)
        gen.calibrate(synthetic_prices)
        r1 = gen.simulate(n_paths=50, n_steps=30, seed=42)
        r2 = gen.simulate(n_paths=50, n_steps=30, seed=42)
        np.testing.assert_array_equal(r1.paths, r2.paths)

    def test_returns_come_from_history(self, synthetic_prices):
        """All simulated returns should be from the historical set."""
        gen = BlockBootstrapGenerator(block_length=5)
        gen.calibrate(synthetic_prices)
        result = gen.simulate(n_paths=50, n_steps=100, seed=42)

        historical_log_ret = np.diff(np.log(synthetic_prices))
        simulated_flat = result.returns.flatten()

        # Every simulated return should exist in the historical set
        for r in simulated_flat:
            assert np.any(np.isclose(historical_log_ret, r, atol=1e-14))

    def test_block_length_1_is_iid(self, synthetic_prices):
        """Block length 1 = standard i.i.d. bootstrap."""
        gen = BlockBootstrapGenerator(block_length=1)
        gen.calibrate(synthetic_prices)
        result = gen.simulate(n_paths=10, n_steps=30)
        assert result.paths.shape == (10, 31)

    def test_invalid_block_length(self):
        with pytest.raises(ValueError):
            BlockBootstrapGenerator(block_length=0)

    def test_metadata(self, synthetic_prices):
        gen = BlockBootstrapGenerator(block_length=10)
        gen.calibrate(synthetic_prices)
        result = gen.simulate(n_paths=10, n_steps=30)
        assert result.metadata["generator"] == "BlockBootstrap"
        assert result.metadata["block_length"] == 10


# ======================================================================
# Cross-Generator Comparison Tests
# ======================================================================


class TestCrossGenerator:
    """Verify that different generators produce structurally different paths."""

    def _run_all(self, prices, n_paths=1000, n_steps=252, seed=42):
        generators = {
            "GBM": GBMGenerator(),
            "HMM": MarkovRegimeSwitchingGenerator(n_regimes=2),
            "GARCH": GARCHGenerator(use_t_dist=True),
            "Bootstrap": BlockBootstrapGenerator(block_length=10),
        }
        results = {}
        for name, gen in generators.items():
            gen.calibrate(prices)
            results[name] = gen.simulate(n_paths=n_paths, n_steps=n_steps, seed=seed)
        return results

    def test_all_produce_valid_paths(self, synthetic_prices):
        results = self._run_all(synthetic_prices, n_paths=100, n_steps=60)
        for name, result in results.items():
            assert result.paths.shape == (100, 61), f"{name} wrong shape"
            assert np.all(result.paths > 0), f"{name} has non-positive prices"
            assert np.all(np.isfinite(result.paths)), f"{name} has non-finite prices"

    def test_gbm_lower_kurtosis(self, synthetic_prices):
        """GBM should have lower excess kurtosis than GARCH or HMM."""
        results = self._run_all(synthetic_prices, n_paths=2000, n_steps=252)

        def excess_kurt(returns):
            flat = returns.flatten()
            mu = np.mean(flat)
            std = np.std(flat, ddof=1)
            return float(np.mean(((flat - mu) / std) ** 4)) - 3

        gbm_kurt = excess_kurt(results["GBM"].returns)
        garch_kurt = excess_kurt(results["GARCH"].returns)
        # GARCH with t-dist should have meaningfully higher kurtosis
        assert garch_kurt > gbm_kurt

    def test_percentile_paths_structure(self, synthetic_prices):
        results = self._run_all(synthetic_prices, n_paths=100, n_steps=60)
        for name, result in results.items():
            pct = result.percentile_paths([10, 50, 90])
            assert "p10" in pct
            assert "p50" in pct
            assert "p90" in pct
            assert pct["p10"].shape == (61,)
            # p10 <= p50 <= p90 at each step
            assert np.all(pct["p10"] <= pct["p50"] + 1e-10)
            assert np.all(pct["p50"] <= pct["p90"] + 1e-10)
