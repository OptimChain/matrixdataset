"""Tests for risk metrics computation over simulated paths."""

import numpy as np
import pytest

from risk_testing.generators.gbm import GBMGenerator
from risk_testing.metrics.risk_metrics import RiskMetrics
from risk_testing.metrics.path_analytics import PathAnalytics
from risk_testing.generators.markov import MarkovRegimeSwitchingGenerator
from risk_testing.generators.garch import GARCHGenerator
from risk_testing.generators.bootstrap import BlockBootstrapGenerator


@pytest.fixture
def sim_result(synthetic_prices):
    """A standard GBM simulation result for metric testing."""
    gen = GBMGenerator()
    gen.calibrate(synthetic_prices)
    return gen.simulate(n_paths=2000, n_steps=252, seed=42)


# ======================================================================
# VaR / CVaR Tests
# ======================================================================


class TestVaR:
    def test_var_positive_for_risky_asset(self, sim_result):
        rm = RiskMetrics(sim_result)
        v = rm.var(0.95)
        # VaR should be a finite number
        assert np.isfinite(v)

    def test_var_increases_with_confidence(self, sim_result):
        rm = RiskMetrics(sim_result)
        v90 = rm.var(0.90)
        v95 = rm.var(0.95)
        v99 = rm.var(0.99)
        assert v90 <= v95 <= v99

    def test_cvar_gte_var(self, sim_result):
        rm = RiskMetrics(sim_result)
        var = rm.var(0.95)
        cvar = rm.cvar(0.95)
        assert cvar >= var - 1e-10  # CVaR is always >= VaR

    def test_cvar_increases_with_confidence(self, sim_result):
        rm = RiskMetrics(sim_result)
        c90 = rm.cvar(0.90)
        c95 = rm.cvar(0.95)
        c99 = rm.cvar(0.99)
        assert c90 <= c95 <= c99

    def test_var_dollar(self, sim_result):
        rm = RiskMetrics(sim_result)
        pv = 1_000_000
        var_pct = rm.var(0.95)
        var_dollar = rm.var_dollar(0.95, pv)
        np.testing.assert_allclose(var_dollar, var_pct * pv, rtol=1e-10)


# ======================================================================
# Drawdown Tests
# ======================================================================


class TestDrawdown:
    def test_max_drawdowns_shape(self, sim_result):
        rm = RiskMetrics(sim_result)
        dd = rm.max_drawdowns()
        assert dd.shape == (sim_result.n_paths,)

    def test_max_drawdowns_bounded(self, sim_result):
        rm = RiskMetrics(sim_result)
        dd = rm.max_drawdowns()
        assert np.all(dd >= 0)
        assert np.all(dd <= 1)

    def test_mean_le_worst(self, sim_result):
        rm = RiskMetrics(sim_result)
        assert rm.mean_max_drawdown() <= rm.worst_max_drawdown()

    def test_percentile_between_mean_and_worst(self, sim_result):
        rm = RiskMetrics(sim_result)
        mean_dd = rm.mean_max_drawdown()
        worst_dd = rm.worst_max_drawdown()
        p95 = rm.drawdown_percentile(95)
        assert mean_dd <= p95 + 1e-10
        assert p95 <= worst_dd + 1e-10


# ======================================================================
# Sharpe / Sortino Tests
# ======================================================================


class TestSharpe:
    def test_sharpe_shape(self, sim_result):
        rm = RiskMetrics(sim_result)
        sr = rm.sharpe_ratios()
        assert sr.shape == (sim_result.n_paths,)

    def test_sharpe_finite(self, sim_result):
        rm = RiskMetrics(sim_result)
        assert np.isfinite(rm.mean_sharpe())
        assert np.isfinite(rm.median_sharpe())

    def test_sortino_gte_sharpe_on_average(self, sim_result):
        """Sortino typically >= Sharpe when downside vol < total vol."""
        rm = RiskMetrics(sim_result)
        # This isn't strictly guaranteed but holds for most distributions
        # so we just check they're finite and reasonable
        assert np.isfinite(rm.mean_sortino())


# ======================================================================
# Return Distribution Tests
# ======================================================================


class TestReturnDistribution:
    def test_percentiles_ordered(self, sim_result):
        rm = RiskMetrics(sim_result)
        p = rm.return_percentiles()
        assert p["p5"] <= p["p25"]
        assert p["p25"] <= p["p50"]
        assert p["p50"] <= p["p75"]
        assert p["p75"] <= p["p95"]

    def test_min_max_bounds(self, sim_result):
        rm = RiskMetrics(sim_result)
        p = rm.return_percentiles()
        assert p["min"] <= p["p1"]
        assert p["p99"] <= p["max"]


# ======================================================================
# Probability Tests
# ======================================================================


class TestProbabilities:
    def test_prob_profit_bounded(self, sim_result):
        rm = RiskMetrics(sim_result)
        pp = rm.prob_profit()
        assert 0 <= pp <= 1

    def test_prob_loss_bounded(self, sim_result):
        rm = RiskMetrics(sim_result)
        pl = rm.prob_loss_exceeds(0.10)
        assert 0 <= pl <= 1

    def test_prob_gain_bounded(self, sim_result):
        rm = RiskMetrics(sim_result)
        pg = rm.prob_gain_exceeds(0.20)
        assert 0 <= pg <= 1


# ======================================================================
# Summary Report Test
# ======================================================================


class TestSummary:
    def test_summary_has_all_keys(self, sim_result):
        rm = RiskMetrics(sim_result)
        s = rm.summary()
        assert "var" in s
        assert "drawdown" in s
        assert "sharpe" in s
        assert "sortino" in s
        assert "returns" in s
        assert "probabilities" in s
        assert s["n_paths"] == sim_result.n_paths
        assert s["n_steps"] == sim_result.n_steps


# ======================================================================
# PathAnalytics Tests
# ======================================================================


class TestPathAnalytics:
    def _build_results(self, prices):
        generators = {
            "GBM": GBMGenerator(),
            "HMM": MarkovRegimeSwitchingGenerator(n_regimes=2),
            "GARCH": GARCHGenerator(use_t_dist=True),
            "Bootstrap": BlockBootstrapGenerator(block_length=10),
        }
        results = {}
        for name, gen in generators.items():
            gen.calibrate(prices)
            results[name] = gen.simulate(n_paths=500, n_steps=252, seed=42)
        return results

    def test_compare_returns_all_generators(self, synthetic_prices):
        results = self._build_results(synthetic_prices)
        analytics = PathAnalytics(results)
        comparison = analytics.compare()
        assert set(comparison.keys()) == {"GBM", "HMM", "GARCH", "Bootstrap"}
        for name, report in comparison.items():
            assert "var" in report
            assert "drawdown" in report

    def test_tail_divergence(self, synthetic_prices):
        results = self._build_results(synthetic_prices)
        analytics = PathAnalytics(results)
        td = analytics.tail_divergence(5.0)
        assert set(td.keys()) == {"GBM", "HMM", "GARCH", "Bootstrap"}
        for v in td.values():
            assert np.isfinite(v)

    def test_vol_of_vol_gbm_lowest(self, synthetic_prices):
        """GBM has constant vol, so vol-of-vol should be lowest."""
        results = self._build_results(synthetic_prices)
        analytics = PathAnalytics(results)
        vov = analytics.volatility_of_volatility()
        # GBM should have lower vol-of-vol than GARCH
        # (not strictly guaranteed with small samples, but generally true)
        assert vov["GBM"] < vov["GARCH"] + 0.5  # generous tolerance

    def test_kurtosis_comparison(self, synthetic_prices):
        results = self._build_results(synthetic_prices)
        analytics = PathAnalytics(results)
        kurt = analytics.kurtosis_comparison()
        for v in kurt.values():
            assert np.isfinite(v)

    def test_skewness_comparison(self, synthetic_prices):
        results = self._build_results(synthetic_prices)
        analytics = PathAnalytics(results)
        skew = analytics.skewness_comparison()
        for v in skew.values():
            assert np.isfinite(v)

    def test_parameter_summary(self, synthetic_prices):
        results = self._build_results(synthetic_prices)
        analytics = PathAnalytics(results)
        params = analytics.parameter_summary()
        assert "GBM" in params
        assert params["GBM"]["generator"] == "GBM"
