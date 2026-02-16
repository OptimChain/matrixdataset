"""
Cross-generator path analytics for comparing Monte Carlo models.

Computes distributional tests and diagnostics to evaluate whether
different generators produce meaningfully different risk profiles.
"""

import numpy as np

from risk_testing.generators.base import SimulationResult
from risk_testing.metrics.risk_metrics import RiskMetrics


class PathAnalytics:
    """Compare simulation results across different path generators.

    Intended usage:
        results = {
            "GBM": gbm_gen.simulate(...),
            "HMM": hmm_gen.simulate(...),
            "GARCH": garch_gen.simulate(...),
            "Bootstrap": boot_gen.simulate(...),
        }
        analytics = PathAnalytics(results)
        comparison = analytics.compare()
    """

    def __init__(
        self,
        results: dict[str, SimulationResult],
        trading_days: int = 252,
        risk_free_rate: float = 0.05,
    ):
        self.results = results
        self.trading_days = trading_days
        self.risk_free_rate = risk_free_rate

    def compare(self, confidence: float = 0.95) -> dict:
        """Generate comparative risk report across all generators."""
        comparison = {}
        for name, result in self.results.items():
            rm = RiskMetrics(result, self.trading_days, self.risk_free_rate)
            comparison[name] = rm.summary(confidence)
        return comparison

    def tail_divergence(self, percentile: float = 5.0) -> dict[str, float]:
        """Compare left-tail severity across generators.

        Returns the Pth percentile of total returns for each generator.
        Lower values indicate heavier left tails / worse tail risk.
        """
        divergence = {}
        for name, result in self.results.items():
            total_ret = result.total_returns
            divergence[name] = float(np.percentile(total_ret, percentile))
        return divergence

    def volatility_of_volatility(self) -> dict[str, float]:
        """Compare vol-of-vol across generators.

        Higher vol-of-vol indicates more volatility clustering.
        GBM should have ~0 vol-of-vol. GARCH and HMM should have higher.
        """
        results = {}
        for name, sim_result in self.results.items():
            # Compute rolling 20-day realized vol for each path
            daily_ret = sim_result.returns
            if daily_ret.shape[1] < 20:
                results[name] = 0.0
                continue

            # Use non-overlapping windows for independence
            n_windows = daily_ret.shape[1] // 20
            truncated = daily_ret[:, : n_windows * 20]
            windowed = truncated.reshape(
                sim_result.n_paths, n_windows, 20
            )
            rolling_vol = np.std(windowed, axis=2, ddof=1)
            # Vol-of-vol: std of rolling vol / mean of rolling vol
            mean_vol = np.mean(rolling_vol, axis=1)
            std_vol = np.std(rolling_vol, axis=1)
            vov = std_vol / np.maximum(mean_vol, 1e-20)
            results[name] = float(np.mean(vov))
        return results

    def kurtosis_comparison(self) -> dict[str, float]:
        """Compare excess kurtosis of daily returns across generators.

        Normal returns have kurtosis = 3 (excess = 0).
        Higher kurtosis indicates fatter tails.
        GBM: ~0, GARCH: > 0, HMM: > 0, Bootstrap: matches historical.
        """
        results = {}
        for name, sim_result in self.results.items():
            flat_returns = sim_result.returns.flatten()
            mu = np.mean(flat_returns)
            std = np.std(flat_returns, ddof=1)
            if std < 1e-20:
                results[name] = 0.0
                continue
            kurt = float(np.mean(((flat_returns - mu) / std) ** 4) - 3.0)
            results[name] = kurt
        return results

    def skewness_comparison(self) -> dict[str, float]:
        """Compare skewness of daily returns across generators."""
        results = {}
        for name, sim_result in self.results.items():
            flat_returns = sim_result.returns.flatten()
            mu = np.mean(flat_returns)
            std = np.std(flat_returns, ddof=1)
            if std < 1e-20:
                results[name] = 0.0
                continue
            skew = float(np.mean(((flat_returns - mu) / std) ** 3))
            results[name] = skew
        return results

    def parameter_summary(self) -> dict[str, dict]:
        """Summarize calibrated parameters for each generator."""
        return {
            name: result.metadata for name, result in self.results.items()
        }
