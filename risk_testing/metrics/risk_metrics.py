"""
Risk metrics computed over Monte Carlo simulated path ensembles.

All metrics operate on a SimulationResult object containing
(n_paths, n_steps+1) price paths and (n_paths, n_steps) log returns.
"""

import numpy as np

from risk_testing.generators.base import SimulationResult


class RiskMetrics:
    """Compute risk metrics from Monte Carlo simulation results.

    Args:
        result: SimulationResult from any path generator.
        trading_days: Trading days per year for annualization.
        risk_free_rate: Annual risk-free rate (default 0.05 = 5%).
    """

    def __init__(
        self,
        result: SimulationResult,
        trading_days: int = 252,
        risk_free_rate: float = 0.05,
    ):
        self.result = result
        self.trading_days = trading_days
        self.risk_free_rate = risk_free_rate
        self._daily_rf = risk_free_rate / trading_days

    # ------------------------------------------------------------------
    # Value at Risk
    # ------------------------------------------------------------------

    def var(self, confidence: float = 0.95) -> float:
        """Value at Risk at given confidence level.

        Returns the loss threshold such that losses exceed this value
        with probability (1 - confidence). Expressed as a positive number
        representing the loss.

        For 95% VaR: "There is a 5% chance of losing more than this amount."
        """
        total_returns = self.result.total_returns
        return float(-np.percentile(total_returns, 100 * (1 - confidence)))

    def var_dollar(self, confidence: float = 0.95, portfolio_value: float = 1e6) -> float:
        """Dollar Value at Risk."""
        return self.var(confidence) * portfolio_value

    # ------------------------------------------------------------------
    # Conditional Value at Risk (Expected Shortfall)
    # ------------------------------------------------------------------

    def cvar(self, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall) at given confidence level.

        Average loss in the worst (1 - confidence) fraction of scenarios.
        Always >= VaR. More sensitive to tail shape than VaR.
        """
        total_returns = self.result.total_returns
        cutoff = np.percentile(total_returns, 100 * (1 - confidence))
        tail_returns = total_returns[total_returns <= cutoff]
        if len(tail_returns) == 0:
            return self.var(confidence)
        return float(-np.mean(tail_returns))

    def cvar_dollar(self, confidence: float = 0.95, portfolio_value: float = 1e6) -> float:
        """Dollar Conditional VaR."""
        return self.cvar(confidence) * portfolio_value

    # ------------------------------------------------------------------
    # Maximum Drawdown
    # ------------------------------------------------------------------

    def max_drawdowns(self) -> np.ndarray:
        """Maximum drawdown for each simulated path.

        Returns array of shape (n_paths,) with positive values
        representing the peak-to-trough decline.
        """
        paths = self.result.paths
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (running_max - paths) / running_max
        return np.max(drawdowns, axis=1)

    def mean_max_drawdown(self) -> float:
        """Average maximum drawdown across all paths."""
        return float(np.mean(self.max_drawdowns()))

    def worst_max_drawdown(self) -> float:
        """Worst-case maximum drawdown across all paths."""
        return float(np.max(self.max_drawdowns()))

    def drawdown_percentile(self, percentile: float = 95) -> float:
        """Percentile of maximum drawdown distribution."""
        return float(np.percentile(self.max_drawdowns(), percentile))

    # ------------------------------------------------------------------
    # Sharpe Ratio
    # ------------------------------------------------------------------

    def sharpe_ratios(self) -> np.ndarray:
        """Annualized Sharpe ratio for each simulated path."""
        daily_returns = self.result.returns
        excess = daily_returns - self._daily_rf
        mean_excess = np.mean(excess, axis=1)
        std_returns = np.std(daily_returns, axis=1, ddof=1)
        std_returns = np.maximum(std_returns, 1e-20)
        return mean_excess / std_returns * np.sqrt(self.trading_days)

    def mean_sharpe(self) -> float:
        return float(np.mean(self.sharpe_ratios()))

    def median_sharpe(self) -> float:
        return float(np.median(self.sharpe_ratios()))

    # ------------------------------------------------------------------
    # Sortino Ratio
    # ------------------------------------------------------------------

    def sortino_ratios(self) -> np.ndarray:
        """Annualized Sortino ratio for each simulated path."""
        daily_returns = self.result.returns
        excess = daily_returns - self._daily_rf
        mean_excess = np.mean(excess, axis=1)
        downside = np.where(excess < 0, excess, 0)
        downside_std = np.sqrt(np.mean(downside**2, axis=1))
        downside_std = np.maximum(downside_std, 1e-20)
        return mean_excess / downside_std * np.sqrt(self.trading_days)

    def mean_sortino(self) -> float:
        return float(np.mean(self.sortino_ratios()))

    # ------------------------------------------------------------------
    # Return Distribution
    # ------------------------------------------------------------------

    def return_percentiles(
        self, percentiles: list[float] | None = None
    ) -> dict[str, float]:
        """Percentiles of total return distribution across paths."""
        if percentiles is None:
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        total_ret = self.result.total_returns
        result = {}
        for p in percentiles:
            result[f"p{int(p)}"] = float(np.percentile(total_ret, p))
        result["mean"] = float(np.mean(total_ret))
        result["std"] = float(np.std(total_ret))
        result["min"] = float(np.min(total_ret))
        result["max"] = float(np.max(total_ret))
        return result

    # ------------------------------------------------------------------
    # Probability Metrics
    # ------------------------------------------------------------------

    def prob_profit(self) -> float:
        """Probability of positive total return."""
        return float(np.mean(self.result.total_returns > 0))

    def prob_loss_exceeds(self, threshold: float = 0.10) -> float:
        """Probability that loss exceeds given threshold (e.g., 10%)."""
        return float(np.mean(self.result.total_returns < -threshold))

    def prob_gain_exceeds(self, threshold: float = 0.20) -> float:
        """Probability that gain exceeds given threshold (e.g., 20%)."""
        return float(np.mean(self.result.total_returns > threshold))

    # ------------------------------------------------------------------
    # Summary Report
    # ------------------------------------------------------------------

    def summary(self, confidence: float = 0.95) -> dict:
        """Comprehensive risk summary across all metrics."""
        return {
            "generator": self.result.metadata.get("generator", "unknown"),
            "n_paths": self.result.n_paths,
            "n_steps": self.result.n_steps,
            "var": {
                "confidence": confidence,
                "var": self.var(confidence),
                "cvar": self.cvar(confidence),
            },
            "drawdown": {
                "mean_max": self.mean_max_drawdown(),
                "worst": self.worst_max_drawdown(),
                "p95": self.drawdown_percentile(95),
            },
            "sharpe": {
                "mean": self.mean_sharpe(),
                "median": self.median_sharpe(),
            },
            "sortino": {
                "mean": self.mean_sortino(),
            },
            "returns": self.return_percentiles(),
            "probabilities": {
                "profit": self.prob_profit(),
                "loss_gt_10pct": self.prob_loss_exceeds(0.10),
                "loss_gt_20pct": self.prob_loss_exceeds(0.20),
                "gain_gt_20pct": self.prob_gain_exceeds(0.20),
            },
        }
