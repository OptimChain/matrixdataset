"""
Geometric Brownian Motion (GBM) path generator.

Model:
    dS = mu * S * dt + sigma * S * dW

Exact solution:
    S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

Parameters:
    mu    - annualized drift (expected return)
    sigma - annualized volatility

This is the simplest parametric model. It assumes:
- Constant drift and volatility
- Independent, identically distributed increments
- Log-normal price distribution

Limitations: no fat tails, no volatility clustering, no regime shifts.
"""

from typing import Any

import numpy as np

from risk_testing.generators.base import PathGenerator, SimulationResult


class GBMGenerator(PathGenerator):
    """Geometric Brownian Motion path generator.

    Args:
        trading_days: Number of trading days per year (252 for equities,
            365 for crypto).
    """

    def __init__(self, trading_days: int = 252):
        self.trading_days = trading_days
        self._mu: float | None = None
        self._sigma: float | None = None
        self._initial_price: float | None = None

    def calibrate(self, prices: np.ndarray, **kwargs) -> None:
        """Calibrate mu and sigma from historical closing prices.

        Uses simple close-to-close estimator:
            mu    = mean(log_returns) * trading_days
            sigma = std(log_returns) * sqrt(trading_days)
        """
        prices = self._validate_prices(prices)
        log_ret = self._log_returns(prices)

        self._mu = float(np.mean(log_ret) * self.trading_days)
        self._sigma = float(np.std(log_ret, ddof=1) * np.sqrt(self.trading_days))
        self._initial_price = float(prices[-1])

    def simulate(
        self,
        n_paths: int = 1000,
        n_steps: int = 252,
        seed: int | None = 42,
    ) -> SimulationResult:
        if self._mu is None or self._sigma is None:
            raise RuntimeError("Must call calibrate() before simulate()")

        rng = np.random.default_rng(seed)
        dt = 1.0 / self.trading_days

        drift = (self._mu - 0.5 * self._sigma**2) * dt
        diffusion = self._sigma * np.sqrt(dt)

        Z = rng.standard_normal((n_paths, n_steps))
        log_increments = drift + diffusion * Z

        # Cumulative sum in log space for numerical stability
        log_paths = np.cumsum(log_increments, axis=1)
        log_paths = np.column_stack([np.zeros(n_paths), log_paths])

        paths = self._initial_price * np.exp(log_paths)

        return SimulationResult(
            paths=paths,
            returns=log_increments,
            metadata={
                "generator": "GBM",
                "mu": self._mu,
                "sigma": self._sigma,
                "initial_price": self._initial_price,
                "trading_days": self.trading_days,
                "n_paths": n_paths,
                "n_steps": n_steps,
            },
        )

    @property
    def params(self) -> dict[str, Any]:
        return {
            "mu": self._mu,
            "sigma": self._sigma,
            "initial_price": self._initial_price,
            "trading_days": self.trading_days,
        }
