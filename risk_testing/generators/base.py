"""
Base classes for Monte Carlo path generators.

All generators produce a SimulationResult containing:
- paths: (n_paths, n_steps+1) array of price levels
- returns: (n_paths, n_steps) array of log returns
- metadata: dict of generator-specific calibration info
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation output."""

    paths: np.ndarray  # (n_paths, n_steps+1) price levels
    returns: np.ndarray  # (n_paths, n_steps) log returns
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_paths(self) -> int:
        return self.paths.shape[0]

    @property
    def n_steps(self) -> int:
        return self.paths.shape[1] - 1

    @property
    def initial_price(self) -> float:
        return float(self.paths[0, 0])

    @property
    def final_prices(self) -> np.ndarray:
        return self.paths[:, -1]

    @property
    def total_returns(self) -> np.ndarray:
        """Simple return from start to end for each path."""
        return self.paths[:, -1] / self.paths[:, 0] - 1.0

    def percentile_paths(
        self, percentiles: list[float] | None = None
    ) -> dict[str, np.ndarray]:
        """Compute percentile paths across simulations at each time step."""
        if percentiles is None:
            percentiles = [5, 10, 25, 50, 75, 90, 95]
        return {
            f"p{int(p)}": np.percentile(self.paths, p, axis=0) for p in percentiles
        }


class PathGenerator(ABC):
    """Abstract base class for Monte Carlo path generators.

    Subclasses must implement:
        - calibrate(prices): fit model parameters from historical data
        - simulate(n_paths, n_steps, seed): generate forward price paths
        - params: property returning the fitted parameter dict
    """

    @abstractmethod
    def calibrate(self, prices: np.ndarray, **kwargs) -> None:
        """Fit generator parameters from historical price series.

        Args:
            prices: 1-D array of historical closing prices.
        """

    @abstractmethod
    def simulate(
        self,
        n_paths: int = 1000,
        n_steps: int = 252,
        seed: int | None = 42,
    ) -> SimulationResult:
        """Generate Monte Carlo price paths.

        Args:
            n_paths: Number of simulated trajectories.
            n_steps: Number of time steps (trading days) per path.
            seed: Random seed for reproducibility (None for random).

        Returns:
            SimulationResult with paths and returns arrays.
        """

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Return fitted model parameters as a dict."""

    def _validate_prices(self, prices: np.ndarray, min_length: int = 10) -> np.ndarray:
        """Validate and clean a price series."""
        prices = np.asarray(prices, dtype=np.float64)
        if prices.ndim != 1:
            raise ValueError(f"Expected 1-D price array, got shape {prices.shape}")
        if len(prices) < min_length:
            raise ValueError(
                f"Need at least {min_length} prices, got {len(prices)}"
            )
        if np.any(prices <= 0):
            raise ValueError("All prices must be positive")
        if np.any(~np.isfinite(prices)):
            raise ValueError("Prices contain NaN or Inf values")
        return prices

    def _log_returns(self, prices: np.ndarray) -> np.ndarray:
        """Compute log returns from a price series."""
        return np.diff(np.log(prices))
