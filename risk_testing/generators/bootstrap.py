"""
Block Bootstrap path generator.

Model:
    Non-parametric. Resamples contiguous blocks of historical log returns
    to construct synthetic forward paths.

Parameters:
    block_length - length of each resampled block (default 10 trading days)
    block_type   - "circular" (wraps around) or "stationary" (random block length)

vs parametric models (GBM, HMM, GARCH):
    - Zero distributional assumptions
    - Preserves empirical return distribution exactly (fat tails, skewness)
    - Preserves short-range autocorrelation within blocks
    - Cannot generate scenarios beyond the historical sample
    - No extrapolation to unseen regimes
    - Only 1-2 tuning parameters

This is useful as a "model-free" baseline to compare against parametric generators.
"""

from typing import Any

import numpy as np

from risk_testing.generators.base import PathGenerator, SimulationResult


class BlockBootstrapGenerator(PathGenerator):
    """Block bootstrap path generator.

    Args:
        block_length: Length of resampled blocks in trading days.
        circular: If True, treat the return series as circular
            (wraps around end to start).
        trading_days: Trading days per year.
    """

    def __init__(
        self,
        block_length: int = 10,
        circular: bool = True,
        trading_days: int = 252,
    ):
        if block_length < 1:
            raise ValueError("block_length must be >= 1")
        self.block_length = block_length
        self.circular = circular
        self.trading_days = trading_days

        self._log_returns: np.ndarray | None = None
        self._initial_price: float | None = None

    def calibrate(self, prices: np.ndarray, **kwargs) -> None:
        """Store historical log returns for resampling."""
        prices = self._validate_prices(prices, min_length=self.block_length + 1)
        self._log_returns = self._log_returns_from(prices)
        self._initial_price = float(prices[-1])

    def _log_returns_from(self, prices: np.ndarray) -> np.ndarray:
        return np.diff(np.log(prices))

    def simulate(
        self,
        n_paths: int = 1000,
        n_steps: int = 252,
        seed: int | None = 42,
    ) -> SimulationResult:
        if self._log_returns is None:
            raise RuntimeError("Must call calibrate() before simulate()")

        rng = np.random.default_rng(seed)
        T = len(self._log_returns)
        B = self.block_length

        log_increments = np.zeros((n_paths, n_steps))

        for i in range(n_paths):
            idx = 0
            while idx < n_steps:
                # Pick a random starting point for a block
                start = rng.integers(0, T)
                block_len = min(B, n_steps - idx)

                if self.circular:
                    # Circular: wrap around if we exceed the end
                    indices = np.arange(start, start + block_len) % T
                else:
                    # Truncate if block extends past the end
                    end = min(start + block_len, T)
                    actual_len = end - start
                    indices = np.arange(start, end)
                    block_len = actual_len

                log_increments[i, idx : idx + block_len] = self._log_returns[indices]
                idx += block_len

        # Build price paths
        log_paths = np.cumsum(log_increments, axis=1)
        log_paths = np.column_stack([np.zeros(n_paths), log_paths])
        paths = self._initial_price * np.exp(log_paths)

        return SimulationResult(
            paths=paths,
            returns=log_increments,
            metadata={
                "generator": "BlockBootstrap",
                "block_length": self.block_length,
                "circular": self.circular,
                "historical_returns_length": T,
                "historical_mean_annual": float(
                    np.mean(self._log_returns) * self.trading_days
                ),
                "historical_vol_annual": float(
                    np.std(self._log_returns, ddof=1) * np.sqrt(self.trading_days)
                ),
                "initial_price": self._initial_price,
            },
        )

    @property
    def params(self) -> dict[str, Any]:
        return {
            "block_length": self.block_length,
            "circular": self.circular,
            "historical_returns_length": (
                len(self._log_returns) if self._log_returns is not None else None
            ),
        }
