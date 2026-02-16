"""Shared fixtures for risk testing tests."""

import numpy as np
import pytest


@pytest.fixture
def synthetic_prices():
    """Generate a synthetic price series with trend and volatility.

    Simulates ~2 years of daily prices with a slight upward drift
    and realistic volatility (~20% annualized).
    """
    rng = np.random.default_rng(12345)
    n_days = 504  # ~2 years
    mu = 0.08 / 252  # ~8% annual drift
    sigma = 0.20 / np.sqrt(252)  # ~20% annual vol

    log_returns = mu + sigma * rng.standard_normal(n_days)
    log_prices = np.cumsum(log_returns)
    prices = 100.0 * np.exp(np.concatenate([[0], log_prices]))
    return prices


@pytest.fixture
def regime_prices():
    """Generate synthetic prices with two distinct regimes.

    First half: low vol bull market (~15% vol, +20% drift)
    Second half: high vol bear market (~40% vol, -10% drift)
    """
    rng = np.random.default_rng(67890)
    n_half = 252

    # Bull regime
    mu1 = 0.20 / 252
    sigma1 = 0.15 / np.sqrt(252)
    bull_returns = mu1 + sigma1 * rng.standard_normal(n_half)

    # Bear regime
    mu2 = -0.10 / 252
    sigma2 = 0.40 / np.sqrt(252)
    bear_returns = mu2 + sigma2 * rng.standard_normal(n_half)

    log_returns = np.concatenate([bull_returns, bear_returns])
    log_prices = np.cumsum(log_returns)
    prices = 100.0 * np.exp(np.concatenate([[0], log_prices]))
    return prices


@pytest.fixture
def short_prices():
    """Minimal price series for edge-case testing."""
    rng = np.random.default_rng(11111)
    returns = 0.001 + 0.01 * rng.standard_normal(50)
    return 100.0 * np.exp(np.concatenate([[0], np.cumsum(returns)]))
