"""
Risk testing framework for Monte Carlo path simulation.

Supports multiple path generation models:
- GBM (Geometric Brownian Motion)
- HMM (Hidden Markov Model / Regime-Switching)
- GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
- Block Bootstrap (non-parametric resampling)

Each generator produces (n_paths, n_steps+1) price matrices that can be
fed into the risk metrics module for VaR, CVaR, drawdown analysis, etc.
"""

from risk_testing.generators.base import PathGenerator, SimulationResult
from risk_testing.generators.gbm import GBMGenerator
from risk_testing.generators.markov import MarkovRegimeSwitchingGenerator
from risk_testing.generators.garch import GARCHGenerator
from risk_testing.generators.bootstrap import BlockBootstrapGenerator
from risk_testing.metrics.risk_metrics import RiskMetrics
from risk_testing.metrics.path_analytics import PathAnalytics

__all__ = [
    "PathGenerator",
    "SimulationResult",
    "GBMGenerator",
    "MarkovRegimeSwitchingGenerator",
    "GARCHGenerator",
    "BlockBootstrapGenerator",
    "RiskMetrics",
    "PathAnalytics",
]
