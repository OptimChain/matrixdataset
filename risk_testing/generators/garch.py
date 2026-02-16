"""
GARCH(1,1) path generator.

Model:
    r_t     = mu + sigma_t * z_t          (return equation)
    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2  (variance equation)

    where z_t ~ N(0,1) or z_t ~ t(nu)

Parameters:
    mu    - constant mean return (daily)
    omega - base variance (floor)
    alpha - ARCH coefficient (shock reactivity: how much yesterday's
            squared return affects today's variance)
    beta  - GARCH coefficient (variance persistence: how much yesterday's
            variance carries forward)
    nu    - degrees of freedom for Student-t innovations (optional)

    Stationarity constraint: alpha + beta < 1
    Unconditional variance: omega / (1 - alpha - beta)

vs GBM:
    - 4-5 params vs 2 params
    - Captures volatility clustering (alpha + beta persistence)
    - Captures fat tails (via t-distribution innovations)
    - Captures leverage effects (with EGARCH extension)
    - Does NOT capture regime shifts
    - Does NOT capture time-varying drift

vs HMM:
    - Fewer parameters than HMM with K>=3 regimes
    - Smooth vol dynamics vs discrete regime jumps
    - Better at modeling gradual vol changes
    - Worse at modeling sudden regime shifts
"""

from typing import Any

import numpy as np

from risk_testing.generators.base import PathGenerator, SimulationResult


class GARCHGenerator(PathGenerator):
    """GARCH(1,1) path generator with optional Student-t innovations.

    Args:
        trading_days: Trading days per year.
        use_t_dist: If True, use Student-t innovations for fat tails.
    """

    def __init__(self, trading_days: int = 252, use_t_dist: bool = True):
        self.trading_days = trading_days
        self.use_t_dist = use_t_dist

        self._mu: float | None = None
        self._omega: float | None = None
        self._alpha: float | None = None
        self._beta: float | None = None
        self._nu: float | None = None  # t-distribution degrees of freedom
        self._initial_price: float | None = None
        self._last_sigma2: float | None = None  # for simulation start
        self._last_ret: float | None = None

    def calibrate(self, prices: np.ndarray, **kwargs) -> None:
        """Calibrate GARCH(1,1) parameters from historical prices.

        Uses a variance-targeting approach for robust estimation:
        1. Estimate unconditional variance from data.
        2. Grid search over (alpha, beta) pairs to maximize
           quasi-maximum likelihood.
        3. Recover omega from the variance-targeting constraint:
           omega = unconditional_var * (1 - alpha - beta)
        """
        prices = self._validate_prices(prices, min_length=30)
        log_ret = self._log_returns(prices)
        self._initial_price = float(prices[-1])

        mu = float(np.mean(log_ret))
        self._mu = mu
        residuals = log_ret - mu
        unconditional_var = float(np.var(residuals, ddof=1))

        # Grid search for alpha, beta
        best_ll = -np.inf
        best_alpha, best_beta = 0.05, 0.90

        alpha_grid = np.arange(0.01, 0.30, 0.02)
        beta_grid = np.arange(0.50, 0.98, 0.02)

        for a in alpha_grid:
            for b in beta_grid:
                if a + b >= 0.9999:
                    continue

                omega = unconditional_var * (1 - a - b)
                if omega <= 0:
                    continue

                ll = self._garch_loglik(residuals, omega, a, b, unconditional_var)
                if ll > best_ll:
                    best_ll = ll
                    best_alpha = a
                    best_beta = b

        self._alpha = float(best_alpha)
        self._beta = float(best_beta)
        self._omega = unconditional_var * (1 - self._alpha - self._beta)

        # Compute conditional variance series (needed for both t-dist and init)
        sigma2_series = self._compute_sigma2_series(
            residuals, self._omega, self._alpha, self._beta, unconditional_var
        )

        # Estimate t-distribution degrees of freedom if requested
        if self.use_t_dist:
            # Method of moments: kurtosis = 3 + 6/(nu-4) for t-dist
            standardized = residuals / np.sqrt(sigma2_series)
            kurt = float(np.mean(standardized**4))
            if kurt > 3.0:
                # nu = 4 + 6/(kurt - 3), but floor at 4.1
                self._nu = max(4.1, 4.0 + 6.0 / (kurt - 3.0))
            else:
                self._nu = 30.0  # effectively normal
        else:
            self._nu = None

        # Store last values for simulation initialization
        self._last_sigma2 = float(sigma2_series[-1]) if len(residuals) > 0 else unconditional_var
        self._last_ret = float(residuals[-1])

    def _compute_sigma2_series(
        self,
        residuals: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
        sigma2_init: float,
    ) -> np.ndarray:
        """Compute conditional variance series."""
        T = len(residuals)
        sigma2 = np.empty(T)
        sigma2[0] = sigma2_init
        for t in range(1, T):
            sigma2[t] = omega + alpha * residuals[t - 1] ** 2 + beta * sigma2[t - 1]
        return sigma2

    def _garch_loglik(
        self,
        residuals: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
        sigma2_init: float,
    ) -> float:
        """Compute Gaussian quasi-log-likelihood for GARCH(1,1)."""
        sigma2 = self._compute_sigma2_series(
            residuals, omega, alpha, beta, sigma2_init
        )
        # Avoid log(0)
        sigma2 = np.maximum(sigma2, 1e-20)
        ll = -0.5 * np.sum(np.log(sigma2) + residuals**2 / sigma2)
        return float(ll)

    def simulate(
        self,
        n_paths: int = 1000,
        n_steps: int = 252,
        seed: int | None = 42,
    ) -> SimulationResult:
        if self._omega is None:
            raise RuntimeError("Must call calibrate() before simulate()")

        rng = np.random.default_rng(seed)

        log_increments = np.zeros((n_paths, n_steps))
        sigma2_history = np.zeros((n_paths, n_steps))

        # Initialize from last calibrated values
        sigma2_prev = np.full(n_paths, self._last_sigma2)
        eps_prev = np.full(n_paths, self._last_ret)

        for t in range(n_steps):
            # Update conditional variance
            sigma2_t = (
                self._omega + self._alpha * eps_prev**2 + self._beta * sigma2_prev
            )
            sigma2_t = np.maximum(sigma2_t, 1e-20)
            sigma2_history[:, t] = sigma2_t

            # Draw innovations
            if self.use_t_dist and self._nu is not None and self._nu < 100:
                Z = rng.standard_t(df=self._nu, size=n_paths)
                # Scale so variance = 1
                Z = Z / np.sqrt(self._nu / (self._nu - 2))
            else:
                Z = rng.standard_normal(n_paths)

            eps_t = np.sqrt(sigma2_t) * Z
            log_increments[:, t] = self._mu + eps_t

            eps_prev = eps_t
            sigma2_prev = sigma2_t

        # Build price paths
        log_paths = np.cumsum(log_increments, axis=1)
        log_paths = np.column_stack([np.zeros(n_paths), log_paths])
        paths = self._initial_price * np.exp(log_paths)

        # Annualize params for metadata
        unconditional_var = self._omega / (1 - self._alpha - self._beta)
        ann_vol = float(np.sqrt(unconditional_var * self.trading_days))

        return SimulationResult(
            paths=paths,
            returns=log_increments,
            metadata={
                "generator": "GARCH(1,1)",
                "mu_daily": self._mu,
                "mu_annual": self._mu * self.trading_days,
                "omega": self._omega,
                "alpha": self._alpha,
                "beta": self._beta,
                "persistence": self._alpha + self._beta,
                "unconditional_vol_annual": ann_vol,
                "nu": self._nu,
                "use_t_dist": self.use_t_dist,
                "initial_price": self._initial_price,
            },
        )

    @property
    def params(self) -> dict[str, Any]:
        persistence = (
            (self._alpha + self._beta)
            if self._alpha is not None and self._beta is not None
            else None
        )
        return {
            "mu": self._mu,
            "omega": self._omega,
            "alpha": self._alpha,
            "beta": self._beta,
            "persistence": persistence,
            "nu": self._nu,
            "use_t_dist": self.use_t_dist,
        }
