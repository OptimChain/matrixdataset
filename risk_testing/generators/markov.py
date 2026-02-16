"""
Markov Regime-Switching path generator.

Model:
    At each time step, the system is in one of K regimes.
    Each regime k has its own drift mu_k and volatility sigma_k.
    Transitions between regimes follow a Markov chain with
    transition matrix P[i,j] = P(regime_{t+1}=j | regime_t=i).

    Within each regime, returns are drawn from:
        r_t = mu_{s_t} * dt + sigma_{s_t} * sqrt(dt) * Z_t

Parameters:
    n_regimes         - number of regimes (default 2: bull/bear)
    regime_mu         - (K,) drift per regime
    regime_sigma      - (K,) volatility per regime
    transition_matrix - (K, K) row-stochastic transition probabilities
    initial_probs     - (K,) initial regime distribution

vs GBM:
    - K*2 + K^2 parameters vs 2 parameters
    - Captures fat tails via regime mixing
    - Captures volatility clustering via persistent regimes
    - Captures regime shifts (bull -> bear transitions)
    - Within-regime dynamics are still i.i.d. normal

Calibration uses a simple EM-like approach: classify returns into
K clusters by magnitude, then estimate transition counts.
"""

from typing import Any

import numpy as np

from risk_testing.generators.base import PathGenerator, SimulationResult


class MarkovRegimeSwitchingGenerator(PathGenerator):
    """Hidden Markov Model regime-switching path generator.

    Args:
        n_regimes: Number of latent regimes.
        trading_days: Trading days per year.
        max_em_iterations: Maximum EM iterations for calibration.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        trading_days: int = 252,
        max_em_iterations: int = 50,
    ):
        self.n_regimes = n_regimes
        self.trading_days = trading_days
        self.max_em_iterations = max_em_iterations

        self._regime_mu: np.ndarray | None = None
        self._regime_sigma: np.ndarray | None = None
        self._transition_matrix: np.ndarray | None = None
        self._initial_probs: np.ndarray | None = None
        self._initial_price: float | None = None

    def calibrate(self, prices: np.ndarray, **kwargs) -> None:
        """Calibrate regime parameters from historical prices.

        Uses a quantile-based regime classification followed by
        maximum-likelihood estimation of per-regime parameters and
        transition probabilities.
        """
        prices = self._validate_prices(prices, min_length=30)
        log_ret = self._log_returns(prices)
        self._initial_price = float(prices[-1])

        K = self.n_regimes

        # --- Quantile-based initial regime assignment ---
        # Split returns into K quantile buckets
        quantile_edges = np.linspace(0, 100, K + 1)
        thresholds = np.percentile(log_ret, quantile_edges[1:-1])
        regime_labels = np.digitize(log_ret, thresholds)  # 0..K-1

        # --- EM-like refinement ---
        for _ in range(self.max_em_iterations):
            # M-step: estimate parameters per regime
            mu_k = np.zeros(K)
            sigma_k = np.zeros(K)
            for k in range(K):
                mask = regime_labels == k
                if np.sum(mask) < 2:
                    mu_k[k] = np.mean(log_ret)
                    sigma_k[k] = np.std(log_ret, ddof=1)
                else:
                    mu_k[k] = np.mean(log_ret[mask])
                    sigma_k[k] = np.std(log_ret[mask], ddof=1)

            # Ensure sigma is positive
            sigma_k = np.maximum(sigma_k, 1e-8)

            # E-step: reassign regimes based on likelihood
            log_likelihoods = np.zeros((len(log_ret), K))
            for k in range(K):
                log_likelihoods[:, k] = -0.5 * np.log(
                    2 * np.pi * sigma_k[k] ** 2
                ) - 0.5 * ((log_ret - mu_k[k]) / sigma_k[k]) ** 2

            new_labels = np.argmax(log_likelihoods, axis=1)

            if np.array_equal(new_labels, regime_labels):
                break
            regime_labels = new_labels

        # Sort regimes by mu so regime 0 = most bearish
        order = np.argsort(mu_k)
        mu_k = mu_k[order]
        sigma_k = sigma_k[order]
        regime_labels = np.array([np.searchsorted(order, r) for r in regime_labels])
        # Remap labels through the sort order
        inv_order = np.argsort(order)
        regime_labels = inv_order[regime_labels]

        # --- Transition matrix from observed transitions ---
        trans = np.ones((K, K))  # Laplace smoothing
        for t in range(len(regime_labels) - 1):
            trans[regime_labels[t], regime_labels[t + 1]] += 1
        trans_matrix = trans / trans.sum(axis=1, keepdims=True)

        # --- Initial state distribution ---
        init_counts = np.ones(K)
        for k in range(K):
            init_counts[k] += np.sum(regime_labels == k)
        initial_probs = init_counts / init_counts.sum()

        # Annualize
        self._regime_mu = mu_k * self.trading_days
        self._regime_sigma = sigma_k * np.sqrt(self.trading_days)
        self._transition_matrix = trans_matrix
        self._initial_probs = initial_probs

    def simulate(
        self,
        n_paths: int = 1000,
        n_steps: int = 252,
        seed: int | None = 42,
    ) -> SimulationResult:
        if self._regime_mu is None:
            raise RuntimeError("Must call calibrate() before simulate()")

        rng = np.random.default_rng(seed)
        K = self.n_regimes
        dt = 1.0 / self.trading_days

        log_increments = np.zeros((n_paths, n_steps))
        regime_history = np.zeros((n_paths, n_steps), dtype=int)

        # Sample initial regimes
        regimes = rng.choice(K, size=n_paths, p=self._initial_probs)

        for t in range(n_steps):
            regime_history[:, t] = regimes

            # Draw returns conditioned on current regime
            mu_t = self._regime_mu[regimes] * dt
            sigma_t = self._regime_sigma[regimes] * np.sqrt(dt)
            drift = mu_t - 0.5 * (self._regime_sigma[regimes] ** 2) * dt
            diffusion = sigma_t

            Z = rng.standard_normal(n_paths)
            log_increments[:, t] = drift + diffusion * Z

            # Transition to next regime
            new_regimes = np.empty(n_paths, dtype=int)
            for k in range(K):
                mask = regimes == k
                if np.any(mask):
                    count = np.sum(mask)
                    new_regimes[mask] = rng.choice(
                        K, size=count, p=self._transition_matrix[k]
                    )
            regimes = new_regimes

        # Build price paths
        log_paths = np.cumsum(log_increments, axis=1)
        log_paths = np.column_stack([np.zeros(n_paths), log_paths])
        paths = self._initial_price * np.exp(log_paths)

        return SimulationResult(
            paths=paths,
            returns=log_increments,
            metadata={
                "generator": "MarkovRegimeSwitching",
                "n_regimes": K,
                "regime_mu": self._regime_mu.tolist(),
                "regime_sigma": self._regime_sigma.tolist(),
                "transition_matrix": self._transition_matrix.tolist(),
                "initial_probs": self._initial_probs.tolist(),
                "initial_price": self._initial_price,
                "regime_history_sample": regime_history[:5].tolist(),
            },
        )

    @property
    def params(self) -> dict[str, Any]:
        return {
            "n_regimes": self.n_regimes,
            "regime_mu": (
                self._regime_mu.tolist() if self._regime_mu is not None else None
            ),
            "regime_sigma": (
                self._regime_sigma.tolist() if self._regime_sigma is not None else None
            ),
            "transition_matrix": (
                self._transition_matrix.tolist()
                if self._transition_matrix is not None
                else None
            ),
            "initial_probs": (
                self._initial_probs.tolist()
                if self._initial_probs is not None
                else None
            ),
        }
