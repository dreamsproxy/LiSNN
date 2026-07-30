from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeightStats:
    mean: float
    std: float
    minimum: float
    maximum: float
    nonzero_fraction: float


class PlasticityMatrix:
    """Dense recurrent matrix with combined STDP and Hebbian plasticity."""

    def __init__(
        self,
        num_neurons: int,
        pre_alpha: float,
        post_alpha: float,
        rng: np.random.Generator,
    ) -> None:
        if num_neurons < 1:
            raise ValueError("num_neurons must be positive")
        self.num_neurons = int(num_neurons)
        self.a_pre = np.float64(pre_alpha)
        self.a_post = np.float64(post_alpha)

        limit = np.sqrt(6.0 / (2.0 * self.num_neurons))
        self.weights = rng.uniform(
            -limit,
            limit,
            (self.num_neurons, self.num_neurons),
        ).astype(np.float64)
        # A Hebbian trace is accumulated evidence, so it starts at zero.
        # The archived baseline randomized this matrix, which injected an
        # unrelated dense association before the first training example.
        self.hebb_weights = np.zeros(
            (self.num_neurons, self.num_neurons),
            dtype=np.float64,
        )

    @staticmethod
    def compute_spikes(
        potentials: np.ndarray,
        thresholds: np.ndarray,
    ) -> np.ndarray:
        return np.maximum(0.0, potentials - thresholds)

    @property
    def stats(self) -> WeightStats:
        return WeightStats(
            mean=float(self.weights.mean()),
            std=float(self.weights.std()),
            minimum=float(self.weights.min()),
            maximum=float(self.weights.max()),
            nonzero_fraction=float(np.count_nonzero(self.weights) / self.weights.size),
        )

    def _stdp_update(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        tau_pre: np.ndarray,
        tau_post: np.ndarray,
    ) -> np.ndarray:
        delta = post_spikes[:, None] - pre_spikes[None, :]
        potentiation = (
            self.a_pre
            * post_spikes[:, None]
            * np.exp(-delta / tau_pre[None, :])
        )
        depression = (
            -self.a_post
            * pre_spikes[None, :]
            * np.exp(delta / tau_post[:, None])
        )
        return np.where(delta > 0.0, potentiation, depression)

    def _hebb_update(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
    ) -> None:
        active_pre = pre_spikes > 0.0
        if not np.any(active_pre):
            return
        active_post = post_spikes > 0.0
        self.hebb_weights[np.ix_(active_post, active_pre)] += self.a_post
        self.hebb_weights[np.ix_(~active_post, active_pre)] -= self.a_pre

    def update_combined(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        tau_pre: np.ndarray,
        tau_post: np.ndarray,
        *,
        clip: bool,
        top_k: int,
    ) -> None:
        if pre_spikes.shape != (self.num_neurons,):
            raise ValueError("pre_spikes has the wrong shape")
        if post_spikes.shape != (self.num_neurons,):
            raise ValueError("post_spikes has the wrong shape")

        top_k = max(1, min(int(top_k), self.num_neurons))
        top_indices = np.argpartition(post_spikes, -top_k)[-top_k:]
        restricted_post = np.zeros_like(post_spikes)
        restricted_post[top_indices] = post_spikes[top_indices]

        self.weights += self._stdp_update(
            pre_spikes,
            restricted_post,
            tau_pre,
            tau_post,
        )
        self._hebb_update(pre_spikes, restricted_post)
        self.weights = 0.5 * (self.weights + self.hebb_weights)

        if clip:
            self.weights = np.clip(self.weights, 1e-8, 1.0)

        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        self.weights /= safe_norms

    def propagate(self, spikes: np.ndarray, method: str = "sum") -> np.ndarray:
        if spikes.shape != (self.num_neurons,):
            raise ValueError("spikes has the wrong shape")
        signals = self.weights @ spikes
        if method == "sum":
            return signals
        if method == "mean":
            return signals / self.num_neurons
        if method == "weighted":
            row_sums = self.weights.sum(axis=1)
            safe_sums = np.where(np.abs(row_sums) > 1e-8, row_sums, 1.0)
            return signals / safe_sums
        raise ValueError("method must be 'sum', 'mean', or 'weighted'")

    def prune(self, threshold: float = 1e-4) -> None:
        self.weights[np.abs(self.weights) < threshold] = 0.0
