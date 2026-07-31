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
    """Dense recurrent matrix with immutable Dale-style neuron polarity.

    Matrix columns are presynaptic neurons because propagation is ``W @ spikes``.
    Every outgoing weight from an excitatory neuron is non-negative, while every
    outgoing weight from an inhibitory neuron is non-positive.

    I/O and hidden populations use independent deterministic random streams.
    Changing ``hidden_neurons`` therefore does not resample I/O neuron types or
    the initial I/O-to-I/O weight block.
    """

    EXCITATORY = np.int8(1)
    INHIBITORY = np.int8(-1)

    def __init__(
        self,
        num_neurons: int,
        pre_alpha: float,
        post_alpha: float,
        *,
        io_neuron_count: int,
        seed: int,
        inhibitory_fraction: float = 0.5,
    ) -> None:
        if num_neurons < 1:
            raise ValueError("num_neurons must be positive")
        if io_neuron_count < 1 or io_neuron_count > num_neurons:
            raise ValueError("io_neuron_count must be in [1, num_neurons]")
        if not 0 <= inhibitory_fraction <= 1:
            raise ValueError("inhibitory_fraction must be in [0, 1]")

        self.num_neurons = int(num_neurons)
        self.io_neuron_count = int(io_neuron_count)
        self.hidden_neuron_count = self.num_neurons - self.io_neuron_count
        self.a_pre = np.float64(pre_alpha)
        self.a_post = np.float64(post_alpha)

        io_type_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 101])
        )
        hidden_type_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 102])
        )
        io_types = self._sample_types(
            self.io_neuron_count,
            inhibitory_fraction,
            io_type_rng,
        )
        hidden_types = self._sample_types(
            self.hidden_neuron_count,
            inhibitory_fraction,
            hidden_type_rng,
        )
        self.neuron_types = np.concatenate((io_types, hidden_types)).astype(
            np.int8,
            copy=False,
        )

        self.weights = np.empty(
            (self.num_neurons, self.num_neurons),
            dtype=np.float64,
        )
        self._initialize_blockwise_weights(seed)
        self.hebb_weights = np.zeros_like(self.weights)
        self._enforce_neuron_types()

    @classmethod
    def _sample_types(
        cls,
        population_size: int,
        inhibitory_fraction: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        types = np.full(
            population_size,
            cls.EXCITATORY,
            dtype=np.int8,
        )
        if population_size == 0:
            return types

        # Round halves upward so a 0.5 ratio gives the closest intuitive split
        # for odd-sized populations rather than Python's banker rounding.
        inhibitory_count = int(
            np.floor(population_size * inhibitory_fraction + 0.5)
        )
        inhibitory_count = min(
            max(inhibitory_count, 0),
            population_size,
        )
        if inhibitory_count:
            inhibitory_indices = rng.choice(
                population_size,
                size=inhibitory_count,
                replace=False,
            )
            types[inhibitory_indices] = cls.INHIBITORY
        return types

    @staticmethod
    def _xavier_limit(fan_in: int, fan_out: int) -> float:
        return float(np.sqrt(6.0 / max(fan_in + fan_out, 1)))

    def _initialize_blockwise_weights(self, seed: int) -> None:
        """Initialize stable I/O and independently sampled hidden blocks."""

        io = self.io_neuron_count
        hidden = self.hidden_neuron_count

        io_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 201])
        )
        io_limit = self._xavier_limit(io, io)
        self.weights[:io, :io] = io_rng.uniform(
            1e-8,
            io_limit,
            (io, io),
        )

        if hidden:
            cross_rng = np.random.default_rng(
                np.random.SeedSequence([seed, 202])
            )
            hidden_rng = np.random.default_rng(
                np.random.SeedSequence([seed, 203])
            )
            cross_limit = self._xavier_limit(io, hidden)
            hidden_limit = self._xavier_limit(hidden, hidden)
            self.weights[:io, io:] = cross_rng.uniform(
                1e-8,
                cross_limit,
                (io, hidden),
            )
            self.weights[io:, :io] = cross_rng.uniform(
                1e-8,
                cross_limit,
                (hidden, io),
            )
            self.weights[io:, io:] = hidden_rng.uniform(
                1e-8,
                hidden_limit,
                (hidden, hidden),
            )

        self.weights *= self.neuron_types[None, :]

    @property
    def inhibitory_count(self) -> int:
        return int(np.sum(self.neuron_types == self.INHIBITORY))

    @property
    def excitatory_count(self) -> int:
        return int(np.sum(self.neuron_types == self.EXCITATORY))

    @property
    def io_inhibitory_count(self) -> int:
        return int(
            np.sum(
                self.neuron_types[: self.io_neuron_count]
                == self.INHIBITORY
            )
        )

    @property
    def io_excitatory_count(self) -> int:
        return self.io_neuron_count - self.io_inhibitory_count

    @property
    def hidden_inhibitory_count(self) -> int:
        return int(
            np.sum(
                self.neuron_types[self.io_neuron_count :]
                == self.INHIBITORY
            )
        )

    @property
    def hidden_excitatory_count(self) -> int:
        return self.hidden_neuron_count - self.hidden_inhibitory_count

    @property
    def stats(self) -> WeightStats:
        return WeightStats(
            mean=float(self.weights.mean()),
            std=float(self.weights.std()),
            minimum=float(self.weights.min()),
            maximum=float(self.weights.max()),
            nonzero_fraction=float(
                np.count_nonzero(self.weights) / self.weights.size
            ),
        )

    def set_column_magnitudes(
        self,
        columns: slice | np.ndarray,
        magnitude: float,
    ) -> None:
        if magnitude < 0:
            raise ValueError("magnitude must be non-negative")
        self.weights[:, columns] = (
            magnitude * self.neuron_types[columns][None, :]
        )

    def _enforce_neuron_types(self) -> None:
        excitatory = self.neuron_types == self.EXCITATORY
        inhibitory = ~excitatory
        self.weights[:, excitatory] = np.maximum(
            self.weights[:, excitatory],
            0.0,
        )
        self.weights[:, inhibitory] = np.minimum(
            self.weights[:, inhibitory],
            0.0,
        )
        self.hebb_weights[:, excitatory] = np.maximum(
            self.hebb_weights[:, excitatory],
            0.0,
        )
        self.hebb_weights[:, inhibitory] = np.minimum(
            self.hebb_weights[:, inhibitory],
            0.0,
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
        unsigned_update = np.where(
            delta > 0.0,
            potentiation,
            depression,
        )
        return unsigned_update * self.neuron_types[None, :]

    def _hebb_update(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
    ) -> None:
        active_pre_indices = np.flatnonzero(pre_spikes > 0.0)
        if not active_pre_indices.size:
            return
        post_update = np.where(
            post_spikes > 0.0,
            self.a_post,
            -self.a_pre,
        )
        self.hebb_weights[:, active_pre_indices] += (
            post_update[:, None]
            * self.neuron_types[active_pre_indices][None, :]
        )

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
        self._enforce_neuron_types()

        if clip:
            self.weights = np.clip(self.weights, -1.0, 1.0)
            self._enforce_neuron_types()

        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        self.weights /= safe_norms
        self._enforce_neuron_types()

    def propagate(
        self,
        spikes: np.ndarray,
        method: str = "sum",
    ) -> np.ndarray:
        if spikes.shape != (self.num_neurons,):
            raise ValueError("spikes has the wrong shape")
        signals = self.weights @ spikes
        if method == "sum":
            return signals
        if method == "mean":
            return signals / self.num_neurons
        if method == "weighted":
            row_sums = self.weights.sum(axis=1)
            safe_sums = np.where(
                np.abs(row_sums) > 1e-8,
                row_sums,
                1.0,
            )
            return signals / safe_sums
        raise ValueError("method must be 'sum', 'mean', or 'weighted'")

    def prune(self, threshold: float = 1e-4) -> None:
        self.weights[np.abs(self.weights) < threshold] = 0.0
