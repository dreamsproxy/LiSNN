from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .config import NetworkConfig
from .lif import TAU, V_THRESHOLD, initialize_population, step_population
from .plasticity import PlasticityMatrix

ProgressCallback = Callable[[int, int], None]


def stable_softmax(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled = scaled - np.max(scaled)
    exponentials = np.exp(np.clip(scaled, -700.0, 700.0))
    total = float(np.sum(exponentials))
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(exponentials, 1.0 / exponentials.size)
    return exponentials / total


@dataclass(frozen=True)
class PredictionResult:
    token_id: int
    confidence: float
    probabilities: np.ndarray
    logits: np.ndarray
    context_token_ids: np.ndarray


class LanguageTrajectoryNetwork:
    """Predict a token from a recent relative-time token trajectory.

    Hidden neurons have no direct external input and no direct readout path.
    They participate only through dense recurrent coupling with the I/O field.

    I/O and hidden populations are initialized from independent deterministic
    random streams. For a fixed seed, changing ``hidden_neurons`` leaves all
    initial I/O neuron types, LIF parameters, and I/O-to-I/O weights unchanged.
    """

    def __init__(
        self,
        vocabulary_size: int,
        config: NetworkConfig | None = None,
    ) -> None:
        if vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive")
        self.config = config or NetworkConfig()
        self.vocabulary_size = int(vocabulary_size)
        self.context_length = self.config.context_length
        self.hidden_neurons = self.config.hidden_neurons

        token_end = self.vocabulary_size
        time_end = token_end + self.context_length
        binding_end = time_end + self.vocabulary_size * self.context_length
        self.token_slice = slice(0, token_end)
        self.time_slice = slice(token_end, time_end)
        self.binding_slice = slice(time_end, binding_end)
        self.query_index = binding_end
        self.io_neuron_count = self.query_index + 1
        self.hidden_slice = slice(
            self.io_neuron_count,
            self.io_neuron_count + self.hidden_neurons,
        )
        self.num_neurons = self.hidden_slice.stop

        self.weights = PlasticityMatrix(
            self.num_neurons,
            self.config.pre_alpha,
            self.config.post_alpha,
            io_neuron_count=self.io_neuron_count,
            seed=self.config.seed,
            inhibitory_fraction=self.config.ei_ratio,
        )

        io_rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, 301])
        )
        io_neurons, io_thresholds, io_tau = initialize_population(
            self.io_neuron_count,
            self.config.dt,
            io_rng,
        )

        if self.hidden_neurons:
            hidden_rng = np.random.default_rng(
                np.random.SeedSequence([self.config.seed, 302])
            )
            hidden_neurons, hidden_thresholds, hidden_tau = (
                initialize_population(
                    self.hidden_neurons,
                    self.config.dt,
                    hidden_rng,
                )
            )
            self.neurons = np.vstack((io_neurons, hidden_neurons))
            self.thresholds = np.concatenate(
                (io_thresholds, hidden_thresholds)
            )
            tau = np.concatenate((io_tau, hidden_tau))
        else:
            self.neurons = io_neurons
            self.thresholds = io_thresholds
            tau = io_tau

        self.neurons[self.query_index] = np.asarray(
            [-65.0, self.config.dt, 20.0, -65.0, -70.0, -55.0],
            dtype=np.float64,
        )
        self.thresholds[self.query_index] = -55.0
        tau[self.query_index] = 20.0

        if self.hidden_neurons:
            self.thresholds[self.hidden_slice] -= (
                self.config.hidden_threshold_offset
            )
            self.neurons[self.hidden_slice, V_THRESHOLD] = self.thresholds[
                self.hidden_slice
            ]
            tau[self.hidden_slice] *= self.config.hidden_tau_scale
            self.neurons[self.hidden_slice, TAU] = tau[self.hidden_slice]

        self.pre_tau = tau.copy()
        self.post_tau = tau.copy()
        self.pre_spikes = np.zeros(self.num_neurons, dtype=np.float64)
        self.post_spikes = np.zeros(self.num_neurons, dtype=np.float64)
        self.global_step_tick = 0

        # Hidden neurons cannot connect directly to the output classifier.
        self.readout_weights = np.zeros(
            (self.vocabulary_size, self.io_neuron_count),
            dtype=np.float64,
        )
        self.readout_bias = np.zeros(
            self.vocabulary_size,
            dtype=np.float64,
        )

        # Strong driver columns apply only to I/O driver neurons, not hidden
        # cells. Their Dale sign remains fixed by the presynaptic neuron type.
        driver_slice = slice(self.time_slice.start, self.io_neuron_count)
        self.weights.set_column_magnitudes(driver_slice, 0.5)

    @property
    def estimated_dense_memory_bytes(self) -> int:
        recurrent = 2 * self.num_neurons * self.num_neurons * 8
        readout = self.vocabulary_size * self.io_neuron_count * 8
        return recurrent + readout

    def binding_index(self, token_id: int, slot: int) -> int:
        if token_id < 0 or token_id >= self.vocabulary_size:
            raise ValueError(f"token_id out of range: {token_id}")
        if slot < 0 or slot >= self.context_length:
            raise ValueError(f"slot out of range: {slot}")
        return (
            self.binding_slice.start
            + slot * self.vocabulary_size
            + token_id
        )

    def reset_dynamic_state(self) -> None:
        self.pre_spikes.fill(0.0)
        self.post_spikes.fill(0.0)
        self.neurons[:, 0] = self.neurons[:, 3] + 2.71
        self.neurons[self.query_index, 0] = -65.0

    def _context_input(
        self,
        token_id: int,
        slot: int,
    ) -> np.ndarray:
        distance_from_newest = self.context_length - 1 - slot
        scale = self.config.signal_scale * (
            self.config.recency_decay ** distance_from_newest
        )
        external = np.zeros(self.num_neurons, dtype=np.float64)
        external[token_id] = scale
        external[self.time_slice.start + slot] = scale
        external[self.binding_index(token_id, slot)] = scale
        return external

    def _query_input(self) -> np.ndarray:
        external = np.zeros(self.num_neurons, dtype=np.float64)
        external[self.query_index] = self.config.signal_scale
        return external

    def step(
        self,
        input_signals: np.ndarray,
        *,
        train_recurrent: bool,
    ) -> np.ndarray:
        if input_signals.shape != (self.num_neurons,):
            raise ValueError("input_signals has the wrong shape")
        self.post_spikes = step_population(
            self.neurons,
            input_signals,
            self.thresholds,
        )
        if train_recurrent:
            top_k = max(
                1,
                int(
                    round(
                        self.num_neurons
                        * self.config.top_k_fraction
                    )
                ),
            )
            clip = (
                self.global_step_tick % self.config.clip_interval != 0
            )
            self.weights.update_combined(
                self.pre_spikes,
                self.post_spikes,
                self.pre_tau,
                self.post_tau,
                clip=clip,
                top_k=top_k,
            )
            self.global_step_tick += 1
        signals = self.weights.propagate(
            self.post_spikes,
            method="mean",
        )
        self.pre_spikes = self.post_spikes.copy()
        return signals

    def encode_context(
        self,
        context_token_ids: Sequence[int] | np.ndarray,
        *,
        train_recurrent: bool,
    ) -> np.ndarray:
        context = np.asarray(context_token_ids, dtype=np.int64)
        if context.ndim != 1 or context.size < 1:
            raise ValueError("context must contain at least one token")
        if np.any(context < 0) or np.any(
            context >= self.vocabulary_size
        ):
            raise ValueError("context contains an out-of-range token id")
        context = context[-self.context_length :]
        start_slot = self.context_length - context.size

        self.reset_dynamic_state()
        recurrent_signal = np.zeros(
            self.num_neurons,
            dtype=np.float64,
        )
        accumulated = np.zeros(
            self.num_neurons,
            dtype=np.float64,
        )

        for offset, token_id in enumerate(context):
            slot = start_slot + offset
            external = self._context_input(int(token_id), slot)
            for tick in range(self.config.ticks_per_token):
                drive = (
                    external
                    + self.config.recurrent_scale * recurrent_signal
                )
                recurrent_signal = self.step(
                    drive,
                    train_recurrent=(
                        train_recurrent
                        and tick
                        == self.config.ticks_per_token - 1
                    ),
                )
                accumulated += self.post_spikes

        query = self._query_input()
        for tick in range(self.config.prediction_ticks):
            drive = (
                query
                + self.config.recurrent_scale * recurrent_signal
            )
            recurrent_signal = self.step(
                drive,
                train_recurrent=(
                    train_recurrent
                    and tick == self.config.prediction_ticks - 1
                ),
            )
            accumulated += self.post_spikes

        # Only I/O activity is visible to the classifier. Hidden activity can
        # alter it solely through recurrent propagation.
        io_feature = accumulated[: self.io_neuron_count]
        norm = float(np.linalg.norm(io_feature))
        if not np.isfinite(norm) or norm <= 1e-12:
            return np.zeros_like(io_feature)
        return io_feature / norm

    def _readout(
        self,
        feature: np.ndarray,
        temperature: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if feature.shape != (self.io_neuron_count,):
            raise ValueError("feature has the wrong shape")
        logits = self.readout_weights @ feature + self.readout_bias
        probabilities = stable_softmax(
            logits,
            self.config.temperature
            if temperature is None
            else temperature,
        )
        return logits, probabilities

    def _update_readout(
        self,
        feature: np.ndarray,
        target_token_id: int,
    ) -> None:
        _, probabilities = self._readout(feature)
        target = np.zeros(self.vocabulary_size, dtype=np.float64)
        target[target_token_id] = 1.0
        error = target - probabilities
        learning_rates = np.where(
            error >= 0.0,
            self.config.readout_alpha,
            self.config.anti_hebbian_alpha,
        )
        self.readout_weights *= 1.0 - self.config.readout_decay
        self.readout_weights += (
            learning_rates * error
        )[:, None] * feature[None, :]
        self.readout_bias += learning_rates * error

    def fit(
        self,
        token_ids: np.ndarray,
        *,
        epochs: int = 1,
        progress: ProgressCallback | None = None,
    ) -> None:
        sequence = np.asarray(token_ids, dtype=np.int64)
        if sequence.ndim != 1 or sequence.size < 2:
            raise ValueError("token_ids must contain at least two tokens")
        if np.any(sequence < 0) or np.any(
            sequence >= self.vocabulary_size
        ):
            raise ValueError("token_ids contains an out-of-range token id")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")

        total = epochs * (sequence.size - 1)
        completed = 0
        for _ in range(epochs):
            for target_position in range(1, sequence.size):
                start = max(
                    0,
                    target_position - self.context_length,
                )
                feature = self.encode_context(
                    sequence[start:target_position],
                    train_recurrent=True,
                )
                self._update_readout(
                    feature,
                    int(sequence[target_position]),
                )
                completed += 1
                if completed % self.config.prune_interval == 0:
                    self.weights.prune(
                        self.config.prune_threshold
                    )
                if progress is not None:
                    progress(completed, total)

    def predict_next(
        self,
        context_token_ids: Sequence[int] | np.ndarray,
        *,
        temperature: float | None = None,
    ) -> PredictionResult:
        context = np.asarray(context_token_ids, dtype=np.int64)
        feature = self.encode_context(
            context,
            train_recurrent=False,
        )
        logits, probabilities = self._readout(feature, temperature)
        token_id = int(np.argmax(probabilities))
        return PredictionResult(
            token_id=token_id,
            confidence=float(probabilities[token_id]),
            probabilities=probabilities,
            logits=logits,
            context_token_ids=context[
                -self.context_length :
            ].copy(),
        )
