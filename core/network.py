from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .config import NetworkConfig
from .lif import initialize_population, step_population
from .plasticity import PlasticityMatrix

ProgressCallback = Callable[[int, int], None]


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    if span <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - minimum) / span


@dataclass(frozen=True)
class RecallResult:
    position: int
    token_id: int
    confidence: float
    token_scores: np.ndarray


class LanguageTimecodeNetwork:
    """Associates one-hot sequence timecodes with token-neuron activity."""

    def __init__(
        self,
        vocabulary_size: int,
        sequence_length: int,
        config: NetworkConfig | None = None,
    ) -> None:
        if vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        self.config = config or NetworkConfig()
        self.vocabulary_size = int(vocabulary_size)
        self.sequence_length = int(sequence_length)
        self.num_neurons = self.vocabulary_size + self.sequence_length
        self.token_slice = slice(0, self.vocabulary_size)
        self.timecode_slice = slice(self.vocabulary_size, self.num_neurons)

        self.rng = np.random.default_rng(self.config.seed)
        self.weights = PlasticityMatrix(
            self.num_neurons,
            self.config.pre_alpha,
            self.config.post_alpha,
            self.rng,
        )
        self.neurons, self.thresholds, tau = initialize_population(
            self.num_neurons,
            self.config.dt,
            self.rng,
            fixed_tail=self.sequence_length,
        )
        self.pre_tau = tau.copy()
        self.post_tau = tau.copy()
        self.pre_spikes = np.zeros(self.num_neurons, dtype=np.float64)
        self.post_spikes = np.zeros(self.num_neurons, dtype=np.float64)
        self.global_step_tick = 0
        self.error_thresholds = np.linspace(
            1.0,
            0.1,
            num=self.config.error_threshold_steps,
            dtype=np.float64,
        )

        # Preserve the baseline's strong initial timecode connectivity, but
        # apply it on presynaptic timecode columns so codes can drive recall.
        self.weights.weights[:, self.timecode_slice] = 0.5

    @property
    def estimated_dense_memory_bytes(self) -> int:
        # recurrent + Hebbian matrices, both float64
        return 2 * self.num_neurons * self.num_neurons * 8

    def reset_dynamic_state(self) -> None:
        self.pre_spikes.fill(0.0)
        self.post_spikes.fill(0.0)
        self.neurons[:, 0] = self.neurons[:, 3] + 2.71
        self.neurons[self.timecode_slice, 0] = -65.0

    def _external_input(self, token_id: int, position: int) -> np.ndarray:
        if token_id < 0 or token_id >= self.vocabulary_size:
            raise ValueError(f"token_id out of range: {token_id}")
        if position < 0 or position >= self.sequence_length:
            raise ValueError(f"position out of range: {position}")
        external = np.zeros(self.num_neurons, dtype=np.float64)
        external[token_id] = self.config.signal_scale
        external[self.vocabulary_size + position] = self.config.signal_scale
        return external

    def _timecode_input(self, position: int) -> np.ndarray:
        if position < 0 or position >= self.sequence_length:
            raise ValueError(f"position out of range: {position}")
        external = np.zeros(self.num_neurons, dtype=np.float64)
        external[self.vocabulary_size + position] = self.config.signal_scale
        return external

    def _spike_population(
        self,
        neurons: np.ndarray,
        input_signals: np.ndarray,
    ) -> np.ndarray:
        # Token and timecode neurons share the same LIF dynamics. The archived
        # baseline skipped the timecode-neuron integration step, which erased
        # the code before recurrent propagation.
        return step_population(neurons, input_signals, self.thresholds)

    def step(self, input_signals: np.ndarray, *, train: bool = True) -> np.ndarray:
        if input_signals.shape != (self.num_neurons,):
            raise ValueError("input_signals has the wrong shape")

        self.post_spikes = self._spike_population(self.neurons, input_signals)

        if train:
            normalized_input = minmax_normalize(input_signals[self.token_slice])
            normalized_output = minmax_normalize(self.post_spikes[self.token_slice])
            error_vector = np.abs(normalized_input - normalized_output)

            for threshold in self.error_thresholds:
                error_indices = np.flatnonzero(error_vector > threshold)
                if error_indices.size:
                    self.post_spikes[error_indices] = 0.0
                    break

            top_k = max(1, int(round(self.num_neurons * self.config.top_k_fraction)))
            clip = self.global_step_tick % self.config.clip_interval != 0
            self.weights.update_combined(
                self.pre_spikes,
                self.post_spikes,
                self.pre_tau,
                self.post_tau,
                clip=clip,
                top_k=top_k,
            )
            self.global_step_tick += 1

        signals = self.weights.propagate(self.post_spikes, method="mean")
        self.pre_spikes = self.post_spikes.copy()
        return signals

    def fit(
        self,
        token_ids: np.ndarray,
        *,
        epochs: int = 1,
        ticks_per_token: int | None = None,
        progress: ProgressCallback | None = None,
        reset_before_fit: bool = True,
    ) -> None:
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.shape != (self.sequence_length,):
            raise ValueError(
                f"token_ids must have shape ({self.sequence_length},), "
                f"received {token_ids.shape}"
            )
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        ticks = ticks_per_token or self.config.ticks_per_token
        if ticks < 1:
            raise ValueError("ticks_per_token must be at least 1")

        if reset_before_fit:
            self.reset_dynamic_state()

        total_positions = epochs * self.sequence_length
        completed = 0
        recurrent_signal = self.pre_spikes.copy()
        for _ in range(epochs):
            for position, token_id in enumerate(token_ids):
                external = self._external_input(int(token_id), position)
                for _ in range(ticks):
                    recurrent_signal = self.step(external + recurrent_signal, train=True)

                if position % self.config.prune_interval == 0:
                    self.weights.prune(self.config.prune_threshold)

                completed += 1
                if progress is not None:
                    progress(completed, total_positions)

    def recall_position(
        self,
        position: int,
        *,
        num_ticks: int | None = None,
    ) -> RecallResult:
        ticks = num_ticks or self.config.recall_ticks
        if ticks < 1:
            raise ValueError("num_ticks must be at least 1")

        local_neurons = self.neurons.copy()
        drive = self._timecode_input(position)
        signals = np.zeros(self.num_neurons, dtype=np.float64)
        accumulated = np.zeros(self.num_neurons, dtype=np.float64)

        for tick in range(ticks):
            if tick > 0:
                drive = drive + signals
            spikes = self._spike_population(local_neurons, drive)
            if not np.all(np.isfinite(spikes)):
                raise FloatingPointError(
                    f"non-finite spike during recall at position={position}, tick={tick}"
                )
            signals = self.weights.propagate(spikes, method="sum")
            accumulated += spikes

        token_scores = accumulated[self.token_slice].copy()
        token_id = int(np.argmax(token_scores))
        positive_total = float(np.sum(np.maximum(token_scores, 0.0)))
        confidence = (
            float(max(token_scores[token_id], 0.0)) / positive_total
            if positive_total > 0.0
            else 0.0
        )
        return RecallResult(
            position=position,
            token_id=token_id,
            confidence=confidence,
            token_scores=token_scores,
        )

    def recall_sequence(
        self,
        positions: Iterable[int] | None = None,
        *,
        num_ticks: int | None = None,
    ) -> list[RecallResult]:
        selected = range(self.sequence_length) if positions is None else positions
        return [
            self.recall_position(int(position), num_ticks=num_ticks)
            for position in selected
        ]
