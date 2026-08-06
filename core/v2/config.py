from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for the relative-time next-token network."""

    dt: float = 1.0
    pre_alpha: float = 0.001
    post_alpha: float = 0.002
    context_length: int = 8
    hidden_neurons: int = 0
    ticks_per_token: int = 8
    prediction_ticks: int = 16
    signal_scale: float = 500.0
    recurrent_scale: float = 1.0
    readout_alpha: float = 0.2
    anti_hebbian_alpha: float = 0.05
    readout_decay: float = 1e-5
    temperature: float = 1.0
    recency_decay: float = 0.9
    ei_ratio: float = 0.5
    hidden_threshold_offset: float = 2.0
    hidden_tau_scale: float = 0.9
    clip_interval: int = 8
    prune_interval: int = 64
    prune_threshold: float = 1e-4
    top_k_fraction: float = 0.25
    error_threshold_steps: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.pre_alpha < 0 or self.post_alpha < 0:
            raise ValueError("plasticity rates must be non-negative")
        if self.context_length < 1:
            raise ValueError("context_length must be at least 1")
        if self.hidden_neurons < 0:
            raise ValueError("hidden_neurons must be non-negative")
        if self.ticks_per_token < 1 or self.prediction_ticks < 1:
            raise ValueError("tick counts must be at least 1")
        if self.signal_scale <= 0:
            raise ValueError("signal_scale must be positive")
        if self.recurrent_scale < 0:
            raise ValueError("recurrent_scale must be non-negative")
        if self.readout_alpha <= 0 or self.anti_hebbian_alpha < 0:
            raise ValueError("readout learning rates are invalid")
        if self.readout_decay < 0:
            raise ValueError("readout_decay must be non-negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.recency_decay <= 1:
            raise ValueError("recency_decay must be in (0, 1]")
        if not 0 <= self.ei_ratio <= 1:
            raise ValueError("ei_ratio must be in [0, 1]")
        if self.hidden_threshold_offset < 0:
            raise ValueError("hidden_threshold_offset must be non-negative")
        if not 0 < self.hidden_tau_scale <= 1:
            raise ValueError("hidden_tau_scale must be in (0, 1]")
        if self.clip_interval < 1 or self.prune_interval < 1:
            raise ValueError("intervals must be at least 1")
        if not 0 < self.top_k_fraction <= 1:
            raise ValueError("top_k_fraction must be in (0, 1]")
        if self.error_threshold_steps < 1:
            raise ValueError("error_threshold_steps must be at least 1")

    @property
    def inhibitory_fraction(self) -> float:
        """Compatibility alias for the inhibitory share of the E/I split."""

        return self.ei_ratio

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
