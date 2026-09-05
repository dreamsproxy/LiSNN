"""Deterministic spiking substrate for the hypothesis workspace.

This is intentionally a small, inspectable prototype. Each hypothesis owns a
heterogeneous LIF population, all hypothesis populations feed a shared global
inhibitory population, and each hypothesis also carries a slow persistence
state. Divergence, balanced, and convergence modes alter thresholds,
recurrence, and inhibitory strength without changing the public substrate
interface.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from cognitive.interfaces import CognitiveSubstrate
from cognitive.schemas import CognitiveEvent, HypothesisDefinition, HypothesisState


@dataclass(frozen=True)
class _ModeParameters:
    threshold_scale: float
    recurrent_gain: float
    inhibition_gain: float


_MODE_PARAMETERS = {
    "divergence": _ModeParameters(0.82, 0.34, 0.04),
    "balanced": _ModeParameters(1.00, 0.28, 0.16),
    "convergence": _ModeParameters(1.10, 0.22, 0.42),
}


@dataclass
class _InhibitoryPopulationState:
    membrane: list[float]
    spikes: list[int]
    trace: float = 0.0


@dataclass
class _PopulationState:
    membrane: list[float]
    adaptation: list[float]
    spikes: list[int]
    trace: float = 0.0
    persistence: float = 0.0
    evidence_for: float = 0.0
    evidence_against: float = 0.0


class LiSNNHypothesisSubstrate(CognitiveSubstrate):
    """Small deterministic LIF workspace with slow hypothesis persistence."""

    def __init__(
        self,
        *,
        neurons_per_hypothesis: int = 16,
        inhibitory_neurons: int = 8,
        mode: str = "balanced",
        membrane_decay: float = 0.84,
        trace_decay: float = 0.88,
        adaptation_decay: float = 0.92,
        adaptation_gain: float = 0.08,
        persistence_decay: float = 0.985,
        persistence_gain: float = 0.10,
        persistence_feedback: float = 0.12,
        contradiction_gain: float = 0.05,
        evidence_gain: float = 1.0,
        confidence_persistence_gain: float = 0.10,
        inhibitory_membrane_decay: float = 0.80,
        inhibitory_trace_decay: float = 0.85,
        inhibitory_input_gain: float = 0.55,
    ) -> None:
        if neurons_per_hypothesis <= 0:
            raise ValueError("neurons_per_hypothesis must be positive")
        if inhibitory_neurons <= 0:
            raise ValueError("inhibitory_neurons must be positive")

        for name, value in (
            ("membrane_decay", membrane_decay),
            ("trace_decay", trace_decay),
            ("adaptation_decay", adaptation_decay),
            ("persistence_decay", persistence_decay),
            ("inhibitory_membrane_decay", inhibitory_membrane_decay),
            ("inhibitory_trace_decay", inhibitory_trace_decay),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")

        for name, value in (
            ("adaptation_gain", adaptation_gain),
            ("persistence_gain", persistence_gain),
            ("persistence_feedback", persistence_feedback),
            ("contradiction_gain", contradiction_gain),
            ("evidence_gain", evidence_gain),
            ("confidence_persistence_gain", confidence_persistence_gain),
            ("inhibitory_input_gain", inhibitory_input_gain),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

        self.neurons_per_hypothesis = neurons_per_hypothesis
        self.inhibitory_neurons = inhibitory_neurons
        self.membrane_decay = membrane_decay
        self.trace_decay = trace_decay
        self.adaptation_decay = adaptation_decay
        self.adaptation_gain = adaptation_gain
        self.persistence_decay = persistence_decay
        self.persistence_gain = persistence_gain
        self.persistence_feedback = persistence_feedback
        self.contradiction_gain = contradiction_gain
        self.evidence_gain = evidence_gain
        self.confidence_persistence_gain = confidence_persistence_gain
        self.inhibitory_membrane_decay = inhibitory_membrane_decay
        self.inhibitory_trace_decay = inhibitory_trace_decay
        self.inhibitory_input_gain = inhibitory_input_gain

        self._definitions: dict[str, HypothesisDefinition] = {}
        self._populations: dict[str, _PopulationState] = {}
        self._inhibitory_population = self._new_inhibitory_population()
        self._last_step = -1
        self.set_mode(mode)

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in _MODE_PARAMETERS:
            raise ValueError(f"unknown mode: {mode!r}")
        self._mode = mode

    def reset(self) -> None:
        self._populations = {
            hypothesis_id: self._new_population()
            for hypothesis_id in self._definitions
        }
        self._inhibitory_population = self._new_inhibitory_population()
        self._last_step = -1

    def register_hypotheses(
        self,
        hypotheses: Sequence[HypothesisDefinition],
    ) -> None:
        definitions = {item.hypothesis_id: item for item in hypotheses}
        if len(definitions) != len(hypotheses):
            raise ValueError("hypothesis ids must be unique")
        self._definitions = definitions
        self.reset()

    def step(self, event: CognitiveEvent) -> Mapping[str, HypothesisState]:
        if event.step <= self._last_step:
            raise ValueError("events must be applied in strictly increasing step order")
        if not self._populations:
            raise RuntimeError("register_hypotheses must be called before step")

        unknown = (set(event.supports) | set(event.weakens)) - set(
            self._populations
        )
        if unknown:
            raise ValueError(
                f"event references unknown hypotheses: {sorted(unknown)}"
            )

        silent_ticks = (
            event.step
            if self._last_step < 0
            else event.step - self._last_step - 1
        )
        for _ in range(silent_ticks):
            self._advance_tick({})

        signed_evidence = {
            hypothesis_id: event.supports.get(hypothesis_id, 0.0)
            - event.weakens.get(hypothesis_id, 0.0)
            for hypothesis_id in self._populations
        }
        self._advance_tick(signed_evidence)

        for hypothesis_id, strength in event.supports.items():
            self._populations[hypothesis_id].evidence_for += strength
        for hypothesis_id, strength in event.weakens.items():
            self._populations[hypothesis_id].evidence_against += strength

        self._last_step = event.step
        return self._public_states()

    def consolidate(self) -> None:
        for population in self._populations.values():
            confidence = population.evidence_for - population.evidence_against
            population.persistence += (
                self.confidence_persistence_gain * max(confidence, 0.0)
            )

    def rank(self) -> list[str]:
        states = self._public_states()
        return sorted(
            states,
            key=lambda hypothesis_id: (
                states[hypothesis_id].confidence,
                states[hypothesis_id].persistence,
                states[hypothesis_id].activation,
                hypothesis_id,
            ),
            reverse=True,
        )

    def export_state(self) -> dict[str, Any]:
        return {
            "config": {
                "neurons_per_hypothesis": self.neurons_per_hypothesis,
                "inhibitory_neurons": self.inhibitory_neurons,
                "mode": self.mode,
                "membrane_decay": self.membrane_decay,
                "trace_decay": self.trace_decay,
                "adaptation_decay": self.adaptation_decay,
                "adaptation_gain": self.adaptation_gain,
                "persistence_decay": self.persistence_decay,
                "persistence_gain": self.persistence_gain,
                "persistence_feedback": self.persistence_feedback,
                "contradiction_gain": self.contradiction_gain,
                "evidence_gain": self.evidence_gain,
                "confidence_persistence_gain": (
                    self.confidence_persistence_gain
                ),
                "inhibitory_membrane_decay": (
                    self.inhibitory_membrane_decay
                ),
                "inhibitory_trace_decay": self.inhibitory_trace_decay,
                "inhibitory_input_gain": self.inhibitory_input_gain,
            },
            "definitions": {
                hypothesis_id: {
                    "hypothesis_id": definition.hypothesis_id,
                    "text": definition.text,
                }
                for hypothesis_id, definition in self._definitions.items()
            },
            "populations": {
                hypothesis_id: {
                    "membrane": population.membrane.copy(),
                    "adaptation": population.adaptation.copy(),
                    "spikes": population.spikes.copy(),
                    "trace": population.trace,
                    "persistence": population.persistence,
                    "evidence_for": population.evidence_for,
                    "evidence_against": population.evidence_against,
                }
                for hypothesis_id, population in self._populations.items()
            },
            "inhibitory_population": {
                "membrane": self._inhibitory_population.membrane.copy(),
                "spikes": self._inhibitory_population.spikes.copy(),
                "trace": self._inhibitory_population.trace,
            },
            "last_step": self._last_step,
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        config = state["config"]
        self.neurons_per_hypothesis = int(
            config["neurons_per_hypothesis"]
        )
        self.inhibitory_neurons = int(config["inhibitory_neurons"])
        self.membrane_decay = float(config["membrane_decay"])
        self.trace_decay = float(config["trace_decay"])
        self.adaptation_decay = float(config["adaptation_decay"])
        self.adaptation_gain = float(config["adaptation_gain"])
        self.persistence_decay = float(config["persistence_decay"])
        self.persistence_gain = float(config["persistence_gain"])
        self.persistence_feedback = float(config["persistence_feedback"])
        self.contradiction_gain = float(config["contradiction_gain"])
        self.evidence_gain = float(config["evidence_gain"])
        self.confidence_persistence_gain = float(
            config["confidence_persistence_gain"]
        )
        self.inhibitory_membrane_decay = float(
            config["inhibitory_membrane_decay"]
        )
        self.inhibitory_trace_decay = float(
            config["inhibitory_trace_decay"]
        )
        self.inhibitory_input_gain = float(
            config["inhibitory_input_gain"]
        )
        self.set_mode(str(config["mode"]))

        self._definitions = {
            hypothesis_id: HypothesisDefinition(**payload)
            for hypothesis_id, payload in state["definitions"].items()
        }
        self._populations = {
            hypothesis_id: _PopulationState(
                membrane=[float(value) for value in payload["membrane"]],
                adaptation=[
                    float(value) for value in payload["adaptation"]
                ],
                spikes=[int(value) for value in payload["spikes"]],
                trace=float(payload["trace"]),
                persistence=float(payload["persistence"]),
                evidence_for=float(payload["evidence_for"]),
                evidence_against=float(payload["evidence_against"]),
            )
            for hypothesis_id, payload in state["populations"].items()
        }

        inhibitory_payload = state["inhibitory_population"]
        self._inhibitory_population = _InhibitoryPopulationState(
            membrane=[
                float(value) for value in inhibitory_payload["membrane"]
            ],
            spikes=[int(value) for value in inhibitory_payload["spikes"]],
            trace=float(inhibitory_payload["trace"]),
        )

        for population in self._populations.values():
            if not (
                len(population.membrane)
                == len(population.adaptation)
                == len(population.spikes)
                == self.neurons_per_hypothesis
            ):
                raise ValueError(
                    "population state does not match neurons_per_hypothesis"
                )

        if not (
            len(self._inhibitory_population.membrane)
            == len(self._inhibitory_population.spikes)
            == self.inhibitory_neurons
        ):
            raise ValueError(
                "inhibitory state does not match inhibitory_neurons"
            )

        self._last_step = int(state["last_step"])

    def _new_inhibitory_population(self) -> _InhibitoryPopulationState:
        return _InhibitoryPopulationState(
            membrane=[0.0] * self.inhibitory_neurons,
            spikes=[0] * self.inhibitory_neurons,
        )

    def _new_population(self) -> _PopulationState:
        return _PopulationState(
            membrane=[0.0] * self.neurons_per_hypothesis,
            adaptation=[0.0] * self.neurons_per_hypothesis,
            spikes=[0] * self.neurons_per_hypothesis,
        )

    def _advance_tick(self, signed_evidence: Mapping[str, float]) -> None:
        params = _MODE_PARAMETERS[self.mode]
        total_trace = sum(
            population.trace for population in self._populations.values()
        )

        inhibitory_membranes: list[float] = []
        inhibitory_spikes: list[int] = []
        for neuron_index, membrane in enumerate(
            self._inhibitory_population.membrane
        ):
            fraction = neuron_index / max(self.inhibitory_neurons - 1, 1)
            threshold = 0.90 + 0.20 * fraction
            updated_membrane = (
                self.inhibitory_membrane_decay * membrane
                + self.inhibitory_input_gain * total_trace
            )
            spike = int(updated_membrane >= threshold)
            if spike:
                updated_membrane = 0.0
            inhibitory_membranes.append(updated_membrane)
            inhibitory_spikes.append(spike)

        inhibitory_rate = (
            sum(inhibitory_spikes) / self.inhibitory_neurons
        )
        inhibitory_trace = (
            self.inhibitory_trace_decay
            * self._inhibitory_population.trace
            + inhibitory_rate
        )
        inhibition = params.inhibition_gain * inhibitory_trace

        next_values: dict[
            str,
            tuple[list[float], list[float], list[int], float, float],
        ] = {}

        for hypothesis_id, population in self._populations.items():
            evidence = signed_evidence.get(hypothesis_id, 0.0)
            membranes: list[float] = []
            adaptations: list[float] = []
            spikes: list[int] = []

            for neuron_index, (membrane, adaptation) in enumerate(
                zip(population.membrane, population.adaptation)
            ):
                fraction = neuron_index / max(
                    self.neurons_per_hypothesis - 1,
                    1,
                )
                heterogeneity = 0.94 + 0.12 * fraction
                threshold = (
                    params.threshold_scale * heterogeneity + adaptation
                )
                sensitivity = 1.04 - 0.08 * fraction
                drive = self.evidence_gain * evidence * sensitivity
                recurrent = params.recurrent_gain * population.trace
                persistence_drive = (
                    self.persistence_feedback * population.persistence
                )
                updated_membrane = (
                    self.membrane_decay * membrane
                    + drive
                    + recurrent
                    + persistence_drive
                    - inhibition
                )
                spike = int(updated_membrane >= threshold)
                if spike:
                    updated_membrane = 0.0
                updated_adaptation = (
                    self.adaptation_decay * adaptation
                    + self.adaptation_gain * spike
                )
                membranes.append(updated_membrane)
                adaptations.append(updated_adaptation)
                spikes.append(spike)

            spike_rate = sum(spikes) / self.neurons_per_hypothesis
            trace = self.trace_decay * population.trace + spike_rate
            persistence = (
                self.persistence_decay * population.persistence
                + self.persistence_gain * spike_rate
                - self.contradiction_gain * max(-evidence, 0.0)
            )
            next_values[hypothesis_id] = (
                membranes,
                adaptations,
                spikes,
                trace,
                max(persistence, 0.0),
            )

        self._inhibitory_population = _InhibitoryPopulationState(
            membrane=inhibitory_membranes,
            spikes=inhibitory_spikes,
            trace=inhibitory_trace,
        )

        for hypothesis_id, values in next_values.items():
            population = self._populations[hypothesis_id]
            (
                population.membrane,
                population.adaptation,
                population.spikes,
                population.trace,
                population.persistence,
            ) = values

    def _public_states(self) -> dict[str, HypothesisState]:
        states: dict[str, HypothesisState] = {}
        for hypothesis_id, population in self._populations.items():
            mean_positive_membrane = sum(
                max(value, 0.0) for value in population.membrane
            ) / self.neurons_per_hypothesis
            activation = (
                population.trace + 0.25 * mean_positive_membrane
            )
            confidence = (
                population.evidence_for - population.evidence_against
            )
            states[hypothesis_id] = HypothesisState(
                activation=activation,
                confidence=confidence,
                persistence=population.persistence,
                evidence_for=population.evidence_for,
                evidence_against=population.evidence_against,
            )
        return deepcopy(states)
