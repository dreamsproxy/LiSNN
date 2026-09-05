"""Deterministic baseline for hypothesis tracking.

This implementation is deliberately simple. It establishes the minimum
behavior that later LiNN and LiSNN substrates must exceed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from cognitive.interfaces import CognitiveSubstrate
from cognitive.schemas import CognitiveEvent, HypothesisDefinition, HypothesisState


class RuleHypothesisTracker(CognitiveSubstrate):
    """Accumulate positive and negative evidence with configurable decay."""

    def __init__(
        self,
        *,
        activation_decay: float = 0.85,
        persistence_gain: float = 0.25,
        persistence_decay: float = 0.98,
        confidence_scale: float = 1.0,
    ) -> None:
        if not 0.0 <= activation_decay <= 1.0:
            raise ValueError("activation_decay must be within [0, 1]")
        if not 0.0 <= persistence_decay <= 1.0:
            raise ValueError("persistence_decay must be within [0, 1]")
        if persistence_gain < 0.0:
            raise ValueError("persistence_gain must be non-negative")
        if confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be positive")

        self.activation_decay = activation_decay
        self.persistence_gain = persistence_gain
        self.persistence_decay = persistence_decay
        self.confidence_scale = confidence_scale
        self._definitions: dict[str, HypothesisDefinition] = {}
        self._states: dict[str, HypothesisState] = {}
        self._last_step = -1

    def reset(self) -> None:
        self._states = {hypothesis_id: HypothesisState() for hypothesis_id in self._definitions}
        self._last_step = -1

    def register_hypotheses(self, hypotheses: Sequence[HypothesisDefinition]) -> None:
        definitions = {item.hypothesis_id: item for item in hypotheses}
        if len(definitions) != len(hypotheses):
            raise ValueError("hypothesis ids must be unique")
        self._definitions = definitions
        self.reset()

    def step(self, event: CognitiveEvent) -> Mapping[str, HypothesisState]:
        if event.step <= self._last_step:
            raise ValueError("events must be applied in strictly increasing step order")
        if not self._states:
            raise RuntimeError("register_hypotheses must be called before step")

        unknown = (set(event.supports) | set(event.weakens)) - set(self._states)
        if unknown:
            raise ValueError(f"event references unknown hypotheses: {sorted(unknown)}")

        elapsed = event.step - self._last_step if self._last_step >= 0 else 1
        activation_factor = self.activation_decay ** elapsed
        persistence_factor = self.persistence_decay ** elapsed

        for state in self._states.values():
            state.activation *= activation_factor
            state.persistence *= persistence_factor

        for hypothesis_id, strength in event.supports.items():
            state = self._states[hypothesis_id]
            state.activation += strength
            state.evidence_for += strength

        for hypothesis_id, strength in event.weakens.items():
            state = self._states[hypothesis_id]
            state.activation -= strength
            state.evidence_against += strength

        for state in self._states.values():
            net_evidence = state.evidence_for - state.evidence_against
            state.confidence = net_evidence / self.confidence_scale
            state.persistence += self.persistence_gain * max(state.activation, 0.0)

        self._last_step = event.step
        return deepcopy(self._states)

    def consolidate(self) -> None:
        for state in self._states.values():
            state.persistence += self.persistence_gain * max(state.confidence, 0.0)

    def rank(self) -> list[str]:
        return sorted(
            self._states,
            key=lambda hypothesis_id: (
                self._states[hypothesis_id].confidence,
                self._states[hypothesis_id].persistence,
                self._states[hypothesis_id].activation,
                hypothesis_id,
            ),
            reverse=True,
        )

    def export_state(self) -> dict[str, Any]:
        return {
            "config": {
                "activation_decay": self.activation_decay,
                "persistence_gain": self.persistence_gain,
                "persistence_decay": self.persistence_decay,
                "confidence_scale": self.confidence_scale,
            },
            "definitions": {
                hypothesis_id: {
                    "hypothesis_id": definition.hypothesis_id,
                    "text": definition.text,
                }
                for hypothesis_id, definition in self._definitions.items()
            },
            "states": {
                hypothesis_id: vars(state).copy()
                for hypothesis_id, state in self._states.items()
            },
            "last_step": self._last_step,
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        config = state["config"]
        self.activation_decay = float(config["activation_decay"])
        self.persistence_gain = float(config["persistence_gain"])
        self.persistence_decay = float(config["persistence_decay"])
        self.confidence_scale = float(config["confidence_scale"])
        self._definitions = {
            hypothesis_id: HypothesisDefinition(**payload)
            for hypothesis_id, payload in state["definitions"].items()
        }
        self._states = {
            hypothesis_id: HypothesisState(**payload)
            for hypothesis_id, payload in state["states"].items()
        }
        self._last_step = int(state["last_step"])
