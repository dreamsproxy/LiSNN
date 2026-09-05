"""Abstract contract shared by rule, recurrent, LiNN, and LiSNN substrates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from .schemas import CognitiveEvent, HypothesisDefinition, HypothesisState


class CognitiveSubstrate(ABC):
    """Stateful hypothesis workspace interface."""

    @abstractmethod
    def reset(self) -> None:
        """Reset dynamic state while preserving configuration."""

    @abstractmethod
    def register_hypotheses(self, hypotheses: Sequence[HypothesisDefinition]) -> None:
        """Replace the currently registered hypothesis set."""

    @abstractmethod
    def step(self, event: CognitiveEvent) -> Mapping[str, HypothesisState]:
        """Apply one event and return the updated hypothesis states."""

    @abstractmethod
    def consolidate(self) -> None:
        """Commit transient state into slower persistent state."""

    @abstractmethod
    def rank(self) -> list[str]:
        """Return hypothesis ids from strongest to weakest."""

    @abstractmethod
    def export_state(self) -> dict[str, Any]:
        """Export all state required to resume the workspace."""

    @abstractmethod
    def import_state(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by export_state."""
