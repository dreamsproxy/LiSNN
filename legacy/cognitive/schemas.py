"""Serializable schemas for hypothesis-tracking research episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class HypothesisDefinition:
    """A hypothesis known to a cognitive substrate."""

    hypothesis_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must not be empty")
        if not self.text.strip():
            raise ValueError("hypothesis text must not be empty")


@dataclass(frozen=True)
class CognitiveEvent:
    """One timestamped piece of evidence applied to the workspace."""

    step: int
    observation: str
    supports: Mapping[str, float] = field(default_factory=dict)
    weakens: Mapping[str, float] = field(default_factory=dict)
    context_id: str = "default"
    features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if not self.observation.strip():
            raise ValueError("observation must not be empty")
        for collection_name, values in (("supports", self.supports), ("weakens", self.weakens)):
            for hypothesis_id, strength in values.items():
                if not hypothesis_id.strip():
                    raise ValueError(f"{collection_name} contains an empty hypothesis id")
                if strength < 0.0:
                    raise ValueError(f"{collection_name} strengths must be non-negative")


@dataclass
class HypothesisState:
    """Inspectable state exported by a cognitive substrate."""

    activation: float = 0.0
    confidence: float = 0.0
    persistence: float = 0.0
    evidence_for: float = 0.0
    evidence_against: float = 0.0


@dataclass(frozen=True)
class ResearchEpisode:
    """A complete benchmark episode and its expected final ranking."""

    episode_id: str
    hypotheses: Sequence[HypothesisDefinition]
    events: Sequence[CognitiveEvent]
    expected_final_ranking: Sequence[str]

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must not be empty")

        ids = [item.hypothesis_id for item in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("hypothesis ids must be unique")

        known = set(ids)
        if set(self.expected_final_ranking) != known:
            raise ValueError("expected_final_ranking must contain every hypothesis exactly once")

        previous_step = -1
        for event in self.events:
            if event.step <= previous_step:
                raise ValueError("events must have strictly increasing step values")
            previous_step = event.step
            referenced = set(event.supports) | set(event.weakens)
            unknown = referenced - known
            if unknown:
                raise ValueError(f"event references unknown hypotheses: {sorted(unknown)}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResearchEpisode":
        hypotheses = tuple(HypothesisDefinition(**item) for item in payload["hypotheses"])
        events = tuple(CognitiveEvent(**item) for item in payload["events"])
        return cls(
            episode_id=str(payload["episode_id"]),
            hypotheses=hypotheses,
            events=events,
            expected_final_ranking=tuple(payload["expected_final_ranking"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
