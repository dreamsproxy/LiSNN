"""Benchmark loading and evaluation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cognitive.interfaces import CognitiveSubstrate
from cognitive.schemas import ResearchEpisode


@dataclass(frozen=True)
class EvaluationResult:
    episode_id: str
    predicted_ranking: Sequence[str]
    expected_ranking: Sequence[str]
    pairwise_accuracy: float


def load_episode(path: str | Path) -> ResearchEpisode:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ResearchEpisode.from_dict(payload)


def pairwise_ranking_accuracy(predicted: Sequence[str], expected: Sequence[str]) -> float:
    if set(predicted) != set(expected):
        raise ValueError("predicted and expected rankings must contain the same ids")

    expected_index = {item: index for index, item in enumerate(expected)}
    correct = 0
    total = 0
    for left_index, left in enumerate(predicted):
        for right in predicted[left_index + 1 :]:
            total += 1
            if expected_index[left] < expected_index[right]:
                correct += 1
    return 1.0 if total == 0 else correct / total


def evaluate_episode(substrate: CognitiveSubstrate, episode: ResearchEpisode) -> EvaluationResult:
    substrate.register_hypotheses(episode.hypotheses)
    for event in episode.events:
        substrate.step(event)
    substrate.consolidate()

    predicted = substrate.rank()
    accuracy = pairwise_ranking_accuracy(predicted, episode.expected_final_ranking)
    return EvaluationResult(
        episode_id=episode.episode_id,
        predicted_ranking=tuple(predicted),
        expected_ranking=tuple(episode.expected_final_ranking),
        pairwise_accuracy=accuracy,
    )
