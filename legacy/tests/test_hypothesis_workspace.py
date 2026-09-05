"""Unit tests for the first hypothesis-workspace implementation slice."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from benchmarks.hypothesis_tracking.evaluator import (
    evaluate_episode,
    load_episode,
    pairwise_ranking_accuracy,
)
from cognitive.schemas import CognitiveEvent, HypothesisDefinition, ResearchEpisode
from substrates.rule_baseline import RuleHypothesisTracker


ROOT = Path(__file__).resolve().parents[1]
EPISODE_PATH = ROOT / "benchmarks" / "hypothesis_tracking" / "episodes" / "afgan_oep_001.json"


class SchemaTests(unittest.TestCase):
    def test_episode_rejects_unknown_hypothesis_reference(self) -> None:
        with self.assertRaises(ValueError):
            ResearchEpisode(
                episode_id="invalid",
                hypotheses=(HypothesisDefinition("H1", "Known"),),
                events=(
                    CognitiveEvent(
                        step=0,
                        observation="Unknown evidence target",
                        supports={"H2": 1.0},
                    ),
                ),
                expected_final_ranking=("H1",),
            )

    def test_episode_round_trip_is_json_serializable(self) -> None:
        episode = load_episode(EPISODE_PATH)
        encoded = json.dumps(episode.to_dict())
        decoded = ResearchEpisode.from_dict(json.loads(encoded))
        self.assertEqual(decoded, episode)


class BaselineTests(unittest.TestCase):
    def test_first_episode_reaches_expected_ranking(self) -> None:
        episode = load_episode(EPISODE_PATH)
        result = evaluate_episode(RuleHypothesisTracker(), episode)
        self.assertEqual(list(result.predicted_ranking), list(result.expected_ranking))
        self.assertEqual(result.pairwise_accuracy, 1.0)

    def test_state_export_import_preserves_ranking(self) -> None:
        episode = load_episode(EPISODE_PATH)
        original = RuleHypothesisTracker()
        original.register_hypotheses(episode.hypotheses)
        for event in episode.events[:2]:
            original.step(event)

        restored = RuleHypothesisTracker()
        restored.import_state(original.export_state())
        self.assertEqual(restored.rank(), original.rank())

        for event in episode.events[2:]:
            original.step(event)
            restored.step(event)
        self.assertEqual(restored.rank(), original.rank())

    def test_pairwise_accuracy_detects_partial_ordering(self) -> None:
        accuracy = pairwise_ranking_accuracy(
            predicted=("H2", "H3", "H1", "H4"),
            expected=("H2", "H1", "H3", "H4"),
        )
        self.assertGreater(accuracy, 0.0)
        self.assertLess(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
