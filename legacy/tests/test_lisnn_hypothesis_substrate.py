"""Behavioral tests for the first LiSNN hypothesis substrate."""

from __future__ import annotations

import unittest
from pathlib import Path

from benchmarks.hypothesis_tracking.evaluator import evaluate_episode, load_episode
from cognitive.interfaces import CognitiveSubstrate
from cognitive.schemas import CognitiveEvent, HypothesisDefinition
from substrates.lisnn_hypothesis import LiSNNHypothesisSubstrate


ROOT = Path(__file__).resolve().parents[1]
EPISODE_PATH = (
    ROOT
    / "benchmarks"
    / "hypothesis_tracking"
    / "episodes"
    / "afgan_oep_001.json"
)

HYPOTHESES = (
    HypothesisDefinition("H1", "First candidate mechanism."),
    HypothesisDefinition("H2", "Second candidate mechanism."),
    HypothesisDefinition("H3", "Third candidate mechanism."),
)


class LiSNNHypothesisSubstrateTests(unittest.TestCase):
    def test_implements_common_substrate_interface(self) -> None:
        self.assertIsInstance(LiSNNHypothesisSubstrate(), CognitiveSubstrate)

    def test_supported_hypothesis_persists_through_silent_ticks(self) -> None:
        substrate = LiSNNHypothesisSubstrate(mode="balanced")
        substrate.register_hypotheses(HYPOTHESES)
        substrate.step(
            CognitiveEvent(
                step=0,
                observation="Strong initial support.",
                supports={"H1": 2.0},
            )
        )

        state = substrate.step(
            CognitiveEvent(
                step=25,
                observation="A long interval passes without new evidence.",
            )
        )

        self.assertGreater(state["H1"].persistence, 0.0)
        self.assertGreater(state["H1"].activation, 0.0)
        self.assertEqual(substrate.rank()[0], "H1")

    def test_contradiction_reduces_confidence_without_erasing_state(self) -> None:
        substrate = LiSNNHypothesisSubstrate()
        substrate.register_hypotheses(HYPOTHESES)

        before = substrate.step(
            CognitiveEvent(
                step=0,
                observation="Initial evidence.",
                supports={"H1": 2.0},
            )
        )["H1"]
        after = substrate.step(
            CognitiveEvent(
                step=1,
                observation="Partially contradicting evidence.",
                weakens={"H1": 0.75},
            )
        )["H1"]

        self.assertLess(after.confidence, before.confidence)
        self.assertGreater(after.confidence, 0.0)
        self.assertGreater(after.persistence, 0.0)

    def test_divergence_mode_allows_competing_hypotheses_to_coexist(self) -> None:
        substrate = LiSNNHypothesisSubstrate(mode="divergence")
        substrate.register_hypotheses(HYPOTHESES)

        state = substrate.step(
            CognitiveEvent(
                step=0,
                observation="Two mechanisms receive comparable evidence.",
                supports={"H1": 1.2, "H2": 1.2},
            )
        )

        self.assertGreater(state["H1"].activation, 0.0)
        self.assertGreater(state["H2"].activation, 0.0)
        self.assertGreater(state["H1"].persistence, 0.0)
        self.assertGreater(state["H2"].persistence, 0.0)

    def test_divergence_preserves_broader_activation_than_convergence(self) -> None:
        event = CognitiveEvent(
            step=0,
            observation="One strong candidate and two weaker alternatives.",
            supports={"H1": 1.4, "H2": 0.9, "H3": 0.9},
        )
        divergent = LiSNNHypothesisSubstrate(mode="divergence")
        convergent = LiSNNHypothesisSubstrate(mode="convergence")
        divergent.register_hypotheses(HYPOTHESES)
        convergent.register_hypotheses(HYPOTHESES)

        divergent_state = divergent.step(event)
        convergent_state = convergent.step(event)

        self.assertGreater(
            divergent_state["H2"].activation,
            convergent_state["H2"].activation,
        )
        self.assertGreater(
            divergent_state["H3"].activation,
            convergent_state["H3"].activation,
        )

    def test_state_export_import_continues_deterministically(self) -> None:
        original = LiSNNHypothesisSubstrate(mode="divergence")
        original.register_hypotheses(HYPOTHESES)
        original.step(
            CognitiveEvent(
                step=0,
                observation="First event.",
                supports={"H1": 1.4, "H2": 0.8},
            )
        )
        original.step(
            CognitiveEvent(
                step=4,
                observation="Second event.",
                supports={"H2": 1.1},
                weakens={"H1": 0.2},
            )
        )

        restored = LiSNNHypothesisSubstrate()
        restored.import_state(original.export_state())

        next_event = CognitiveEvent(
            step=9,
            observation="Third event after restoration.",
            supports={"H3": 1.3},
            weakens={"H2": 0.1},
        )
        original.step(next_event)
        restored.step(next_event)

        self.assertEqual(restored.export_state(), original.export_state())
        self.assertEqual(restored.rank(), original.rank())

    def test_export_contains_expected_population_sizes(self) -> None:
        substrate = LiSNNHypothesisSubstrate(
            neurons_per_hypothesis=16,
            inhibitory_neurons=8,
        )
        substrate.register_hypotheses(HYPOTHESES)
        state = substrate.export_state()

        for population in state["populations"].values():
            self.assertEqual(len(population["membrane"]), 16)
            self.assertEqual(len(population["adaptation"]), 16)
            self.assertEqual(len(population["spikes"]), 16)
        self.assertEqual(len(state["inhibitory_population"]["membrane"]), 8)
        self.assertEqual(len(state["inhibitory_population"]["spikes"]), 8)

    def test_afgan_episode_reaches_reference_ranking(self) -> None:
        episode = load_episode(EPISODE_PATH)
        result = evaluate_episode(LiSNNHypothesisSubstrate(), episode)

        self.assertEqual(
            list(result.predicted_ranking),
            list(result.expected_ranking),
        )
        self.assertEqual(result.pairwise_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
