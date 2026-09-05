"""Run one hypothesis-tracking episode against the LiSNN substrate."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.hypothesis_tracking.evaluator import (
    load_episode,
    pairwise_ranking_accuracy,
)
from substrates.lisnn_hypothesis import LiSNNHypothesisSubstrate


DEFAULT_EPISODE = Path(__file__).with_name("episodes") / "afgan_oep_001.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode",
        nargs="?",
        type=Path,
        default=DEFAULT_EPISODE,
        help="Path to a research episode JSON file.",
    )
    parser.add_argument(
        "--mode",
        choices=("divergence", "balanced", "convergence"),
        default="balanced",
        help="Cognitive operating mode used for the episode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = load_episode(args.episode)
    substrate = LiSNNHypothesisSubstrate(mode=args.mode)
    substrate.register_hypotheses(episode.hypotheses)

    for event in episode.events:
        substrate.step(event)
    substrate.consolidate()

    predicted = substrate.rank()
    accuracy = pairwise_ranking_accuracy(
        predicted,
        episode.expected_final_ranking,
    )
    exported = substrate.export_state()

    print(f"Episode: {episode.episode_id}")
    print(f"Mode: {substrate.mode}")
    print(f"Predicted: {' > '.join(predicted)}")
    print(f"Expected:  {' > '.join(episode.expected_final_ranking)}")
    print(f"Pairwise ranking accuracy: {accuracy:.3f}")
    print("Final hypothesis state:")

    for hypothesis_id in predicted:
        population = exported["populations"][hypothesis_id]
        confidence = (
            population["evidence_for"] - population["evidence_against"]
        )
        print(
            f"  {hypothesis_id}: "
            f"confidence={confidence:.3f}, "
            f"trace={population['trace']:.3f}, "
            f"persistence={population['persistence']:.3f}"
        )

    inhibitory = exported["inhibitory_population"]
    print(f"Global inhibitory trace: {inhibitory['trace']:.3f}")


if __name__ == "__main__":
    main()
