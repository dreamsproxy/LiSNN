"""Run one hypothesis-tracking episode against the rule baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.hypothesis_tracking.evaluator import evaluate_episode, load_episode
from substrates.rule_baseline import RuleHypothesisTracker


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = load_episode(args.episode)
    result = evaluate_episode(RuleHypothesisTracker(), episode)

    print(f"Episode: {result.episode_id}")
    print(f"Predicted: {' > '.join(result.predicted_ranking)}")
    print(f"Expected:  {' > '.join(result.expected_ranking)}")
    print(f"Pairwise ranking accuracy: {result.pairwise_accuracy:.3f}")


if __name__ == "__main__":
    main()
