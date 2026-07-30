from __future__ import annotations

import argparse
from pathlib import Path

from .config import NetworkConfig
from .model import LanguageTimecodeModel
from .text import TextCorpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the LiSNN language timecode baseline and reconstruct the "
            "training sequence from position codes."
        )
    )
    parser.add_argument("--dataset-dir", default="datasets")
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument(
        "--tokenizer",
        choices=("character", "word"),
        default="character",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--ticks-per-token", type=int, default=32)
    parser.add_argument("--recall-ticks", type=int, default=128)
    parser.add_argument("--signal-scale", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="language_timecode_model.npz")
    parser.add_argument("--output", default="language_timecode_recall.txt")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    corpus = TextCorpus.from_directory(
        args.dataset_dir,
        tokenizer_mode=args.tokenizer,
        max_tokens=args.max_tokens,
        pattern=args.pattern,
    )
    config = NetworkConfig(
        ticks_per_token=args.ticks_per_token,
        recall_ticks=args.recall_ticks,
        signal_scale=args.signal_scale,
        seed=args.seed,
    )
    model = LanguageTimecodeModel(corpus, config)

    memory_mib = model.network.estimated_dense_memory_bytes / (1024 ** 2)
    print(
        f"Loaded {corpus.sequence_length} tokens from "
        f"{len(corpus.source_paths)} file(s); vocabulary={corpus.vocabulary.size}."
    )
    print(
        f"Network neurons={model.network.num_neurons}; "
        f"dense plasticity matrices approximately {memory_mib:.2f} MiB."
    )

    last_percent = -1

    def progress(completed: int, total: int) -> None:
        nonlocal last_percent
        percent = int((completed / total) * 100)
        if percent != last_percent and (percent % 5 == 0 or completed == total):
            print(f"Training: {completed}/{total} positions ({percent}%)")
            last_percent = percent

    model.fit(
        epochs=args.epochs,
        ticks_per_token=args.ticks_per_token,
        progress=progress,
    )
    evaluation = model.evaluate(num_ticks=args.recall_ticks)

    Path(args.output).write_text(
        evaluation.reconstructed_text,
        encoding="utf-8",
    )
    model.save(args.checkpoint)

    print(
        f"Recall accuracy: {evaluation.correct}/{evaluation.total} "
        f"({evaluation.accuracy:.3%})"
    )
    print(f"Mean recall confidence: {evaluation.mean_confidence:.6f}")
    print(f"Reconstructed text written to {args.output}")
    print(f"Checkpoint written to {args.checkpoint}")


if __name__ == "__main__":
    main()
