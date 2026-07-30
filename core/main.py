from __future__ import annotations

import argparse
from pathlib import Path

from .config import NetworkConfig
from .model import LanguageTrajectoryModel
from .text import TextCorpus, TextTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train LiSNN v2 to predict the next token from a recent "
            "relative-time token trajectory."
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
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--ticks-per-token", type=int, default=8)
    parser.add_argument("--prediction-ticks", type=int, default=16)
    parser.add_argument("--signal-scale", type=float, default=500.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--generate-tokens", type=int, default=64)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--checkpoint", default="language_trajectory_model.npz")
    parser.add_argument(
        "--output",
        default="language_next_token_generation.txt",
    )
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
        context_length=args.context_length,
        ticks_per_token=args.ticks_per_token,
        prediction_ticks=args.prediction_ticks,
        signal_scale=args.signal_scale,
        temperature=args.temperature,
        seed=args.seed,
    )
    model = LanguageTrajectoryModel(corpus, config)
    memory_mib = model.network.estimated_dense_memory_bytes / (1024 ** 2)
    print(
        f"Loaded {corpus.sequence_length} tokens from "
        f"{len(corpus.source_paths)} file(s); vocabulary={corpus.vocabulary.size}."
    )
    print(
        f"Context length={config.context_length}; "
        f"neurons={model.network.num_neurons}; "
        f"estimated dense state={memory_mib:.2f} MiB."
    )

    last_percent = -1

    def progress(completed: int, total: int) -> None:
        nonlocal last_percent
        percent = int((completed / total) * 100)
        if percent != last_percent and (percent % 5 == 0 or completed == total):
            print(f"Training: {completed}/{total} windows ({percent}%)")
            last_percent = percent

    model.fit(epochs=args.epochs, progress=progress)
    evaluation = model.evaluate(temperature=args.temperature)
    print(
        f"Next-token accuracy: {evaluation.correct}/{evaluation.total} "
        f"({evaluation.accuracy:.3%})"
    )
    print(f"Mean top-1 confidence: {evaluation.mean_confidence:.6f}")
    print(f"Cross-entropy: {evaluation.cross_entropy:.6f}")
    print(f"Perplexity: {evaluation.perplexity:.6f}")

    if args.prompt is None:
        prompt_ids = corpus.token_ids[
            : min(config.context_length, corpus.sequence_length)
        ]
        prompt = TextTokenizer.detokenize(corpus.vocabulary.decode(prompt_ids))
    else:
        prompt = args.prompt
    generated = model.generate(
        prompt,
        num_tokens=args.generate_tokens,
        temperature=args.temperature,
        sample=args.sample,
        seed=args.seed,
    )
    Path(args.output).write_text(generated, encoding="utf-8")
    model.save(args.checkpoint)
    print(f"Prompt: {prompt!r}")
    print(f"Generated text written to {args.output}")
    print(f"Checkpoint written to {args.checkpoint}")


if __name__ == "__main__":
    main()
