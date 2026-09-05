from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import NetworkConfig
from .network import LanguageTimecodeNetwork, RecallResult
from .text import TextCorpus, TextTokenizer, Vocabulary


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    correct: int
    total: int
    mean_confidence: float
    reconstructed_text: str
    recalled_token_ids: np.ndarray


class LanguageTimecodeModel:
    """High-level text wrapper around the recurrent timecode network."""

    def __init__(
        self,
        corpus: TextCorpus,
        config: NetworkConfig | None = None,
    ) -> None:
        self.corpus = corpus
        self.network = LanguageTimecodeNetwork(
            vocabulary_size=corpus.vocabulary.size,
            sequence_length=corpus.sequence_length,
            config=config,
        )

    @property
    def vocabulary(self) -> Vocabulary:
        return self.corpus.vocabulary

    def fit(
        self,
        *,
        epochs: int = 1,
        ticks_per_token: int | None = None,
        progress=None,
    ) -> None:
        self.network.fit(
            self.corpus.token_ids,
            epochs=epochs,
            ticks_per_token=ticks_per_token,
            progress=progress,
        )

    def recall(
        self,
        positions: Iterable[int] | None = None,
        *,
        num_ticks: int | None = None,
    ) -> list[RecallResult]:
        return self.network.recall_sequence(positions, num_ticks=num_ticks)

    def reconstruct_text(self, *, num_ticks: int | None = None) -> str:
        results = self.recall(num_ticks=num_ticks)
        tokens = self.vocabulary.decode([result.token_id for result in results])
        return TextTokenizer.detokenize(tokens)

    def evaluate(self, *, num_ticks: int | None = None) -> EvaluationResult:
        results = self.recall(num_ticks=num_ticks)
        recalled = np.asarray([result.token_id for result in results], dtype=np.int64)
        expected = self.corpus.token_ids
        correct = int(np.sum(recalled == expected))
        total = int(expected.size)
        confidences = [result.confidence for result in results]
        reconstructed = TextTokenizer.detokenize(self.vocabulary.decode(recalled))
        return EvaluationResult(
            accuracy=correct / total,
            correct=correct,
            total=total,
            mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
            reconstructed_text=reconstructed,
            recalled_token_ids=recalled,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        metadata = {
            "config": self.network.config.to_dict(),
            "tokenizer_mode": self.corpus.tokenizer_mode,
            "vocabulary": list(self.vocabulary.tokens),
            "source_paths": list(self.corpus.source_paths),
        }
        np.savez_compressed(
            destination,
            metadata=np.asarray(json.dumps(metadata)),
            token_ids=self.corpus.token_ids,
            neurons=self.network.neurons,
            thresholds=self.network.thresholds,
            pre_tau=self.network.pre_tau,
            post_tau=self.network.post_tau,
            pre_spikes=self.network.pre_spikes,
            post_spikes=self.network.post_spikes,
            weights=self.network.weights.weights,
            hebb_weights=self.network.weights.hebb_weights,
            global_step_tick=np.asarray(self.network.global_step_tick, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LanguageTimecodeModel":
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"].item()))
            vocabulary = Vocabulary(tuple(metadata["vocabulary"]))
            token_ids = checkpoint["token_ids"].astype(np.int64)
            tokens = tuple(vocabulary.decode(token_ids))
            corpus = TextCorpus(
                tokens=tokens,
                token_ids=token_ids,
                vocabulary=vocabulary,
                source_paths=tuple(metadata.get("source_paths", ())),
                tokenizer_mode=metadata["tokenizer_mode"],
            )
            model = cls(corpus, NetworkConfig(**metadata["config"]))
            model.network.neurons[...] = checkpoint["neurons"]
            model.network.thresholds[...] = checkpoint["thresholds"]
            model.network.pre_tau[...] = checkpoint["pre_tau"]
            model.network.post_tau[...] = checkpoint["post_tau"]
            model.network.pre_spikes[...] = checkpoint["pre_spikes"]
            model.network.post_spikes[...] = checkpoint["post_spikes"]
            model.network.weights.weights[...] = checkpoint["weights"]
            model.network.weights.hebb_weights[...] = checkpoint["hebb_weights"]
            model.network.global_step_tick = int(checkpoint["global_step_tick"].item())
        return model
