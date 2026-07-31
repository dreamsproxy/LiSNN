from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import NetworkConfig
from .network import LanguageTrajectoryNetwork, PredictionResult
from .text import TextCorpus, TextTokenizer, Vocabulary


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    correct: int
    total: int
    mean_confidence: float
    cross_entropy: float
    perplexity: float
    predicted_token_ids: np.ndarray


class LanguageTrajectoryModel:
    """High-level next-token model over relative token-time trajectories."""

    def __init__(
        self,
        corpus: TextCorpus,
        config: NetworkConfig | None = None,
    ) -> None:
        self.corpus = corpus
        self.network = LanguageTrajectoryNetwork(
            vocabulary_size=corpus.vocabulary.size,
            config=config,
        )

    @property
    def vocabulary(self) -> Vocabulary:
        return self.corpus.vocabulary

    @property
    def tokenizer(self) -> TextTokenizer:
        return TextTokenizer(self.corpus.tokenizer_mode)

    def fit(self, *, epochs: int = 1, progress=None) -> None:
        self.network.fit(
            self.corpus.token_ids,
            epochs=epochs,
            progress=progress,
        )

    def predict_token_ids(
        self,
        context_token_ids: Sequence[int] | np.ndarray,
        *,
        temperature: float | None = None,
    ) -> PredictionResult:
        return self.network.predict_next(
            context_token_ids,
            temperature=temperature,
        )

    def predict_text(
        self,
        context_text: str,
        *,
        temperature: float | None = None,
    ) -> PredictionResult:
        tokens = self.tokenizer.tokenize(context_text)
        token_ids = self.vocabulary.encode(tokens)
        return self.predict_token_ids(
            token_ids,
            temperature=temperature,
        )

    def evaluate(
        self,
        *,
        temperature: float | None = None,
    ) -> EvaluationResult:
        sequence = self.corpus.token_ids
        predictions: list[int] = []
        confidences: list[float] = []
        losses: list[float] = []
        correct = 0
        for target_position in range(1, sequence.size):
            start = max(
                0,
                target_position - self.network.context_length,
            )
            context = sequence[start:target_position]
            result = self.predict_token_ids(
                context,
                temperature=temperature,
            )
            target = int(sequence[target_position])
            predictions.append(result.token_id)
            confidences.append(result.confidence)
            probability = max(
                float(result.probabilities[target]),
                1e-12,
            )
            losses.append(-np.log(probability))
            correct += int(result.token_id == target)
        total = sequence.size - 1
        cross_entropy = float(np.mean(losses))
        return EvaluationResult(
            accuracy=correct / total,
            correct=correct,
            total=total,
            mean_confidence=float(np.mean(confidences)),
            cross_entropy=cross_entropy,
            perplexity=float(
                np.exp(min(cross_entropy, 700.0))
            ),
            predicted_token_ids=np.asarray(
                predictions,
                dtype=np.int64,
            ),
        )

    def generate(
        self,
        prompt_text: str,
        *,
        num_tokens: int,
        temperature: float | None = None,
        sample: bool = False,
        seed: int | None = None,
    ) -> str:
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        prompt_tokens = self.tokenizer.tokenize(prompt_text)
        if not prompt_tokens:
            raise ValueError(
                "prompt_text must contain at least one token"
            )
        generated_ids = self.vocabulary.encode(
            prompt_tokens
        ).tolist()
        rng = np.random.default_rng(
            self.network.config.seed if seed is None else seed
        )
        for _ in range(num_tokens):
            result = self.predict_token_ids(
                generated_ids,
                temperature=temperature,
            )
            if sample:
                next_id = int(
                    rng.choice(
                        self.vocabulary.size,
                        p=result.probabilities,
                    )
                )
            else:
                next_id = result.token_id
            generated_ids.append(next_id)
        return TextTokenizer.detokenize(
            self.vocabulary.decode(generated_ids)
        )

    def save(self, path: str | Path) -> None:
        metadata = {
            "version": 2.2,
            "config": self.network.config.to_dict(),
            "tokenizer_mode": self.corpus.tokenizer_mode,
            "vocabulary": list(self.vocabulary.tokens),
            "source_paths": list(self.corpus.source_paths),
        }
        np.savez_compressed(
            Path(path),
            metadata=np.asarray(json.dumps(metadata)),
            token_ids=self.corpus.token_ids,
            neurons=self.network.neurons,
            thresholds=self.network.thresholds,
            pre_tau=self.network.pre_tau,
            post_tau=self.network.post_tau,
            recurrent_weights=self.network.weights.weights,
            hebb_weights=self.network.weights.hebb_weights,
            neuron_types=self.network.weights.neuron_types,
            readout_weights=self.network.readout_weights,
            readout_bias=self.network.readout_bias,
            global_step_tick=np.asarray(
                self.network.global_step_tick,
                dtype=np.int64,
            ),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "LanguageTrajectoryModel":
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            metadata = json.loads(
                str(checkpoint["metadata"].item())
            )
            config_data = dict(metadata["config"])
            if "ei_ratio" not in config_data:
                config_data["ei_ratio"] = config_data.pop(
                    "inhibitory_fraction",
                    0.8,
                )

            vocabulary = Vocabulary(
                tuple(metadata["vocabulary"])
            )
            token_ids = checkpoint["token_ids"].astype(np.int64)
            corpus = TextCorpus(
                tokens=tuple(vocabulary.decode(token_ids)),
                token_ids=token_ids,
                vocabulary=vocabulary,
                source_paths=tuple(
                    metadata.get("source_paths", ())
                ),
                tokenizer_mode=metadata["tokenizer_mode"],
            )
            model = cls(
                corpus,
                NetworkConfig(**config_data),
            )
            model.network.neurons[...] = checkpoint["neurons"]
            model.network.thresholds[...] = checkpoint["thresholds"]
            model.network.pre_tau[...] = checkpoint["pre_tau"]
            model.network.post_tau[...] = checkpoint["post_tau"]
            model.network.weights.weights[...] = checkpoint[
                "recurrent_weights"
            ]
            model.network.weights.hebb_weights[...] = checkpoint[
                "hebb_weights"
            ]
            if "neuron_types" in checkpoint.files:
                model.network.weights.neuron_types[...] = checkpoint[
                    "neuron_types"
                ].astype(np.int8)
            model.network.weights._enforce_neuron_types()
            model.network.readout_weights[...] = checkpoint[
                "readout_weights"
            ]
            model.network.readout_bias[...] = checkpoint[
                "readout_bias"
            ]
            model.network.global_step_tick = int(
                checkpoint["global_step_tick"].item()
            )
        return model
