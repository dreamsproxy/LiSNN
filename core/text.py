from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_WORD_PATTERN = re.compile(r"\s+|[\w]+|[^\w\s]", flags=re.UNICODE)


class TextTokenizer:
    """Small deterministic tokenizer with exact detokenization."""

    MODES = {"character", "word"}

    def __init__(self, mode: str = "character") -> None:
        if mode not in self.MODES:
            choices = ", ".join(sorted(self.MODES))
            raise ValueError(f"unknown tokenizer mode {mode!r}; choose {choices}")
        self.mode = mode

    def tokenize(self, text: str) -> list[str]:
        if self.mode == "character":
            return list(text)
        return _WORD_PATTERN.findall(text)

    @staticmethod
    def detokenize(tokens: Iterable[str]) -> str:
        return "".join(tokens)


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("vocabulary cannot be empty")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("vocabulary tokens must be unique")

    @classmethod
    def build(cls, tokens: Sequence[str]) -> "Vocabulary":
        if not tokens:
            raise ValueError("cannot build a vocabulary from an empty sequence")
        return cls(tuple(sorted(set(tokens))))

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def encode(self, tokens: Sequence[str]) -> np.ndarray:
        lookup = self.token_to_id
        try:
            return np.asarray([lookup[token] for token in tokens], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"token is not in the vocabulary: {exc.args[0]!r}") from exc

    def decode(self, token_ids: Sequence[int] | np.ndarray) -> list[str]:
        decoded: list[str] = []
        for raw_id in token_ids:
            token_id = int(raw_id)
            if token_id < 0 or token_id >= self.size:
                raise ValueError(f"token id out of range: {token_id}")
            decoded.append(self.tokens[token_id])
        return decoded


@dataclass(frozen=True)
class TextCorpus:
    tokens: tuple[str, ...]
    token_ids: np.ndarray
    vocabulary: Vocabulary
    source_paths: tuple[str, ...]
    tokenizer_mode: str

    @property
    def text(self) -> str:
        return TextTokenizer.detokenize(self.tokens)

    @property
    def sequence_length(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_text(
        cls,
        text: str,
        tokenizer_mode: str = "character",
        max_tokens: int | None = None,
        source_paths: Sequence[str] = (),
    ) -> "TextCorpus":
        tokenizer = TextTokenizer(tokenizer_mode)
        tokens = tokenizer.tokenize(text)
        if max_tokens is not None:
            if max_tokens < 1:
                raise ValueError("max_tokens must be at least 1")
            tokens = tokens[:max_tokens]
        if not tokens:
            raise ValueError("the corpus contains no tokens")
        vocabulary = Vocabulary.build(tokens)
        token_ids = vocabulary.encode(tokens)
        return cls(
            tokens=tuple(tokens),
            token_ids=token_ids,
            vocabulary=vocabulary,
            source_paths=tuple(source_paths),
            tokenizer_mode=tokenizer_mode,
        )

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        tokenizer_mode: str = "character",
        max_tokens: int | None = None,
        pattern: str = "*.txt",
        separator: str = "\n",
    ) -> "TextCorpus":
        root = Path(directory)
        paths = sorted(path for path in root.glob(pattern) if path.is_file())
        if not paths:
            raise FileNotFoundError(f"no files matching {pattern!r} in {root}")
        texts = [path.read_text(encoding="utf-8", errors="replace") for path in paths]
        return cls.from_text(
            separator.join(texts),
            tokenizer_mode=tokenizer_mode,
            max_tokens=max_tokens,
            source_paths=[str(path) for path in paths],
        )
