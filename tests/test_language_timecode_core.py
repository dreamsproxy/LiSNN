from pathlib import Path

import numpy as np

from core.v1 import LanguageTimecodeModel, NetworkConfig, TextCorpus, TextTokenizer


def small_config(seed: int = 7) -> NetworkConfig:
    return NetworkConfig(
        ticks_per_token=1,
        recall_ticks=2,
        signal_scale=100.0,
        clip_interval=2,
        prune_interval=2,
        top_k_fraction=0.5,
        seed=seed,
    )


def test_character_tokenizer_roundtrip() -> None:
    text = "Hello, LiSNN!\n"
    tokenizer = TextTokenizer("character")
    assert tokenizer.detokenize(tokenizer.tokenize(text)) == text


def test_word_tokenizer_roundtrip_preserves_spacing() -> None:
    text = "Hello,  LiSNN!\nNext line."
    tokenizer = TextTokenizer("word")
    assert tokenizer.detokenize(tokenizer.tokenize(text)) == text


def test_network_fit_and_recall_are_finite() -> None:
    corpus = TextCorpus.from_text("abca", tokenizer_mode="character")
    model = LanguageTimecodeModel(corpus, small_config())
    model.fit()
    results = model.recall(num_ticks=2)

    assert len(results) == corpus.sequence_length
    assert all(0 <= result.token_id < corpus.vocabulary.size for result in results)
    assert all(np.all(np.isfinite(result.token_scores)) for result in results)
    assert np.all(np.isfinite(model.network.weights.weights))


def test_seed_reproduces_initial_state() -> None:
    corpus = TextCorpus.from_text("abc", tokenizer_mode="character")
    first = LanguageTimecodeModel(corpus, small_config(seed=11))
    second = LanguageTimecodeModel(corpus, small_config(seed=11))

    np.testing.assert_allclose(
        first.network.weights.weights,
        second.network.weights.weights,
    )
    np.testing.assert_allclose(first.network.neurons, second.network.neurons)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    corpus = TextCorpus.from_text("abba", tokenizer_mode="character")
    model = LanguageTimecodeModel(corpus, small_config())
    model.fit()
    checkpoint = tmp_path / "model.npz"
    model.save(checkpoint)

    restored = LanguageTimecodeModel.load(checkpoint)
    assert restored.corpus.tokens == model.corpus.tokens
    np.testing.assert_allclose(
        restored.network.weights.weights,
        model.network.weights.weights,
    )
    np.testing.assert_allclose(restored.network.neurons, model.network.neurons)


def test_toy_sequence_is_recalled_from_timecodes() -> None:
    corpus = TextCorpus.from_text("abca", tokenizer_mode="character")
    config = NetworkConfig(
        ticks_per_token=16,
        recall_ticks=32,
        signal_scale=500.0,
        seed=0,
    )
    model = LanguageTimecodeModel(corpus, config)
    model.fit()
    evaluation = model.evaluate()

    assert evaluation.accuracy == 1.0
    assert evaluation.reconstructed_text == "abca"
