from pathlib import Path

import numpy as np

from core import LanguageTrajectoryModel, NetworkConfig, TextCorpus


def small_config(seed: int = 7) -> NetworkConfig:
    return NetworkConfig(
        context_length=2,
        ticks_per_token=1,
        prediction_ticks=2,
        signal_scale=500.0,
        readout_alpha=0.5,
        anti_hebbian_alpha=0.2,
        recency_decay=1.0,
        clip_interval=2,
        prune_interval=32,
        top_k_fraction=0.5,
        seed=seed,
    )


def test_layout_contains_relative_token_time_bindings() -> None:
    corpus = TextCorpus.from_text("abcabc")
    model = LanguageTrajectoryModel(corpus, small_config())
    network = model.network
    expected = corpus.vocabulary.size + 2 + corpus.vocabulary.size * 2 + 1
    assert network.num_neurons == expected
    assert network.binding_index(0, 0) != network.binding_index(0, 1)


def test_prediction_probabilities_are_finite_and_normalized() -> None:
    corpus = TextCorpus.from_text("abcabc")
    model = LanguageTrajectoryModel(corpus, small_config())
    model.fit(epochs=1)
    result = model.predict_token_ids(corpus.token_ids[:2])
    assert np.all(np.isfinite(result.probabilities))
    assert np.isclose(result.probabilities.sum(), 1.0)
    assert 0 <= result.token_id < corpus.vocabulary.size


def test_repeating_trajectory_learns_next_token() -> None:
    corpus = TextCorpus.from_text("abcabcabcabcabc")
    model = LanguageTrajectoryModel(corpus, small_config(seed=0))
    model.fit(epochs=8)
    evaluation = model.evaluate()
    assert evaluation.accuracy >= 0.85
    result = model.predict_text("ab")
    predicted = corpus.vocabulary.decode([result.token_id])[0]
    assert predicted == "c"


def test_checkpoint_roundtrip_preserves_predictions(tmp_path: Path) -> None:
    corpus = TextCorpus.from_text("abcabcabc")
    model = LanguageTrajectoryModel(corpus, small_config(seed=3))
    model.fit(epochs=3)
    before = model.predict_text("ab")
    checkpoint = tmp_path / "trajectory.npz"
    model.save(checkpoint)
    restored = LanguageTrajectoryModel.load(checkpoint)
    after = restored.predict_text("ab")
    np.testing.assert_allclose(after.probabilities, before.probabilities)
    assert after.token_id == before.token_id


def test_generation_extends_prompt() -> None:
    corpus = TextCorpus.from_text("abcabcabcabc")
    model = LanguageTrajectoryModel(corpus, small_config(seed=5))
    model.fit(epochs=6)
    generated = model.generate("ab", num_tokens=4)
    assert generated.startswith("ab")
    assert len(generated) == 6
