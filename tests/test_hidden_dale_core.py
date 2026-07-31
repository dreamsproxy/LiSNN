from pathlib import Path

import numpy as np

from core import LanguageTrajectoryModel, NetworkConfig, TextCorpus


def hidden_config(hidden_neurons: int = 12) -> NetworkConfig:
    return NetworkConfig(
        context_length=2,
        hidden_neurons=hidden_neurons,
        ticks_per_token=1,
        prediction_ticks=2,
        signal_scale=500.0,
        readout_alpha=0.2,
        anti_hebbian_alpha=0.05,
        prune_interval=1000,
        seed=4,
    )


def test_hidden_neurons_have_no_direct_io_path() -> None:
    model = LanguageTrajectoryModel(
        TextCorpus.from_text("abcabc"),
        hidden_config(),
    )
    network = model.network

    assert network.hidden_slice.stop - network.hidden_slice.start == 12
    assert np.all(
        network._context_input(0, 0)[network.hidden_slice] == 0.0
    )
    assert np.all(network._query_input()[network.hidden_slice] == 0.0)
    assert network.readout_weights.shape == (
        network.vocabulary_size,
        network.io_neuron_count,
    )
    assert network.readout_weights.shape[1] < network.num_neurons

    # Dense recurrence couples hidden and I/O neurons in both directions.
    assert np.all(
        np.abs(
            network.weights.weights[
                network.hidden_slice,
                : network.io_neuron_count,
            ]
        )
        > 0.0
    )
    assert np.all(
        np.abs(
            network.weights.weights[
                : network.io_neuron_count,
                network.hidden_slice,
            ]
        )
        > 0.0
    )


def test_hidden_neurons_are_more_excitable() -> None:
    model = LanguageTrajectoryModel(
        TextCorpus.from_text("abcabc"),
        hidden_config(hidden_neurons=32),
    )
    network = model.network

    assert float(np.mean(network.thresholds[network.hidden_slice])) < float(
        np.mean(network.thresholds[: network.io_neuron_count])
    )
    assert float(np.mean(network.pre_tau[network.hidden_slice])) < float(
        np.mean(network.pre_tau[: network.io_neuron_count])
    )


def test_dale_polarity_is_immutable_during_training() -> None:
    model = LanguageTrajectoryModel(
        TextCorpus.from_text("abcabcabc"),
        hidden_config(),
    )
    network = model.network
    original_types = network.weights.neuron_types.copy()

    expected_inhibitory = round(network.num_neurons * 0.8)
    assert network.weights.inhibitory_count == expected_inhibitory

    model.fit(epochs=2)

    np.testing.assert_array_equal(
        network.weights.neuron_types,
        original_types,
    )
    excitatory = original_types > 0
    inhibitory = original_types < 0
    assert np.all(network.weights.weights[:, excitatory] >= 0.0)
    assert np.all(network.weights.weights[:, inhibitory] <= 0.0)


def test_checkpoint_preserves_hidden_and_neuron_types(tmp_path: Path) -> None:
    model = LanguageTrajectoryModel(
        TextCorpus.from_text("abcabcabc"),
        hidden_config(),
    )
    model.fit(epochs=1)
    checkpoint = tmp_path / "hidden.npz"
    model.save(checkpoint)

    restored = LanguageTrajectoryModel.load(checkpoint)

    assert restored.network.hidden_neurons == model.network.hidden_neurons
    np.testing.assert_array_equal(
        restored.network.weights.neuron_types,
        model.network.weights.neuron_types,
    )
    np.testing.assert_allclose(
        restored.predict_text("ab").probabilities,
        model.predict_text("ab").probabilities,
    )
