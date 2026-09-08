"""Regression tests for the stabilized LiSNN construction API."""

from __future__ import annotations

import numpy as np

import Network
import NeuronModels as nm
from lisnn.neurons import NeuronType


def test_homogeneous_population() -> None:
    model = Network.create_nn(
        population=8,
        neuron_type="GLIF5",
        randomize_params=False,
    )

    assert model.pool.shape == (8, nm.NEURON_WIDTH)
    assert model.pool.dtype == np.float32
    assert model.homogeneous is True
    assert model.type_counts == {NeuronType.GLIF5: 8}
    assert model.type_slices[NeuronType.GLIF5] == slice(0, 8)


def test_mixed_population_uses_default_for_remainder() -> None:
    model = Network.create_nn(
        population=8,
        neuron_type={
            "default": "LIF",
            "GLIF5": 2,
            "AdEx": 3,
        },
        randomize_params=True,
        seed=1,
    )

    assert model.homogeneous is False
    assert model.type_counts == {
        NeuronType.GLIF5: 2,
        NeuronType.ADEX: 3,
        NeuronType.LIF: 3,
    }
    assert model.type_slices[NeuronType.GLIF5] == slice(0, 2)
    assert model.type_slices[NeuronType.ADEX] == slice(2, 5)
    assert model.type_slices[NeuronType.LIF] == slice(5, 8)
    assert model.neuron_types.tolist() == [
        "glif5",
        "glif5",
        "adex",
        "adex",
        "adex",
        "lif",
        "lif",
        "lif",
    ]


def test_enum_population_spec_is_accepted() -> None:
    model = Network.create_nn(
        population=4,
        neuron_type=NeuronType.ADEX,
    )

    assert model.type_counts == {NeuronType.ADEX: 4}


def test_izhikevich_initial_recovery_state() -> None:
    model = Network.create_nn(
        population=4,
        neuron_type=NeuronType.IZHIKEVICH,
        randomize_params=True,
        seed=1,
    )

    expected_u = model.pool[:, nm.IZH_B] * model.pool[:, nm.V]
    np.testing.assert_allclose(model.pool[:, nm.ADAPT], expected_u)


def test_population_seed_is_deterministic() -> None:
    model_a = Network.create_nn(
        population=8,
        neuron_type="LIF",
        randomize_params=True,
        seed=1,
    )
    model_b = Network.create_nn(
        population=8,
        neuron_type="LIF",
        randomize_params=True,
        seed=1,
    )

    np.testing.assert_array_equal(model_a.pool, model_b.pool)
