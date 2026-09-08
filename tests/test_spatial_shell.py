"""Regression tests for the optional LiSNN spatial substrate."""

from __future__ import annotations

import numpy as np
import pytest

import Network
from lisnn.spatial import CompartmentType, MorphologyTable, SpatialVolume


def test_non_spatial_network_remains_supported() -> None:
    model = Network.create_nn(
        population=8,
        neuron_type="LIF",
        seed=1,
    )

    assert model.space is None


def test_uniform_spatial_volume() -> None:
    model = Network.create_nn(
        population=8,
        neuron_type="GLIF5",
        spatial={
            "size": (100.0, 200.0, 300.0),
            "origin": (-50.0, 10.0, 25.0),
            "units": "um",
            "boundary": "closed",
            "placement": "uniform",
        },
        seed=1,
    )

    assert isinstance(model.space, SpatialVolume)
    assert model.space.positions.shape == (8, 3)
    assert model.space.positions.dtype == np.float32
    assert model.space.units == "um"
    assert model.space.boundary == "closed"
    assert model.space.placement == "uniform"

    lower = model.space.origin
    upper = model.space.origin + model.space.size

    assert np.all(model.space.positions >= lower[None, :])
    assert np.all(model.space.positions <= upper[None, :])


def test_spatial_seed_is_deterministic_and_rng_independent() -> None:
    spatial = {
        "size": (1000.0, 1000.0, 1000.0),
        "placement": "uniform",
    }

    model_a = Network.create_nn(
        population=16,
        neuron_type="AdEx",
        randomize_params=True,
        spatial=spatial,
        seed=42,
    )
    model_b = Network.create_nn(
        population=16,
        neuron_type="AdEx",
        randomize_params=True,
        spatial=spatial,
        seed=42,
    )
    model_without_space = Network.create_nn(
        population=16,
        neuron_type="AdEx",
        randomize_params=True,
        seed=42,
    )

    assert model_a.space is not None
    assert model_b.space is not None

    np.testing.assert_array_equal(
        model_a.space.positions,
        model_b.space.positions,
    )

    # Creating spatial coordinates must not consume or perturb the neuron
    # parameter RNG trajectory.
    np.testing.assert_array_equal(
        model_a.pool,
        model_without_space.pool,
    )


def test_explicit_spatial_positions() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0],
            [50.0, 50.0, 50.0],
        ],
        dtype=np.float32,
    )

    model = Network.create_nn(
        population=3,
        spatial={
            "size": (100.0, 100.0, 100.0),
            "placement": "explicit",
            "positions": positions,
        },
    )

    assert model.space is not None
    np.testing.assert_array_equal(model.space.positions, positions)


def test_explicit_positions_must_fit_volume() -> None:
    with pytest.raises(ValueError, match="inside the spatial volume"):
        Network.create_nn(
            population=2,
            spatial={
                "size": (10.0, 10.0, 10.0),
                "placement": "explicit",
                "positions": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [11.0, 5.0, 5.0],
                    ],
                    dtype=np.float32,
                ),
            },
        )


def test_morphology_table_is_inactive_empty_shell() -> None:
    morphology = MorphologyTable.empty()

    assert morphology.compartment_count == 0
    assert morphology.position.shape == (0, 3)
    assert CompartmentType.NODE_OF_RANVIER.value == 4
