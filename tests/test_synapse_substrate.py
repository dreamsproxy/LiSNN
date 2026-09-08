"""Regression tests for the M2 sparse synaptic edge substrate."""

from __future__ import annotations

import numpy as np
import pytest

import Network
from lisnn.synapses import SynapseEdges, create_fixed_out_degree, create_synapses


def test_explicit_synapses_use_canonical_dtypes() -> None:
    edges = create_synapses(
        population=4,
        pre_idx=[0, 0, 2],
        post_idx=[1, 3, 1],
        weights=[0.25, 0.5, 0.75],
    )

    assert isinstance(edges, SynapseEdges)
    assert edges.population_size == 4
    assert edges.edge_count == 3
    assert len(edges) == 3
    assert edges.pre_idx.dtype == np.int32
    assert edges.post_idx.dtype == np.int32
    assert edges.weight.dtype == np.float32
    np.testing.assert_array_equal(edges.pre_idx, [0, 0, 2])
    np.testing.assert_array_equal(edges.post_idx, [1, 3, 1])
    np.testing.assert_allclose(edges.weight, [0.25, 0.5, 0.75])


def test_scalar_weight_broadcasts_per_edge() -> None:
    edges = create_synapses(
        population=3,
        pre_idx=[0, 1],
        post_idx=[1, 2],
        weights=np.float32(0.4),
    )

    np.testing.assert_array_equal(
        edges.weight,
        np.array([0.4, 0.4], dtype=np.float32),
    )


def test_invalid_explicit_edges_are_rejected() -> None:
    with pytest.raises(ValueError, match="identical shape"):
        create_synapses(
            population=4,
            pre_idx=[0, 1],
            post_idx=[1],
        )

    with pytest.raises(IndexError, match="outside the population"):
        create_synapses(
            population=4,
            pre_idx=[0, 4],
            post_idx=[1, 2],
        )

    with pytest.raises(ValueError, match="autapses are disabled"):
        create_synapses(
            population=4,
            pre_idx=[0],
            post_idx=[0],
        )

    with pytest.raises(ValueError, match="duplicate ordered synaptic edges"):
        create_synapses(
            population=4,
            pre_idx=[0, 0],
            post_idx=[1, 1],
        )

    with pytest.raises(ValueError, match="non-negative"):
        create_synapses(
            population=4,
            pre_idx=[0],
            post_idx=[1],
            weights=[-0.1],
        )


def test_autapses_and_duplicate_contacts_are_explicit_opt_ins() -> None:
    edges = create_synapses(
        population=2,
        pre_idx=[0, 0],
        post_idx=[0, 0],
        weights=[0.3, 0.4],
        allow_self=True,
        allow_duplicates=True,
    )

    assert edges.edge_count == 2
    np.testing.assert_array_equal(edges.pre_idx, [0, 0])
    np.testing.assert_array_equal(edges.post_idx, [0, 0])


def test_fixed_out_degree_is_seed_deterministic() -> None:
    edges_a = create_fixed_out_degree(
        population=16,
        synapses_per_neuron=4,
        weight_range=(0.1, 0.8),
        seed=7,
    )
    edges_b = create_fixed_out_degree(
        population=16,
        synapses_per_neuron=4,
        weight_range=(0.1, 0.8),
        seed=7,
    )

    np.testing.assert_array_equal(edges_a.pre_idx, edges_b.pre_idx)
    np.testing.assert_array_equal(edges_a.post_idx, edges_b.post_idx)
    np.testing.assert_array_equal(edges_a.weight, edges_b.weight)

    assert edges_a.edge_count == 16 * 4
    assert np.all(edges_a.pre_idx != edges_a.post_idx)
    assert np.all(edges_a.weight >= np.float32(0.1))
    assert np.all(edges_a.weight < np.float32(0.8))

    for neuron in range(16):
        mask = edges_a.pre_idx == neuron
        targets = edges_a.post_idx[mask]
        assert targets.size == 4
        assert np.unique(targets).size == 4


def test_fixed_out_degree_supports_zero_and_maximum_degree() -> None:
    empty = create_fixed_out_degree(
        population=4,
        synapses_per_neuron=0,
        seed=1,
    )
    assert empty.edge_count == 0
    assert empty.pre_idx.dtype == np.int32
    assert empty.weight.dtype == np.float32

    complete = create_fixed_out_degree(
        population=4,
        synapses_per_neuron=3,
        weight_range=(0.5, 0.5),
        seed=1,
    )
    assert complete.edge_count == 12
    assert np.all(complete.pre_idx != complete.post_idx)
    np.testing.assert_array_equal(
        complete.weight,
        np.full(12, 0.5, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="maximum is 3"):
        create_fixed_out_degree(
            population=4,
            synapses_per_neuron=4,
        )


def test_synapse_substrate_is_neuron_model_independent() -> None:
    model = Network.create_nn(
        population=8,
        neuron_type={
            "default": "LIF",
            "GLIF5": 2,
            "AdEx": 3,
        },
        randomize_params=True,
        seed=11,
    )

    edges = create_fixed_out_degree(
        population=model.population_size,
        synapses_per_neuron=3,
        seed=11,
    )

    assert edges.population_size == model.population_size
    assert edges.edge_count == model.population_size * 3
    assert edges.pre_idx.max() < model.population_size
    assert edges.post_idx.max() < model.population_size
