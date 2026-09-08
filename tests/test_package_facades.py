"""Compatibility regression tests for canonical lisnn package ownership."""

from __future__ import annotations

import numpy as np

import Network
import NeuronModels as legacy_neurons
import debug as legacy_debug
import debugging as legacy_debugging
from lisnn.debugging import population_smoke_test, synapse_smoke_test
from lisnn.network import create_nn
from lisnn.neurons import kernels


def test_network_root_module_is_compatibility_facade() -> None:
    assert Network.create_nn is create_nn
    model = Network.create_nn(population=4, neuron_type="LIF")
    assert model.pool.shape == (4, kernels.NEURON_WIDTH)


def test_neuronmodels_root_module_is_compatibility_facade() -> None:
    assert legacy_neurons.lif_step is kernels.lif_step
    assert legacy_neurons.glif5_step is kernels.glif5_step
    assert legacy_neurons.NEURON_WIDTH == kernels.NEURON_WIDTH

    a = legacy_neurons.new_population(4, randomize_params=True, seed=1)
    b = kernels.new_population(4, randomize_params=True, seed=1)
    np.testing.assert_array_equal(a, b)


def test_debug_root_modules_are_compatibility_facades() -> None:
    assert legacy_debug.population_smoke_test is population_smoke_test
    assert legacy_debug.synapse_smoke_test is synapse_smoke_test
    assert legacy_debugging.population_smoke_test is population_smoke_test
    assert legacy_debugging.synapse_smoke_test is synapse_smoke_test


def test_synapse_smoke_is_callable_from_canonical_package() -> None:
    result = synapse_smoke_test(
        population=8,
        synapses_per_neuron=3,
        seed=1,
        verbose=False,
    )
    assert result["passed"] is True
