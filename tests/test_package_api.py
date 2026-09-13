"""Canonical APIs work without the removed root compatibility modules."""

import lisnn
from lisnn import debugging, network, neurons, synapses
from lisnn.neurons import kernels, models


def test_public_constructors_are_canonical():
    assert lisnn.create_nn is network.create_nn
    assert lisnn.new_population is neurons.new_population
    assert lisnn.create_synapses is synapses.create_synapses
    assert models.lif_step is kernels.lif_step


def test_public_smoke_api_is_callable():
    assert debugging.index_smoke_test(verbose=False)["passed"]
    assert debugging.units_smoke_test(verbose=False)["passed"]
