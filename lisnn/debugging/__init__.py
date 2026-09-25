"""Canonical LiSNN debugging API."""

from lisnn.debugging.neuron_smoke import population_smoke_test
from lisnn.debugging.synapse_smoke import synapse_smoke_test
from lisnn.debugging.foundation_smoke import index_smoke_test, input_smoke_test, units_smoke_test
from lisnn.debugging.network_smoke import network_smoke_test, spatial_smoke_test
from lisnn.debugging.plasticity_observation_smoke import plasticity_observation_smoke_test
from lisnn.debugging.pair_smoke import pair_smoke_test
from lisnn.debugging.runner import run_all_smoke_tests

__all__ = [
    "population_smoke_test",
    "synapse_smoke_test",
    "index_smoke_test",
    "input_smoke_test",
    "units_smoke_test",
    "network_smoke_test",
    "spatial_smoke_test",
    "plasticity_observation_smoke_test",
    "pair_smoke_test",
    "run_all_smoke_tests",
]
