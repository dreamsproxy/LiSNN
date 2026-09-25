"""Canonical LiSNN debugging API."""

from lisnn.debugging.neuron_smoke import population_smoke_test
from lisnn.debugging.synapse_smoke import synapse_smoke_test
from lisnn.debugging.foundation_smoke import index_smoke_test, input_smoke_test, units_smoke_test
from lisnn.debugging.network_smoke import network_smoke_test, spatial_smoke_test
from lisnn.debugging.plasticity_observation_smoke import plasticity_observation_smoke_test
from lisnn.debugging.pair_smoke import pair_smoke_test
from lisnn.debugging.triplet_smoke import triplet_smoke_test
from lisnn.debugging.voltage_smoke import voltage_smoke_test
from lisnn.debugging.recurrent_smoke import recurrent_smoke_test
from lisnn.debugging.io_smoke import io_smoke_test
from lisnn.debugging.stream_runtime_smoke import stream_runtime_smoke_test
from lisnn.debugging.regime_smoke import regime_smoke_test
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
    "triplet_smoke_test",
    "voltage_smoke_test",
    "recurrent_smoke_test",
    "io_smoke_test",
    "stream_runtime_smoke_test",
    "regime_smoke_test",
    "run_all_smoke_tests",
]
