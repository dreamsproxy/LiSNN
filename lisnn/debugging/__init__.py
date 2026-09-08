"""Canonical LiSNN debugging API."""

from lisnn.debugging.neuron_smoke import population_smoke_test
from lisnn.debugging.synapse_smoke import synapse_smoke_test

__all__ = [
    "population_smoke_test",
    "synapse_smoke_test",
]
