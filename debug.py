"""Backward-compatible debugging import surface.

Canonical smoke/debug implementations live in ``lisnn.debugging``.
"""

from lisnn.debugging import population_smoke_test, synapse_smoke_test

__all__ = [
    "population_smoke_test",
    "synapse_smoke_test",
]
