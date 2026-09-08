"""Backward-compatible debugging import surface.

Preferred canonical usage is ``from lisnn import debugging`` or
``from lisnn.debugging import ...``. Existing ``import debugging as debug``
continues to work.
"""

from lisnn.debugging import population_smoke_test, synapse_smoke_test

__all__ = [
    "population_smoke_test",
    "synapse_smoke_test",
]
