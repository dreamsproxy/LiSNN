"""Backward-compatible debugging import surface.

Preferred usage:

    import debugging as debug
    debug.population_smoke_test(...)
"""

from debug import population_smoke_test

__all__ = ["population_smoke_test"]
