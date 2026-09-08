"""Backward-compatible Network import surface.

Canonical implementation lives in ``lisnn.network``. Existing experiments can
continue using ``import Network`` without modification.
"""

from lisnn.network import (
    SNN,
    NeuronPopulationSpec,
    TypeCounts,
    create_nn,
    create_snn,
    parse_population_spec,
)
from lisnn.neurons import NeuronType

__all__ = [
    "SNN",
    "NeuronType",
    "NeuronPopulationSpec",
    "TypeCounts",
    "create_nn",
    "create_snn",
    "parse_population_spec",
]
