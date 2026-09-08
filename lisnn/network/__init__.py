"""Network construction API."""

from lisnn.network.core import SNN, create_nn, create_snn
from lisnn.network.spec import NeuronPopulationSpec, TypeCounts, parse_population_spec

__all__ = [
    "SNN",
    "create_nn",
    "create_snn",
    "NeuronPopulationSpec",
    "TypeCounts",
    "parse_population_spec",
]
