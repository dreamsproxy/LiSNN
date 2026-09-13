"""Network construction API."""

from lisnn.network.core import SNN, create_nn, create_snn
from lisnn.network.spec import NeuronPopulationSpec, TypeCounts, parse_population_spec
from lisnn.spatial import SpatialConfig, SpatialSpec, SpatialVolume

from lisnn.network.runtime import FixedWeightRuntime, TickResult

__all__ = [
    "FixedWeightRuntime",
    "TickResult",

    "SNN",
    "create_nn",
    "create_snn",
    "NeuronPopulationSpec",
    "TypeCounts",
    "parse_population_spec",
    "SpatialConfig",
    "SpatialSpec",
    "SpatialVolume",
]

