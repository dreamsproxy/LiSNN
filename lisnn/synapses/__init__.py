"""Canonical LiSNN synapse substrate API."""

from lisnn.synapses.core import (
    SynapseEdges,
    topology_validation,
    SynapseIndexArray,
    SynapseWeightArray,
    WeightSpec,
    create_fixed_out_degree,
    create_synapses,
)

from lisnn.synapses.propagation import propagate, PropagationResult

__all__ = [
    "propagate",
    "PropagationResult",

    "topology_validation",
    "SynapseEdges",
    "SynapseIndexArray",
    "SynapseWeightArray",
    "WeightSpec",
    "create_synapses",
    "create_fixed_out_degree",
]

