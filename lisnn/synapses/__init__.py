"""Canonical LiSNN synapse substrate API."""

from lisnn.synapses.core import (
    SynapseEdges,
    SynapseIndexArray,
    SynapseWeightArray,
    WeightSpec,
    create_fixed_out_degree,
    create_synapses,
)

__all__ = [
    "SynapseEdges",
    "SynapseIndexArray",
    "SynapseWeightArray",
    "WeightSpec",
    "create_synapses",
    "create_fixed_out_degree",
]
