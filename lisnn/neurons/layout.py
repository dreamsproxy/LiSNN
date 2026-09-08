"""Shared neuron-matrix ABI.

These indices define the stable column layout of every LiSNN neuron population.
The numerical implementation lives in ``lisnn.neurons.kernels``; this module
provides the canonical import location for layout consumers such as networks,
plasticity, serialization, visualization, and debugging.
"""

from lisnn.neurons.kernels import (
    ADAPT,
    ADEX,
    ALIF,
    ASC_1,
    ASC_2,
    CADEX,
    CORE,
    EXP,
    GLIF,
    IZH,
    NEURON_WIDTH,
    REFRACTORY,
    SLICES,
    STATE,
    THETA_S,
    THETA_V,
    V,
    COLUMN_NAMES,
)

__all__ = [
    "V",
    "ADAPT",
    "THETA_S",
    "ASC_1",
    "ASC_2",
    "THETA_V",
    "REFRACTORY",
    "STATE",
    "CORE",
    "ALIF",
    "IZH",
    "EXP",
    "ADEX",
    "GLIF",
    "CADEX",
    "NEURON_WIDTH",
    "COLUMN_NAMES",
    "SLICES",
]
