"""Public neuron API for LiSNN.

The current numerical kernels remain in the compatibility module
``NeuronModels`` while the stable typed registry and package-level imports are
introduced here. This keeps existing experiments working during the cleanup.
"""

from NeuronModels import (
    ADAPT,
    ASC_1,
    ASC_2,
    NEURON_WIDTH,
    REFRACTORY,
    THETA_S,
    THETA_V,
    V,
    adaptive_lif_step,
    adex_step,
    cadex_glif_step,
    cadex_step,
    glif3_step,
    glif4_step,
    glif5_step,
    initialize_population_parameters,
    izhikevich_step,
    lif_step,
    new_population,
)
from lisnn.neurons.registry import (
    NEURON_STEP_REGISTRY,
    NEURON_TYPE_ALIASES,
    NeuronType,
    get_step_function,
    normalize_neuron_type,
)

__all__ = [
    "NeuronType",
    "NEURON_TYPE_ALIASES",
    "NEURON_STEP_REGISTRY",
    "normalize_neuron_type",
    "get_step_function",
    "new_population",
    "initialize_population_parameters",
    "lif_step",
    "adaptive_lif_step",
    "izhikevich_step",
    "adex_step",
    "glif3_step",
    "glif4_step",
    "glif5_step",
    "cadex_step",
    "cadex_glif_step",
    "V",
    "ADAPT",
    "THETA_S",
    "ASC_1",
    "ASC_2",
    "THETA_V",
    "REFRACTORY",
    "NEURON_WIDTH",
]
