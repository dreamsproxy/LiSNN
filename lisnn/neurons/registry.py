"""Neuron type identifiers and vectorized step-function registry."""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import NeuronModels as nm

from lisnn.types import NeuronStep


class NeuronType(str, Enum):
    """Canonical serialized identifiers for supported LiSNN neuron models."""

    LIF = "lif"
    ADAPTIVE_LIF = "adaptive_lif"
    IZHIKEVICH = "izhikevich"
    ADEX = "adex"
    GLIF3 = "glif3"
    GLIF4 = "glif4"
    GLIF5 = "glif5"
    CADEX = "cadex"
    CADEX_GLIF = "cadex_glif"


_NEURON_TYPE_ALIASES: Final[dict[str, NeuronType]] = {
    "lif": NeuronType.LIF,
    "adaptivelif": NeuronType.ADAPTIVE_LIF,
    "adaptive_lif": NeuronType.ADAPTIVE_LIF,
    "alif": NeuronType.ADAPTIVE_LIF,
    "izhikevich": NeuronType.IZHIKEVICH,
    "adex": NeuronType.ADEX,
    "glif3": NeuronType.GLIF3,
    "glif4": NeuronType.GLIF4,
    "glif5": NeuronType.GLIF5,
    "cadex": NeuronType.CADEX,
    "cadex_glif": NeuronType.CADEX_GLIF,
    "cadec_glif": NeuronType.CADEX_GLIF,
}

NEURON_TYPE_ALIASES = MappingProxyType(_NEURON_TYPE_ALIASES)


NEURON_STEP_REGISTRY: Final[dict[NeuronType, NeuronStep]] = {
    NeuronType.LIF: nm.lif_step,
    NeuronType.ADAPTIVE_LIF: nm.adaptive_lif_step,
    NeuronType.IZHIKEVICH: nm.izhikevich_step,
    NeuronType.ADEX: nm.adex_step,
    NeuronType.GLIF3: nm.glif3_step,
    NeuronType.GLIF4: nm.glif4_step,
    NeuronType.GLIF5: nm.glif5_step,
    NeuronType.CADEX: nm.cadex_step,
    NeuronType.CADEX_GLIF: nm.cadex_glif_step,
}


def normalize_neuron_type(neuron_type: str | NeuronType) -> NeuronType:
    """Return the canonical enum value for a user-facing neuron type."""

    if isinstance(neuron_type, NeuronType):
        return neuron_type

    if not isinstance(neuron_type, str):
        raise TypeError("neuron type must be a string or NeuronType")

    key = (
        neuron_type.strip().lower().replace("-", "_").replace(" ", "")
    )

    try:
        return _NEURON_TYPE_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(model.value for model in NeuronType)
        raise ValueError(
            f"Unsupported neuron type: {neuron_type!r}. Supported: {supported}"
        ) from exc


def get_step_function(neuron_type: str | NeuronType) -> NeuronStep:
    """Resolve a neuron type directly to its vectorized step function."""

    return NEURON_STEP_REGISTRY[normalize_neuron_type(neuron_type)]
