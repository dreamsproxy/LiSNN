"""Population specification parsing for LiSNN networks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import TypeAlias

import numpy as np

from lisnn.neurons.registry import NeuronType, normalize_neuron_type


NeuronPopulationSpec: TypeAlias = (
    str
    | NeuronType
    | Mapping[str | NeuronType, int | str | NeuronType]
)
TypeCounts: TypeAlias = OrderedDict[NeuronType, int]


def parse_population_spec(
    population: int,
    neuron_type: NeuronPopulationSpec,
) -> tuple[NeuronType, TypeCounts]:
    """Normalize homogeneous or mixed population specifications.

    Mixed mappings must contain ``"default"``. Explicit integer counts are
    assigned first; all unassigned neurons are given the default type.
    Groups are kept in insertion order so each type maps to one contiguous
    population slice.
    """

    if isinstance(neuron_type, (str, NeuronType)):
        canonical = normalize_neuron_type(neuron_type)
        return canonical, OrderedDict([(canonical, population)])

    if not isinstance(neuron_type, Mapping):
        raise TypeError("neuron_type must be a string, NeuronType, or mapping")

    if "default" not in neuron_type:
        raise ValueError("Mixed neuron populations must contain a 'default' entry")

    default_raw = neuron_type["default"]
    if not isinstance(default_raw, (str, NeuronType)):
        raise TypeError("neuron_type['default'] must be a neuron type")

    default_type = normalize_neuron_type(default_raw)
    counts: TypeCounts = OrderedDict()
    explicitly_assigned = 0

    for model_name, count in neuron_type.items():
        if model_name == "default":
            continue

        canonical = normalize_neuron_type(model_name)

        if not isinstance(count, (int, np.integer)):
            raise TypeError(
                f"Population count for {model_name!r} must be an integer"
            )

        count_int = int(count)
        if count_int < 0:
            raise ValueError(
                f"Population count for {model_name!r} cannot be negative"
            )

        explicitly_assigned += count_int
        counts[canonical] = counts.get(canonical, 0) + count_int

    if explicitly_assigned > population:
        raise ValueError(
            "Specified mixed neuron populations exceed total population: "
            f"{explicitly_assigned} > {population}"
        )

    remaining = population - explicitly_assigned
    if remaining:
        counts[default_type] = counts.get(default_type, 0) + remaining

    counts = OrderedDict(
        (model_name, count)
        for model_name, count in counts.items()
        if count > 0
    )

    return default_type, counts
