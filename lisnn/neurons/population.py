"""Typed neuron-population construction API."""

from __future__ import annotations

import numpy as np

from lisnn.neurons import kernels as nm
from lisnn.types import NeuronPopulation, Seed


def initialize_population_parameters(
    neurons: NeuronPopulation,
    randomize_params: bool = False,
    rng: np.random.Generator | None = None,
) -> NeuronPopulation:
    """Initialize all supported neuron parameter families in-place."""

    return nm.initialize_population_parameters(
        neurons,
        randomize_params=randomize_params,
        rng=rng,
    )


def new_population(
    n_neurons: int,
    fill: float | np.float32 = np.float32(0.0),
    randomize_params: bool = False,
    seed: Seed = None,
) -> NeuronPopulation:
    """Create the shared ``(N, NEURON_WIDTH)`` float32 neuron matrix."""

    return nm.new_population(
        n_neurons,
        fill=fill,
        randomize_params=randomize_params,
        seed=seed,
    )


__all__ = [
    "initialize_population_parameters",
    "new_population",
]
