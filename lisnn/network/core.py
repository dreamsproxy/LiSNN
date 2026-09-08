"""Core LiSNN network container.

This module currently owns neuron-population construction and optional spatial
placement. Synaptic weights, connectivity, propagation, plasticity, and network
stepping belong to later network modules.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray

from lisnn.neurons import kernels as nm
from lisnn.neurons.registry import NeuronType
from lisnn.network.spec import NeuronPopulationSpec, TypeCounts, parse_population_spec
from lisnn.spatial import SpatialSpec, SpatialVolume, create_spatial_volume
from lisnn.types import NeuronPopulation, Seed


class SNN:
    """Initialized LiSNN neuron population and its construction metadata."""

    population_size: int
    randomize_params: bool
    seed: Seed
    default_neuron_type: NeuronType
    type_counts: TypeCounts
    homogeneous: bool
    pool: NeuronPopulation
    neurons: NeuronPopulation
    type_slices: OrderedDict[NeuronType, slice]
    neuron_types: NDArray[np.str_]
    space: SpatialVolume | None

    def __init__(
        self,
        population: int = 8,
        neuron_type: NeuronPopulationSpec = NeuronType.LIF,
        fill: float | np.float32 = np.float32(0.0),
        randomize_params: bool = False,
        spatial: SpatialSpec = None,
        seed: Seed = None,
    ) -> None:
        """Construct neuron state, contiguous model groups, and optional space.

        PSEUDOCODE:
            validate population size
            normalize homogeneous/mixed population specification
            create shared float32 neuron matrix
            assign one contiguous slice to every neuron type
            initialize model-specific dynamic state
            optionally create a geometry-only SpatialVolume

        ``spatial=None`` preserves the non-spatial behavior of earlier LiSNN
        construction APIs.
        """

        if not isinstance(population, (int, np.integer)):
            raise TypeError("population must be an integer")

        population = int(population)
        if population <= 0:
            raise ValueError("population must be greater than zero")

        self.population_size = population
        self.randomize_params = bool(randomize_params)
        self.seed = seed

        self.default_neuron_type, self.type_counts = parse_population_spec(
            population,
            neuron_type,
        )
        self.homogeneous = len(self.type_counts) == 1

        self.pool = nm.new_population(
            population,
            fill=fill,
            randomize_params=randomize_params,
            seed=seed,
        )
        self.neurons = self.pool

        self.type_slices = OrderedDict()
        self.neuron_types = np.empty(population, dtype="<U16")

        cursor = 0
        for model_type, count in self.type_counts.items():
            start = cursor
            stop = cursor + count
            model_slice = slice(start, stop)

            self.type_slices[model_type] = model_slice
            self.neuron_types[model_slice] = model_type.value
            cursor = stop

        if cursor != population:
            raise RuntimeError(
                "Internal population construction error: "
                f"assigned {cursor} neurons for population {population}"
            )

        self._initialize_model_states()

        # Spatial geometry is deliberately independent of neuron dynamics.
        # Reusing the public seed creates deterministic placement through an
        # independent RNG without consuming neuron-parameter RNG state.
        self.space = create_spatial_volume(
            population,
            spatial,
            seed=seed,
        )

    def _initialize_model_states(self) -> None:
        """Initialize dynamic states whose neutral value is model-specific."""

        izh_slice = self.type_slices.get(NeuronType.IZHIKEVICH)
        if izh_slice is None:
            return

        # Izhikevich recovery state u begins at b*V. The shared ADAPT column
        # stores u for Izhikevich neurons.
        self.pool[izh_slice, nm.ADAPT] = (
            self.pool[izh_slice, nm.IZH_B]
            * self.pool[izh_slice, nm.V]
        ).astype(np.float32)

    def __repr__(self) -> str:
        population_description = ", ".join(
            f"{model_type.value}={count}"
            for model_type, count in self.type_counts.items()
        )
        spatial_description = "none" if self.space is None else repr(self.space)
        return (
            "SNN("
            f"population={self.population_size}, "
            f"neurons=[{population_description}], "
            f"randomize_params={self.randomize_params}, "
            f"space={spatial_description}"
            ")"
        )


def create_nn(
    population: int = 8,
    neuron_type: NeuronPopulationSpec = NeuronType.LIF,
    fill: float | np.float32 = np.float32(0.0),
    randomize_params: bool = False,
    spatial: SpatialSpec = None,
    seed: Seed = None,
) -> SNN:
    """Create an initialized SNN without synaptic connectivity or weights."""

    return SNN(
        population=population,
        neuron_type=neuron_type,
        fill=fill,
        randomize_params=randomize_params,
        spatial=spatial,
        seed=seed,
    )


create_snn = create_nn
