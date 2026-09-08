"""Sparse synaptic edge-state substrate for LiSNN.

The canonical synapse representation is structure-of-arrays rather than a dense
``N x N`` matrix. Each array has one entry per directed synaptic edge.

M2.1 intentionally stores only topology and long-term efficacy. Propagation,
delays, receptor kinetics, short-term plasticity, and long-term plasticity are
separate later modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


SynapseIndexArray: TypeAlias = NDArray[np.int32]
SynapseWeightArray: TypeAlias = NDArray[np.float32]
WeightSpec: TypeAlias = float | np.float32 | ArrayLike


def _validate_population(population: int) -> int:
    if not isinstance(population, (int, np.integer)):
        raise TypeError("population must be an integer")

    population = int(population)
    if population <= 0:
        raise ValueError("population must be greater than zero")
    if population > np.iinfo(np.int32).max:
        raise ValueError("population exceeds int32 synapse-index capacity")

    return population


def _as_index_vector(name: str, values: ArrayLike) -> SynapseIndexArray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if raw.size == 0:
        return np.empty(0, dtype=np.int32)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must contain integer neuron indices")

    return raw.astype(np.int32, copy=False)


def _as_weight_vector(weights: WeightSpec, edge_count: int) -> SynapseWeightArray:
    if np.isscalar(weights):
        result = np.full(edge_count, weights, dtype=np.float32)
    else:
        raw = np.asarray(weights)
        if raw.ndim != 1:
            raise ValueError("weights must be a scalar or 1D array")
        if raw.shape[0] != edge_count:
            raise ValueError(
                "weights must have one value per edge: "
                f"expected {edge_count}, got {raw.shape[0]}"
            )
        result = raw.astype(np.float32, copy=False)

    if not np.all(np.isfinite(result)):
        raise ValueError("weights must contain only finite values")
    if np.any(result < np.float32(0.0)):
        raise ValueError(
            "synaptic efficacy weights are unsigned and must be non-negative; "
            "excitatory/inhibitory polarity belongs to neuron/receptor identity"
        )

    return result


@dataclass(slots=True)
class SynapseEdges:
    """Directed sparse synapses stored as one vector entry per edge.

    ``weight`` is an unsigned long-term efficacy magnitude. This deliberately
    keeps Dale/receptor polarity separate from efficacy so later conductance
    models can determine sign from presynaptic/receptor identity.

    By default LiSNN permits at most one ordered edge ``pre -> post`` and rejects
    autapses. Both constraints can be relaxed explicitly for sub-experiments.
    """

    population_size: int
    pre_idx: SynapseIndexArray
    post_idx: SynapseIndexArray
    weight: SynapseWeightArray
    allow_self: bool = False
    allow_duplicates: bool = False

    def __post_init__(self) -> None:
        self.population_size = _validate_population(self.population_size)
        self.pre_idx = _as_index_vector("pre_idx", self.pre_idx)
        self.post_idx = _as_index_vector("post_idx", self.post_idx)

        if self.pre_idx.shape != self.post_idx.shape:
            raise ValueError("pre_idx and post_idx must have identical shape")

        self.weight = _as_weight_vector(self.weight, self.pre_idx.size)
        self.allow_self = bool(self.allow_self)
        self.allow_duplicates = bool(self.allow_duplicates)

        if self.pre_idx.size == 0:
            return

        if np.any(self.pre_idx < 0) or np.any(self.pre_idx >= self.population_size):
            raise IndexError("pre_idx contains neuron indices outside the population")
        if np.any(self.post_idx < 0) or np.any(self.post_idx >= self.population_size):
            raise IndexError("post_idx contains neuron indices outside the population")

        if not self.allow_self and np.any(self.pre_idx == self.post_idx):
            raise ValueError("autapses are disabled; set allow_self=True to permit them")

        if not self.allow_duplicates:
            # int64 key prevents multiplication overflow for the supported
            # int32 population/index range.
            edge_keys = (
                self.pre_idx.astype(np.int64) * np.int64(self.population_size)
                + self.post_idx.astype(np.int64)
            )
            if np.unique(edge_keys).size != edge_keys.size:
                raise ValueError(
                    "duplicate ordered synaptic edges are disabled; "
                    "set allow_duplicates=True to permit multiple contacts"
                )

    @property
    def edge_count(self) -> int:
        """Number of directed synaptic edges."""

        return int(self.pre_idx.size)

    def __len__(self) -> int:
        return self.edge_count


def create_synapses(
    population: int,
    pre_idx: ArrayLike,
    post_idx: ArrayLike,
    weights: WeightSpec = np.float32(1.0),
    *,
    allow_self: bool = False,
    allow_duplicates: bool = False,
) -> SynapseEdges:
    """Create a validated sparse synapse set from explicit directed edges."""

    population = _validate_population(population)
    pre = _as_index_vector("pre_idx", pre_idx)
    post = _as_index_vector("post_idx", post_idx)

    if pre.shape != post.shape:
        raise ValueError("pre_idx and post_idx must have identical shape")

    weight = _as_weight_vector(weights, pre.size)
    return SynapseEdges(
        population_size=population,
        pre_idx=pre,
        post_idx=post,
        weight=weight,
        allow_self=allow_self,
        allow_duplicates=allow_duplicates,
    )


def create_fixed_out_degree(
    population: int,
    synapses_per_neuron: int,
    *,
    weight_range: tuple[float, float] = (0.05, 1.0),
    allow_self: bool = False,
    seed: int | None = None,
) -> SynapseEdges:
    """Create random directed topology with exact fan-out per neuron.

    Degree, rather than a global connection probability, is the scaling
    primitive. For every presynaptic neuron exactly ``synapses_per_neuron``
    unique postsynaptic targets are selected without replacement.
    """

    population = _validate_population(population)

    if not isinstance(synapses_per_neuron, (int, np.integer)):
        raise TypeError("synapses_per_neuron must be an integer")
    synapses_per_neuron = int(synapses_per_neuron)
    if synapses_per_neuron < 0:
        raise ValueError("synapses_per_neuron must be non-negative")

    max_degree = population if allow_self else population - 1
    if synapses_per_neuron > max_degree:
        raise ValueError(
            "synapses_per_neuron exceeds the number of unique permitted targets: "
            f"maximum is {max_degree}"
        )

    low, high = (float(weight_range[0]), float(weight_range[1]))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("weight_range values must be finite")
    if low < 0.0:
        raise ValueError("weight_range lower bound must be non-negative")
    if high < low:
        raise ValueError("weight_range upper bound must be >= lower bound")

    edge_count = population * synapses_per_neuron
    if edge_count == 0:
        return SynapseEdges(
            population_size=population,
            pre_idx=np.empty(0, dtype=np.int32),
            post_idx=np.empty(0, dtype=np.int32),
            weight=np.empty(0, dtype=np.float32),
            allow_self=allow_self,
        )

    rng = np.random.default_rng(seed)
    pre = np.repeat(
        np.arange(population, dtype=np.int32),
        synapses_per_neuron,
    )
    post = np.empty(edge_count, dtype=np.int32)

    cursor = 0
    for presynaptic in range(population):
        if allow_self:
            selected = rng.choice(
                population,
                size=synapses_per_neuron,
                replace=False,
            ).astype(np.int32, copy=False)
        else:
            # Sample from [0, population-2] and remap values at/above the
            # presynaptic index upward by one. This excludes self without
            # allocating an N-sized candidate vector for every neuron.
            selected = rng.choice(
                population - 1,
                size=synapses_per_neuron,
                replace=False,
            ).astype(np.int32, copy=False)
            selected += (selected >= presynaptic).astype(np.int32)

        selected.sort()

        next_cursor = cursor + synapses_per_neuron
        post[cursor:next_cursor] = selected
        cursor = next_cursor

    if high == low:
        weight = np.full(edge_count, low, dtype=np.float32)
    else:
        weight = rng.uniform(low, high, size=edge_count).astype(np.float32)

    return SynapseEdges(
        population_size=population,
        pre_idx=pre,
        post_idx=post,
        weight=weight,
        allow_self=allow_self,
        allow_duplicates=False,
    )


__all__ = [
    "SynapseEdges",
    "SynapseIndexArray",
    "SynapseWeightArray",
    "WeightSpec",
    "create_synapses",
    "create_fixed_out_degree",
]
