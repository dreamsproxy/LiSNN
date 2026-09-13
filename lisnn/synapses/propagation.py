"""Sparse fixed integrated-current propagation, independent of neuron models."""

from dataclasses import dataclass

import numpy as np

from lisnn.synapses.core import SynapseEdges
from lisnn.validation import scalar32, vector32


@dataclass(frozen=True)
class PropagationResult:
    active_edges: np.ndarray
    edge_impulse_pA_ms: np.ndarray
    edge_current_pA: np.ndarray
    synaptic_current_pA: np.ndarray


def binary_spikes(values, population):
    raw = np.asarray(values)
    if raw.shape != (population,) or raw.dtype.kind not in 'iuf':
        raise ValueError(f'spikes must be a real binary vector of shape ({population},)')
    # Check before float32 conversion so near-one analog values cannot round to 1.
    if not np.all((raw == 0) | (raw == 1)):
        raise ValueError('spikes must contain only binary events 0 or 1')
    return vector32('spikes', raw, population, scalar=False)


def validated_edges(edges):
    if not isinstance(edges, SynapseEdges):
        raise TypeError('edges must be SynapseEdges')
    # Revalidate because the existing edge container exposes mutable arrays.
    return SynapseEdges(edges.population_size, edges.pre_idx, edges.post_idx,
                        edges.weight, edges.allow_self, edges.allow_duplicates)


def propagate(previous_spikes, edges, dt, impulse_scale, *, return_details=False):
    """Accumulate pA currents; impulse_scale is J0 in pA*ms per unit efficacy.

    No spike/edge input is mutated. Duplicate contacts contribute independently.
    Every call returns fresh arrays. Causal scheduling belongs to the caller.
    """
    edges = validated_edges(edges)
    dt = scalar32('dt', dt, positive=True)
    scale = scalar32('impulse_scale', impulse_scale, nonnegative=True)
    spikes = binary_spikes(previous_spikes, edges.population_size)
    active = spikes[edges.pre_idx]
    current = np.zeros(edges.population_size, dtype=np.float32)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        impulse = active * edges.weight * scale
        per_edge = impulse / dt
        np.add.at(current, edges.post_idx, per_edge)
    if not np.all(np.isfinite(current)):
        raise FloatingPointError('Synaptic accumulation produced non-finite current')
    if return_details:
        return PropagationResult(active.astype(bool), impulse, per_edge, current)
    return current
