"""Smoke tests for the M2.1 sparse synaptic substrate.

This module is intentionally callable outside pytest so small topology and
validation experiments can be run interactively while developing later M2
propagation/plasticity stages.
"""

from __future__ import annotations

import numpy as np

from lisnn.synapses import create_fixed_out_degree, create_synapses


def synapse_smoke_test(
    population: int = 8,
    synapses_per_neuron: int = 3,
    seed: int = 1,
    verbose: bool = True,
) -> dict[str, object]:
    """Exercise the M2.1 sparse edge substrate and return check results.

    Checks:
    - canonical int32/float32 dtypes
    - exact edge count and fixed out-degree
    - no default autapses
    - no duplicate ordered edges
    - finite, non-negative efficacy weights
    - deterministic seeded generation
    - explicit-edge constructor correctness
    - expected rejection of invalid autapses/duplicates
    """

    if population <= 1:
        raise ValueError("population must be greater than one for synapse smoke test")
    if synapses_per_neuron < 0:
        raise ValueError("synapses_per_neuron must be non-negative")
    if synapses_per_neuron > population - 1:
        raise ValueError("synapses_per_neuron cannot exceed population - 1")

    edges = create_fixed_out_degree(
        population=population,
        synapses_per_neuron=synapses_per_neuron,
        seed=seed,
    )
    replay = create_fixed_out_degree(
        population=population,
        synapses_per_neuron=synapses_per_neuron,
        seed=seed,
    )

    expected_edges = population * synapses_per_neuron
    out_degree = np.bincount(edges.pre_idx, minlength=population)
    edge_keys = (
        edges.pre_idx.astype(np.int64) * np.int64(population)
        + edges.post_idx.astype(np.int64)
    )

    explicit = create_synapses(
        population=population,
        pre_idx=np.array([0, 1], dtype=np.int32),
        post_idx=np.array([1, 0], dtype=np.int32),
        weights=np.array([0.25, 0.75], dtype=np.float32),
    )

    autapse_rejected = False
    try:
        create_synapses(
            population=population,
            pre_idx=[0],
            post_idx=[0],
        )
    except ValueError:
        autapse_rejected = True

    duplicate_rejected = False
    try:
        create_synapses(
            population=population,
            pre_idx=[0, 0],
            post_idx=[1, 1],
        )
    except ValueError:
        duplicate_rejected = True

    checks = {
        "pre_dtype_int32": edges.pre_idx.dtype == np.int32,
        "post_dtype_int32": edges.post_idx.dtype == np.int32,
        "weight_dtype_float32": edges.weight.dtype == np.float32,
        "edge_count": edges.edge_count == expected_edges,
        "exact_out_degree": bool(np.all(out_degree == synapses_per_neuron)),
        "no_autapses": bool(np.all(edges.pre_idx != edges.post_idx)),
        "no_duplicate_edges": np.unique(edge_keys).size == edge_keys.size,
        "finite_weights": bool(np.all(np.isfinite(edges.weight))),
        "nonnegative_weights": bool(np.all(edges.weight >= np.float32(0.0))),
        "deterministic_pre": bool(np.array_equal(edges.pre_idx, replay.pre_idx)),
        "deterministic_post": bool(np.array_equal(edges.post_idx, replay.post_idx)),
        "deterministic_weight": bool(np.array_equal(edges.weight, replay.weight)),
        "explicit_constructor": (
            explicit.edge_count == 2
            and explicit.pre_idx.tolist() == [0, 1]
            and explicit.post_idx.tolist() == [1, 0]
            and np.allclose(explicit.weight, np.array([0.25, 0.75], dtype=np.float32))
        ),
        "autapse_rejected_by_default": autapse_rejected,
        "duplicate_rejected_by_default": duplicate_rejected,
    }

    passed = bool(all(checks.values()))

    if verbose:
        print("LiSNN M2.1 SYNAPSE SMOKE TEST")
        print(f"population={population} fanout={synapses_per_neuron} edges={edges.edge_count}")
        for name, ok in checks.items():
            print(f"  {name:<32} {'PASS' if ok else 'FAIL'}")
        print(f"OVERALL: {'PASS' if passed else 'FAIL'}")

    return {
        "passed": passed,
        "checks": checks,
        "population": population,
        "synapses_per_neuron": synapses_per_neuron,
        "edge_count": edges.edge_count,
        "pre_idx": edges.pre_idx.copy(),
        "post_idx": edges.post_idx.copy(),
        "weight": edges.weight.copy(),
    }


__all__ = ["synapse_smoke_test"]
