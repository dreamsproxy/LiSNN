"""Construction and optional spatial-shell checks without pytest dependencies."""

import numpy as np

from lisnn.debugging.checks import rejects, report
from lisnn.network import create_nn
from lisnn.neurons import NeuronType
from lisnn.neurons import kernels as k
from lisnn.spatial import MorphologyTable


def network_smoke_test(verbose=True):
    checks = {}
    for neuron_type in NeuronType:
        model = create_nn(4, neuron_type=neuron_type, randomize_params=True, seed=7)
        again = create_nn(4, neuron_type=neuron_type, randomize_params=True, seed=7)
        checks[f"{neuron_type.value}_layout"] = model.pool.shape == (4, k.NEURON_WIDTH) and model.pool.dtype == np.float32
        checks[f"{neuron_type.value}_seed"] = np.array_equal(model.pool, again.pool)
        checks[f"{neuron_type.value}_finite"] = np.all(np.isfinite(model.pool))
        checks[f"{neuron_type.value}_slice"] = model.type_slices[neuron_type] == slice(0, 4)
    mixed = create_nn(8, neuron_type={"default": "LIF", "GLIF5": 2, "AdEx": 3})
    checks["mixed_counts"] = mixed.type_counts == {NeuronType.GLIF5: 2, NeuronType.ADEX: 3, NeuronType.LIF: 3}
    checks["mixed_slices"] = mixed.type_slices == {NeuronType.GLIF5: slice(0, 2), NeuronType.ADEX: slice(2, 5), NeuronType.LIF: slice(5, 8)}
    checks["optional_space"] = mixed.space is None
    izh = create_nn(4, neuron_type="Izhikevich")
    checks["izh_recovery_initialization"] = np.allclose(izh.pool[:, k.ADAPT], izh.pool[:, k.IZH_B] * izh.pool[:, k.V])
    return report(checks, verbose)


def spatial_smoke_test(verbose=True):
    config = {"size": (100, 200, 300), "origin": (-50, 10, 25), "placement": "uniform"}
    kwargs = {"population": 8, "neuron_type": "AdEx", "randomize_params": True, "seed": 7}
    model = create_nn(**kwargs, spatial=config)
    again = create_nn(**kwargs, spatial=config)
    bare = create_nn(**kwargs)
    space = model.space
    positions = np.array([[0, 0, 0], [5, 5, 5]], dtype=np.float32)
    explicit = create_nn(2, spatial={"size": (10, 10, 10), "placement": "explicit", "positions": positions})
    checks = {
        "coordinate_layout": space.positions.shape == (8, 3) and space.positions.dtype == np.float32,
        "finite_positions": np.all(np.isfinite(space.positions)),
        "inside_volume": np.all(space.positions >= space.origin) and np.all(space.positions <= space.origin + space.size),
        "seeded_positions": np.array_equal(space.positions, again.space.positions),
        "independent_neuron_rng": np.array_equal(model.pool, bare.pool),
        "explicit_positions": np.array_equal(explicit.space.positions, positions),
        "out_of_bounds_rejected": rejects(ValueError, create_nn, 2, spatial={"size": (1, 1, 1), "placement": "explicit", "positions": positions}),
        "morphology_shell_empty": MorphologyTable.empty().compartment_count == 0,
    }
    return report(checks, verbose)
