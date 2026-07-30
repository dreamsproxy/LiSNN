from __future__ import annotations

import numpy as np

POTENTIAL = 0
DT = 1
TAU = 2
V_REST = 3
V_RESET = 4
V_THRESHOLD = 5
N_PARAMS = 6


def initialize_population(
    num_neurons: int,
    dt: float,
    rng: np.random.Generator,
    fixed_tail: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the token and timecode neuron population."""

    if num_neurons < 1:
        raise ValueError("num_neurons must be positive")
    if fixed_tail < 0 or fixed_tail > num_neurons:
        raise ValueError("fixed_tail must be between 0 and num_neurons")

    v_rest = rng.uniform(-66.0, -64.0, num_neurons).astype(np.float64)
    v_reset = v_rest - np.float64(5.0)
    tau = rng.uniform(19.5, 20.5, num_neurons).astype(np.float64)
    thresholds = rng.uniform(-55.0, -45.0, num_neurons).astype(np.float64)
    potentials = v_rest + np.float64(2.71)

    if fixed_tail:
        tail = slice(num_neurons - fixed_tail, num_neurons)
        v_rest[tail] = -65.0
        v_reset[tail] = -70.0
        tau[tail] = 20.0
        thresholds[tail] = -55.0
        potentials[tail] = -65.0

    neurons = np.column_stack(
        (
            potentials,
            np.full(num_neurons, dt, dtype=np.float64),
            tau,
            v_rest,
            v_reset,
            thresholds,
        )
    )
    return neurons, thresholds, tau


def step_population(
    neurons: np.ndarray,
    input_current: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Advance a LIF population and return non-binary spike magnitudes."""

    if neurons.ndim != 2 or neurons.shape[1] != N_PARAMS:
        raise ValueError(f"neurons must have shape (N, {N_PARAMS})")
    if input_current.shape != (neurons.shape[0],):
        raise ValueError("input_current shape does not match the neuron population")
    if thresholds.shape != input_current.shape:
        raise ValueError("threshold shape does not match the neuron population")

    potential = neurons[:, POTENTIAL]
    delta_v = (
        (neurons[:, V_REST] - potential) + input_current
    ) * (neurons[:, DT] / neurons[:, TAU])
    pre_reset_potential = potential + delta_v

    finite = np.isfinite(pre_reset_potential)
    pre_reset_potential = np.where(
        finite,
        pre_reset_potential,
        neurons[:, V_REST],
    )
    spikes = np.maximum(0.0, pre_reset_potential - thresholds)
    fired = spikes > 0.0
    neurons[:, POTENTIAL] = np.where(
        fired,
        neurons[:, V_RESET],
        pre_reset_potential,
    )
    return spikes.astype(np.float64, copy=False)
