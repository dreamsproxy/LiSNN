"""Editable mixed-neuron experiment with explicit currents and local learning."""

import argparse

import numpy as np

from lisnn import create_nn
from lisnn.synapses import create_fixed_out_degree


PARAMETERS = {
    "off": None,
    "pair": dict(tau_plus_ms=20, tau_minus_ms=20, a_plus=0.001, a_minus=0.001, w_max=1),
    "triplet": dict(tau_plus_ms=20, tau_minus_ms=20, tau_x_ms=50, tau_y_ms=50,
                    a2_plus=0.001, a2_minus=0.001, a3_plus=0.001, a3_minus=0.001, w_max=1),
    "voltage": dict(tau_x_ms=20, tau_minus_ms=10, tau_plus_ms=7,
                    theta_minus_mV=-70, theta_plus_mV=-50,
                    a_ltd=0.000001, a_ltp=0.00000001, w_max=1),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plasticity", choices=PARAMETERS, default="pair")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.seed < 0:
        parser.error("steps must be positive and seed nonnegative")

    # Edit composition, edges, dt, impulse integral and learner parameters here.
    network = create_nn(12, neuron_type={"default": "LIF", "AdEx": 3,
                                        "Izhikevich": 3}, seed=args.seed)
    edges = create_fixed_out_degree(12, 2, weight_range=(0.01, 0.02), seed=args.seed)
    network.configure_runtime(edges, dt=0.1, impulse_scale=100,
                              plasticity=args.plasticity,
                              plasticity_params=PARAMETERS[args.plasticity])
    initial_weights = network.runtime.weights.copy()
    spikes = np.zeros(12, dtype=np.int64)
    for tick in range(args.steps):
        # Experiment-owned current schedule; no task objective in the substrate.
        current_pA = np.full(12, 600 if tick % 100 < 50 else 0, dtype=np.float32)
        result = network.step(current_pA)
        spikes += result.current_spikes.astype(np.int64)
        if (tick + 1) % max(1, args.steps // 10) == 0 or tick + 1 == args.steps:
            filled = 20 * (tick + 1) // args.steps
            print(f"[{'=' * filled}{'.' * (20 - filled)}] {tick + 1}/{args.steps} "
                  f"spikes={int(spikes.sum())}", flush=True)

    print("model slices:", network.type_slices)
    print("spikes per cell:", spikes.tolist())
    print("final voltage mV:", result.voltage_mV.tolist())
    print("efficacy change:", (network.runtime.weights - initial_weights).tolist())
    print("This finite stimulation run demonstrates the API; it makes no adaptation claim.")


if __name__ == "__main__":
    main()
