"""Editable two-cell learning/control probe: python -m examples.recurrent_learning.

The pause is a declared 3 ms quiet interval, not evidence of durable memory.
The probe compares subsequent synaptic current and weight from the same
initial neuron/edge state under Pair learning and learning disabled.
"""

from lisnn.network import create_nn
from lisnn.neurons import kernels as k
from lisnn.synapses import create_synapses


def experiment(plasticity):
    model = create_nn(2, neuron_type="lif")
    model.pool[:, k.T_REF] = 0
    model.pool[1, k.V] = -50.1
    edges = create_synapses(2, [0], [1], [0.5])
    params = dict(tau_plus_ms=10, tau_minus_ms=10, a_plus=0.2,
                  a_minus=0.1, w_max=1) if plasticity == "pair" else None
    model.configure_runtime(edges, dt=0.1, impulse_scale=100,
                            plasticity=plasticity, plasticity_params=params)
    model.step([32000, 0])  # Pre event at tick 0.
    training = model.step()  # The propagated current induces post spike at tick 1.
    for _ in range(30):
        model.step()  # Quiet for 3 ms; no input or feedback current.
    model.step([32000, 0])
    probe = model.step()
    return training, probe, model.runtime.weights.copy()


def main():
    for rule in ("off", "pair"):
        training, probe, weights = experiment(rule)
        print(f"{rule:>4} training spikes={training.current_spikes.tolist()} "
              f"probe synaptic current={probe.propagation.synaptic_current_pA[1]:.3f} pA "
              f"final efficacy={weights[0]:.6f}")


if __name__ == "__main__":
    main()
