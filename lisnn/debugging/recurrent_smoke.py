"""Independent small recurrent learner/checkpoint smoke suite."""

import numpy as np

from lisnn.debugging.checks import report
from lisnn.network import create_nn
from lisnn.neurons import kernels as k
from lisnn.synapses import create_synapses


def recurrent_smoke_test(verbose=True):
    model = create_nn(2, neuron_type="lif")
    model.pool[:, k.T_REF] = 0
    model.pool[1, k.V] = -50.1
    model.configure_runtime(create_synapses(2, [0], [1], [0.5]), dt=0.1,
                            impulse_scale=100, plasticity="pair",
                            plasticity_params=dict(tau_plus_ms=10, tau_minus_ms=10,
                                                   a_plus=0.2, a_minus=0.1, w_max=1))
    model.step([32000, 0])
    previous = model.step()
    learned = previous.weights_after[0]
    model.step([32000, 0])
    consequence = model.step()
    checks = {
        "post_spike_from_previous_pre": np.array_equal(previous.current_spikes, [0, 1]),
        "pair_learning": learned > 0.5,
        "changed_future_current": consequence.propagation.synaptic_current_pA[1] > 500,
    }
    return report(checks, verbose)


if __name__ == "__main__":
    raise SystemExit(0 if recurrent_smoke_test()["passed"] else 1)
