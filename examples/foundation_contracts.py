"""Run with: python -m examples.foundation_contracts"""

import numpy as np

from lisnn.neurons import kernels as k
from lisnn.synapses import create_synapses


def main():
    for index in [2**32, -(2**32)]:
        try:
            create_synapses(2, [index], [1])
        except IndexError as exc:
            print(f"Rejected original index {index}: {exc}")
        else:
            raise AssertionError("An out-of-range index wrapped into the population")

    edges = create_synapses(3, [2, 0], [1, 2])
    print("Valid directed edges:", list(zip(edges.pre_idx.tolist(), edges.post_idx.tolist())))

    pool = k.new_population(2)
    pool[:, k.V] = [-55, -65]
    spikes = k.lif_step(pool, [0, 100], dt=1)
    expected = np.array([-55.5, -64.5])
    np.testing.assert_allclose(pool[:, k.V], expected, atol=1e-5, rtol=0)
    np.testing.assert_array_equal(spikes, [0, 0])
    print("LIF voltages after 1 ms:", pool[:, k.V].tolist(), "mV")
    print("Expected:", expected.tolist(), "mV; spikes:", spikes.tolist())


if __name__ == "__main__":
    main()
