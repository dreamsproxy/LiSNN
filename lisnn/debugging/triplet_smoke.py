"""Independently callable triplet timing diagnostics."""

import numpy as np

from lisnn.debugging.checks import report
from lisnn.plasticity import TripletSTDP
from lisnn.synapses import create_synapses


def triplet_smoke_test(verbose=True):
    rule = TripletSTDP(create_synapses(2, [0], [1], [0.5]),
                       tau_plus_ms=10, tau_minus_ms=10,
                       tau_x_ms=20, tau_y_ms=20, a2_plus=0.1,
                       a2_minus=0.12, a3_plus=0.2, a3_minus=0.15, w_max=1)
    rule.step([1, 0], 1)
    pair = rule.step([0, 1], 1)
    triplet = rule.step([0, 1], 1)
    checks = {
        "pair_ltp": np.allclose(pair.pair_ltp, [0.1 * np.exp(-0.1)]),
        "first_post_no_triplet": np.array_equal(pair.triplet_ltp, [0]),
        "second_post_triplet": np.allclose(triplet.triplet_ltp,
                                            [0.2 * np.exp(-0.2) * np.exp(-0.05)]),
        "bounded_unsigned_weights": 0 <= rule.weights[0] <= 1,
    }
    return report(checks, verbose)


if __name__ == "__main__":
    raise SystemExit(0 if triplet_smoke_test()["passed"] else 1)
