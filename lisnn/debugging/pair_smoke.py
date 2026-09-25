"""Independently callable reference Pair STDP smoke test."""

import numpy as np

from lisnn.debugging.checks import report
from lisnn.plasticity import PairSTDP
from lisnn.synapses import create_synapses


def pair_smoke_test(verbose=True):
    edges = create_synapses(3, [0, 1], [2, 2], [0.5, 0.5])
    rule = PairSTDP(edges, tau_plus_ms=10, tau_minus_ms=10,
                    a_plus=0.1, a_minus=0.12, w_max=1)
    rule.step([1, 0, 0], 1)
    rule.step([0, 1, 0], 1)
    update = rule.step([0, 0, 1], 1)
    checks = {
        "pre_before_post_ltp": np.allclose(update.potentiation,
                                             [0.1 * np.exp(-0.2), 0.1 * np.exp(-0.1)]),
        "recent_pre_stronger": rule.weights[1] > rule.weights[0] > 0.5,
        "no_ltd_without_post_history": np.all(update.depression == 0),
        "fixed_input_edges": np.allclose(edges.weight, [0.5, 0.5]),
    }
    return report(checks, verbose)


if __name__ == "__main__":
    raise SystemExit(0 if pair_smoke_test()["passed"] else 1)
