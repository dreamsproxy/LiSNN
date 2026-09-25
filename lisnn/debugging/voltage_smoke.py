"""Independent voltage-gated LTP/LTD smoke diagnostics."""

import numpy as np

from lisnn.debugging.checks import report
from lisnn.plasticity import VoltageSTDP
from lisnn.synapses import create_synapses


def voltage_smoke_test(verbose=True):
    rule = VoltageSTDP(create_synapses(2, [0], [1], [0.5]),
                       tau_x_ms=10, tau_minus_ms=1, tau_plus_ms=1,
                       theta_minus_mV=-60, theta_plus_mV=-50,
                       a_ltd=0.01, a_ltp=0.001, w_max=1,
                       initial_voltage_mV=-65)
    first = rule.step([1, 0], [-65, -40], 1)
    second = rule.step([0, 0], [-65, -40], 1)
    third = rule.step([1, 0], [-65, -65], 1)
    checks = {
        "first_event_no_history": np.all(first.depression == 0) and np.all(first.potentiation == 0),
        "nonspiking_ltp": np.all(second.potentiation > 0),
        "pre_event_ltd_without_post_spike": np.all(third.depression < 0),
        "current_gate_closed_on_third": np.all(third.potentiation == 0),
        "bounded_unsigned_weight": 0 <= rule.weights[0] <= 1,
    }
    return report(checks, verbose)


if __name__ == "__main__":
    raise SystemExit(0 if voltage_smoke_test()["passed"] else 1)
