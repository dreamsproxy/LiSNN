"""Independently callable ACTIVE/SLEEP transition diagnostics."""

import numpy as np

from lisnn.debugging.checks import report
from lisnn.io import InterfacePort, RegimeController, StreamRuntime, Transducer
from lisnn.network import create_nn
from lisnn.synapses import create_synapses


def regime_smoke_test(verbose=True):
    network = create_nn(2)
    network.configure_runtime(create_synapses(2, [], []), dt=1, impulse_scale=100)
    ports = {"inner": Transducer(InterfacePort("inner", 2, [0, 1], source="INTERNAL"))}
    control = RegimeController(StreamRuntime(network, transducers=ports))
    control.transition(1, "SLEEP")
    control.transition(3, "ACTIVE")
    active = control.step([100, 0])
    sleep = control.step([100, 0])
    control.pulse(port_name="inner", current_pA=[0, 10])
    pulse = control.step()
    again = control.step([100, 0])
    checks = {
        "external_attenuation": np.allclose(active.routed.neural.external_current_pA, [100, 0])
                                and np.allclose(sleep.routed.neural.external_current_pA, [5, 0]),
        "internal_pulse": np.allclose(pulse.routed.neural.feedback_current_pA, [0, 10]),
        "state_returns": again.regime.name == "ACTIVE" and again.regime.actions_enabled,
        "continuing_clock": control.runtime.tick == 4,
    }
    return report(checks, verbose)


if __name__ == "__main__":
    raise SystemExit(0 if regime_smoke_test()["passed"] else 1)
