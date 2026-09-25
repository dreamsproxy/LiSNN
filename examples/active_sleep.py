"""Editable ACTIVE -> SLEEP -> ACTIVE policy example."""

from lisnn.io import InterfacePort, RegimeController, StreamRuntime, Transducer
from lisnn.network import create_nn
from lisnn.synapses import create_synapses


def main():
    model = create_nn(2)
    model.configure_runtime(create_synapses(2, [], []), dt=1, impulse_scale=100)
    ports = {"inner": Transducer(InterfacePort("inner", 2, [0, 1], source="INTERNAL"))}
    regime = RegimeController(StreamRuntime(model, transducers=ports), seed=4,
                              background_port="inner", background_probability=0.5,
                              background_amplitude_pA=5, max_background_events=2)
    regime.transition(1, "SLEEP")
    regime.transition(5, "ACTIVE")
    for t in range(7):
        result = regime.step([100, 0])
        print(f"tick={t} state={result.regime.name} "
              f"external={result.routed.neural.external_current_pA[0]:.2f} pA "
              f"feedback={result.routed.neural.feedback_current_pA.tolist()} "
              f"learning={result.regime.plasticity_enabled} "
              f"actions={result.regime.actions_enabled}")
        if t == 1:
            regime.pulse(port_name="inner", current_pA=[0, 10])


if __name__ == "__main__":
    main()
