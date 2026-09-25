"""Run a tiny synthetic character medium with an explicit presentation rule."""

from lisnn.io import InterfacePort
from lisnn.media import CharacterMedium
from lisnn.network import create_nn
from lisnn.synapses import create_synapses


def main():
    text = "cat"
    network = create_nn(8)
    network.configure_runtime(create_synapses(8, [], []), dt=1, impulse_scale=100)
    port = InterfacePort("text", 8, [0, 1, 2, 3], source="EXTERNAL")
    medium = CharacterMedium(port, seed=7, max_abs_current_pA=80)

    # This is an experiment rule, not a property of characters or simulation.
    integration_steps_per_character = 3
    for character in text:
        current = medium.current_for_character(character)
        for _ in range(integration_steps_per_character):
            result = network.step(current)
            print(repr(character), "time_ms=", result.time_ms,
                  "input_pA=", result.external_current_pA.tolist(),
                  "spikes=", result.current_spikes.tolist())


if __name__ == "__main__":
    main()
