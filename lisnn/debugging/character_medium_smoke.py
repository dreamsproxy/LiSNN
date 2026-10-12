"""Check that characters map reproducibly to bounded direct neural input."""

import numpy as np

from lisnn.io import InterfacePort
from lisnn.media import CharacterMedium
from lisnn.network import create_nn
from lisnn.synapses import create_synapses


def character_medium_smoke_test(verbose=False):
    port = InterfacePort("text", 4, [0, 2], source="EXTERNAL")
    medium = CharacterMedium(port, seed=13, max_abs_current_pA=25)
    network = create_nn(4)
    network.configure_runtime(create_synapses(4, [], []), dt=1, impulse_scale=100)
    input_current = medium.current_for_id(medium.encode("a")[0])
    observed = network.step(input_current).external_current_pA
    ok = (medium.decode(medium.encode("abc")) == "abc"
          and np.array_equal(input_current, observed)
          and np.max(np.abs(observed)) <= 25
          and observed[1] == observed[3] == 0)
    if verbose:
        print("character medium current pA:", observed.tolist())
    return {"passed": bool(ok), "mapped_current_pA": observed.tolist()}
