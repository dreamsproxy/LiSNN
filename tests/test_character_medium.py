"""Contract checks for the synthetic text medium."""

import numpy as np
import pytest

from lisnn.io import InterfacePort
from lisnn.media import CharacterMedium
from lisnn.network import create_nn
from lisnn.synapses import create_synapses


def test_ascii_ids_roundtrip_and_non_ascii_fails():
    assert CharacterMedium.encode("aA\n") == (97, 65, 10)
    assert CharacterMedium.decode((97, 65, 10)) == "aA\n"
    with pytest.raises(ValueError, match="ASCII"):
        CharacterMedium.encode("猫")
    with pytest.raises(ValueError, match="ASCII"):
        CharacterMedium.decode((128,))


def test_embedding_is_bounded_seeded_and_spatial():
    port = InterfacePort("characters", 5, [1, 3, 4], source="EXTERNAL")
    medium = CharacterMedium(port, seed=9, max_abs_current_pA=20)
    other = CharacterMedium(port, seed=9, max_abs_current_pA=20)
    assert medium.embeddings.shape == (128, 3)
    np.testing.assert_array_equal(medium.embeddings, other.embeddings)
    a = medium.current_for_character("a")
    assert a.dtype == np.float32 and a.shape == (5,)
    assert a[0] == a[2] == 0 and np.max(np.abs(a)) <= 20
    assert not np.array_equal(a, medium.current_for_character("b"))
    a[:] = 0
    assert np.any(medium.current_for_character("a") != 0)
    with pytest.raises(ValueError):
        medium.current_for_character("ab")


def test_current_is_accepted_by_neural_runtime_without_frame_scheduler():
    network = create_nn(4)
    network.configure_runtime(create_synapses(4, [], []), dt=1, impulse_scale=100)
    port = InterfacePort("text", 4, [0, 2], source="EXTERNAL")
    medium = CharacterMedium(port, seed=0, max_abs_current_pA=30)
    current = medium.current_for_character("a")
    first = network.step(current)
    second = network.step(current)
    np.testing.assert_array_equal(first.external_current_pA, current)
    np.testing.assert_array_equal(second.external_current_pA, current)
    assert first.time_ms == 0 and second.time_ms == 1
