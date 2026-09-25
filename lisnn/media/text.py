"""A fixed, 7-bit ASCII synthetic medium; no token-to-tick policy."""

import numpy as np

from lisnn.io import InterfacePort
from lisnn.validation import scalar32


class CharacterMedium:
    """ASCII IDs and fixed seeded embeddings to an EXTERNAL port, in pA.

    Presentation timing belongs to the experiment: `current_for_id` returns
    one spatial current pattern, with no frame, tick or implicit hold duration.
    Embeddings are synthetic lookup vectors, not trained linguistic semantics.
    """

    def __init__(self, port, *, seed, max_abs_current_pA):
        if not isinstance(port, InterfacePort) or port.source != "EXTERNAL" or port.role != "input":
            raise ValueError("character medium requires an EXTERNAL input port")
        if len(port.neuron_indices) == 0:
            raise ValueError("character medium requires at least one input neuron")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.port = port
        self.max_abs_current_pA = scalar32("max_abs_current_pA", max_abs_current_pA, positive=True)
        rng = np.random.default_rng(int(seed))
        embedding = rng.uniform(-1, 1, (128, len(port.neuron_indices)))
        embedding /= np.max(np.abs(embedding), axis=1, keepdims=True)
        self._embedding = embedding.astype(np.float32)
        self._embedding.flags.writeable = False

    @property
    def embeddings(self):
        return self._embedding.copy()

    @staticmethod
    def encode(text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        try:
            return tuple(text.encode("ascii"))
        except UnicodeEncodeError as error:
            raise ValueError("text must contain only 7-bit ASCII characters") from error

    @staticmethod
    def decode(ids):
        values = []
        for token_id in ids:
            if isinstance(token_id, (bool, np.bool_)) or not isinstance(token_id, (int, np.integer)):
                raise TypeError("character IDs must be integers")
            if not 0 <= token_id < 128:
                raise ValueError("character ID must be in the ASCII range 0..127")
            values.append(int(token_id))
        return bytes(values).decode("ascii")

    def current_for_id(self, token_id):
        if isinstance(token_id, (bool, np.bool_)) or not isinstance(token_id, (int, np.integer)):
            raise TypeError("character ID must be an integer")
        if not 0 <= token_id < 128:
            raise ValueError("character ID must be in the ASCII range 0..127")
        result = np.zeros(self.port.population_size, dtype=np.float32)
        result[self.port.neuron_indices] = self._embedding[token_id] * self.max_abs_current_pA
        return result

    def current_for_character(self, character):
        ids = self.encode(character)
        if len(ids) != 1:
            raise ValueError("supply exactly one character")
        return self.current_for_id(ids[0])
