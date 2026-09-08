"""Future morphology interfaces for LiSNN spatial simulations.

Milestone 1 does not construct axons or dendrites. This module only reserves a
stable vocabulary and array layout for future compartment geometry so later
work does not overload soma coordinates with morphology data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class CompartmentType(IntEnum):
    """Reserved compartment identifiers for future morphology work."""

    SOMA = 0
    DENDRITE = 1
    AXON = 2
    AXON_TERMINAL = 3
    NODE_OF_RANVIER = 4


@dataclass(slots=True)
class MorphologyTable:
    """Vectorized compartment-geometry container.

    This is an intentionally inactive shell. A future morphology phase may use
    one row per physical compartment/path node while keeping ownership and
    parentage explicit.
    """

    owner_neuron: NDArray[np.int32]
    parent: NDArray[np.int32]
    position: NDArray[np.float32]
    radius: NDArray[np.float32]
    compartment_type: NDArray[np.uint8]

    @classmethod
    def empty(cls) -> "MorphologyTable":
        """Return an empty morphology container for future attachment."""

        return cls(
            owner_neuron=np.empty(0, dtype=np.int32),
            parent=np.empty(0, dtype=np.int32),
            position=np.empty((0, 3), dtype=np.float32),
            radius=np.empty(0, dtype=np.float32),
            compartment_type=np.empty(0, dtype=np.uint8),
        )

    @property
    def compartment_count(self) -> int:
        return int(self.owner_neuron.shape[0])
