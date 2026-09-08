"""Shared public type aliases for LiSNN."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


Float32Array: TypeAlias = NDArray[np.float32]
NeuronPopulation: TypeAlias = NDArray[np.float32]
InputCurrent: TypeAlias = float | np.float32 | NDArray[np.float32]
SpikeVector: TypeAlias = NDArray[np.float32]
Seed: TypeAlias = int | None

NeuronTypeLike: TypeAlias = str | "NeuronType"
PopulationSpec: TypeAlias = (
    NeuronTypeLike
    | Mapping[str | "NeuronType", int | NeuronTypeLike]
)

NeuronStep: TypeAlias = Callable[
    [NeuronPopulation, InputCurrent, float | np.float32],
    SpikeVector,
]
