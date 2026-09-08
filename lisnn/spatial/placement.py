"""Spatial placement helpers for LiSNN.

This module contains geometry-only operations. It intentionally does not
implement synaptic connectivity, membrane physics, diffusion, morphology,
or probe coupling.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


PositionArray = NDArray[np.float32]
Vector3 = tuple[float, float, float] | Sequence[float] | NDArray[np.floating]


def as_vector3(value: Vector3, *, name: str) -> NDArray[np.float32]:
    """Return a finite float32 ``(3,)`` vector.

    Parameters
    ----------
    value:
        Any three-element numeric sequence.
    name:
        Human-readable field name used in validation errors.
    """

    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_volume_geometry(
    size: Vector3,
    origin: Vector3,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Validate and normalize a rectangular volume definition."""

    size_array = as_vector3(size, name="size")
    origin_array = as_vector3(origin, name="origin")

    if np.any(size_array <= 0.0):
        raise ValueError("every spatial volume dimension must be greater than zero")

    return size_array, origin_array


def uniform_positions(
    population: int,
    size: NDArray[np.float32],
    origin: NDArray[np.float32],
    *,
    rng: np.random.Generator,
) -> PositionArray:
    """Place soma centers uniformly inside an axis-aligned rectangular volume.

    The generated coordinates represent soma centers only. They do not imply
    soma radius, collision avoidance, dendritic extent, or axonal geometry.
    """

    if population <= 0:
        raise ValueError("population must be greater than zero")

    unit_positions = rng.random(
        (population, 3),
        dtype=np.float32,
    )

    return (
        origin[None, :]
        + unit_positions * size[None, :]
    ).astype(np.float32, copy=False)


def validate_explicit_positions(
    positions: object,
    population: int,
    size: NDArray[np.float32],
    origin: NDArray[np.float32],
) -> PositionArray:
    """Validate explicit soma coordinates against the enclosing volume."""

    array = np.asarray(positions, dtype=np.float32)

    expected_shape = (population, 3)
    if array.shape != expected_shape:
        raise ValueError(
            "explicit spatial positions must have shape "
            f"{expected_shape}, received {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError("explicit spatial positions must contain only finite values")

    lower = origin[None, :]
    upper = (origin + size)[None, :]

    if np.any(array < lower) or np.any(array > upper):
        raise ValueError("explicit spatial positions must lie inside the spatial volume")

    return array.astype(np.float32, copy=False)
