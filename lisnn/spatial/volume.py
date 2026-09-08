"""First-class spatial volume for LiSNN.

Milestone 1 intentionally models geometry only. The volume provides a common
coordinate frame for soma positions and future morphology, extracellular
fields, probes, diffusion, vasculature, and clearance systems.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, NotRequired, Required, TypedDict, TypeAlias

import numpy as np
from numpy.typing import NDArray

from lisnn.spatial.placement import (
    PositionArray,
    Vector3,
    uniform_positions,
    validate_explicit_positions,
    validate_volume_geometry,
)
from lisnn.types import Seed


BoundaryMode: TypeAlias = Literal["closed"]
PlacementMode: TypeAlias = Literal["uniform", "explicit"]


class SpatialConfig(TypedDict, total=False):
    """Dictionary form accepted by ``Network.create_nn(spatial=...)``."""

    size: Required[Vector3]
    origin: NotRequired[Vector3]
    units: NotRequired[str]
    boundary: NotRequired[BoundaryMode]
    placement: NotRequired[PlacementMode]
    positions: NotRequired[object]


@dataclass(slots=True)
class SpatialVolume:
    """Axis-aligned physical volume and soma-center coordinates.

    Coordinates currently use micrometres (``"um"``) as the sole supported
    physical unit. ``positions[n]`` is the soma center of neuron ``n``.

    No physical interactions are implied by the existence of this object.
    In particular, Milestone 1 does not implement collision avoidance,
    morphology, extracellular diffusion, ion transport, neurotransmitter
    transport, tissue regions, conduction delays, or probe physics.
    """

    size: NDArray[np.float32]
    origin: NDArray[np.float32]
    positions: PositionArray
    units: str = "um"
    boundary: BoundaryMode = "closed"
    placement: PlacementMode = "uniform"

    @property
    def population_size(self) -> int:
        """Number of soma coordinates stored in the volume."""

        return int(self.positions.shape[0])

    @property
    def min_corner(self) -> NDArray[np.float32]:
        """Lower XYZ corner of the volume."""

        return self.origin.copy()

    @property
    def max_corner(self) -> NDArray[np.float32]:
        """Upper XYZ corner of the volume."""

        return (self.origin + self.size).astype(np.float32, copy=False)

    @property
    def center(self) -> NDArray[np.float32]:
        """Center XYZ coordinate of the volume."""

        return (self.origin + self.size * np.float32(0.5)).astype(
            np.float32,
            copy=False,
        )

    @property
    def volume_um3(self) -> float:
        """Rectangular volume in cubic micrometres."""

        return float(np.prod(self.size, dtype=np.float32))

    def __repr__(self) -> str:
        size = tuple(float(value) for value in self.size)
        return (
            "SpatialVolume("
            f"size={size} {self.units}, "
            f"population={self.population_size}, "
            f"placement={self.placement!r}, "
            f"boundary={self.boundary!r}"
            ")"
        )


SpatialSpec: TypeAlias = SpatialVolume | SpatialConfig | Mapping[str, object] | None


def _validate_units(units: object) -> str:
    """Normalize spatial units.

    Micrometres are intentionally the only Milestone-1 unit. Supporting one
    explicit physical unit avoids silent conversions while future morphology
    and field solvers are still undefined.
    """

    if not isinstance(units, str):
        raise TypeError("spatial units must be a string")

    normalized = units.strip().lower()
    if normalized not in {"um", "µm", "μm"}:
        raise ValueError("Milestone 1 currently supports spatial units='um' only")

    return "um"


def _validate_boundary(boundary: object) -> BoundaryMode:
    if boundary != "closed":
        raise ValueError("Milestone 1 currently supports boundary='closed' only")
    return "closed"


def _validate_placement(placement: object) -> PlacementMode:
    if placement not in {"uniform", "explicit"}:
        raise ValueError("spatial placement must be 'uniform' or 'explicit'")
    return placement  # type: ignore[return-value]


def create_spatial_volume(
    population: int,
    spatial: SpatialSpec,
    *,
    seed: Seed = None,
) -> SpatialVolume | None:
    """Create or validate the optional spatial substrate for a network.

    Parameters
    ----------
    population:
        Number of neurons in the owning SNN.
    spatial:
        ``None`` for a non-spatial network, an existing ``SpatialVolume``, or
        a mapping with at least ``size=(x, y, z)``.
    seed:
        Seed used only for spatial placement. It creates an independent NumPy
        generator, so adding geometry does not consume the neuron-parameter
        generator state.
    """

    if spatial is None:
        return None

    if isinstance(spatial, SpatialVolume):
        if spatial.population_size != population:
            raise ValueError(
                "SpatialVolume population does not match network population: "
                f"{spatial.population_size} != {population}"
            )
        return spatial

    if not isinstance(spatial, Mapping):
        raise TypeError("spatial must be None, SpatialVolume, or a mapping")

    if "size" not in spatial:
        raise ValueError("spatial configuration requires size=(x, y, z)")

    size, origin = validate_volume_geometry(
        spatial["size"],  # type: ignore[arg-type]
        spatial.get("origin", (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
    )

    units = _validate_units(spatial.get("units", "um"))
    boundary = _validate_boundary(spatial.get("boundary", "closed"))
    placement = _validate_placement(spatial.get("placement", "uniform"))

    if placement == "uniform":
        if "positions" in spatial:
            raise ValueError(
                "spatial positions were provided with placement='uniform'; "
                "use placement='explicit'"
            )

        rng = np.random.default_rng(seed)
        positions = uniform_positions(
            population,
            size,
            origin,
            rng=rng,
        )

    else:
        if "positions" not in spatial:
            raise ValueError("placement='explicit' requires spatial['positions']")

        positions = validate_explicit_positions(
            spatial["positions"],
            population,
            size,
            origin,
        )

    return SpatialVolume(
        size=size,
        origin=origin,
        positions=positions,
        units=units,
        boundary=boundary,
        placement=placement,
    )
