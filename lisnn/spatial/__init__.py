"""Spatial substrate API for LiSNN."""

from lisnn.spatial.morphology import CompartmentType, MorphologyTable
from lisnn.spatial.placement import PositionArray, Vector3
from lisnn.spatial.volume import (
    BoundaryMode,
    PlacementMode,
    SpatialConfig,
    SpatialSpec,
    SpatialVolume,
    create_spatial_volume,
)

__all__ = [
    "BoundaryMode",
    "PlacementMode",
    "SpatialConfig",
    "SpatialSpec",
    "SpatialVolume",
    "create_spatial_volume",
    "PositionArray",
    "Vector3",
    "CompartmentType",
    "MorphologyTable",
]
