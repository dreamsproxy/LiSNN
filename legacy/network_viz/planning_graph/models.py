"""Serializable data model for the LiSNN planning graph.

The model stays independent from Qt so JSON, SQLite, tests, and future LiSNN
visualizers all operate on the same graph document.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Iterable, Mapping
from uuid import uuid4

from .grid import (
    NODE_HEIGHT,
    NODE_MIN_HEIGHT,
    NODE_MIN_WIDTH,
    NODE_WIDTH,
    snap_dimension,
    snap_value,
)


DOCUMENT_VERSION = 1
DEFAULT_GROUP_COLOR = "#3B5B78"
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")

NODE_KINDS = (
    "Idea",
    "Goal",
    "Question",
    "Task",
    "Evidence",
    "Decision",
    "Constraint",
    "Result",
    "Note",
)

NODE_STATUSES = (
    "None",
    "Backlog",
    "Active",
    "Blocked",
    "Complete",
    "Rejected",
)

EDGE_TYPES = (
    "Related",
    "Supports",
    "Contradicts",
    "Depends On",
    "Leads To",
    "Part Of",
    "Refines",
    "Blocks",
)


def _nonempty(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def normalize_hex_color(value: str) -> str:
    normalized = str(value).strip().upper()
    if not _HEX_COLOR.fullmatch(normalized):
        raise ValueError("color must use #RRGGBB format")
    return normalized


def _font_size(value: int | float, field_name: str) -> int:
    result = int(value)
    if not 6 <= result <= 72:
        raise ValueError(f"{field_name} must be within [6, 72]")
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class PlanningNode:
    """One semantic block in a planning graph."""

    node_id: str
    kind: str
    title: str
    body: str = ""
    status: str = "None"
    priority: int = 0
    tags: list[str] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    width: float = NODE_WIDTH
    height: float = NODE_HEIGHT
    header_font_size: int = 8
    title_font_size: int = 10
    body_font_size: int = 8
    footer_font_size: int = 8

    def __post_init__(self) -> None:
        self.node_id = _nonempty(self.node_id, "node_id")
        self.kind = _nonempty(self.kind, "kind")
        self.title = _nonempty(self.title, "title")
        self.status = _nonempty(self.status, "status")
        self.priority = int(self.priority)
        if not 0 <= self.priority <= 5:
            raise ValueError("priority must be within [0, 5]")
        self.tags = [str(tag).strip() for tag in self.tags if str(tag).strip()]
        self.x = snap_value(float(self.x))
        self.y = snap_value(float(self.y))
        self.width = snap_dimension(float(self.width), NODE_MIN_WIDTH)
        self.height = snap_dimension(float(self.height), NODE_MIN_HEIGHT)
        self.header_font_size = _font_size(self.header_font_size, "header_font_size")
        self.title_font_size = _font_size(self.title_font_size, "title_font_size")
        self.body_font_size = _font_size(self.body_font_size, "body_font_size")
        self.footer_font_size = _font_size(self.footer_font_size, "footer_font_size")

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        title: str,
        x: float = 0.0,
        y: float = 0.0,
        **kwargs: Any,
    ) -> "PlanningNode":
        return cls(
            node_id=str(uuid4()),
            kind=kind,
            title=title,
            x=x,
            y=y,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanningNode":
        return cls(
            node_id=str(payload["node_id"]),
            kind=str(payload.get("kind", "Note")),
            title=str(payload.get("title", "Untitled")),
            body=str(payload.get("body", "")),
            status=str(payload.get("status", "None")),
            priority=int(payload.get("priority", 0)),
            tags=[str(tag) for tag in payload.get("tags", [])],
            x=float(payload.get("x", 0.0)),
            y=float(payload.get("y", 0.0)),
            width=float(payload.get("width", NODE_WIDTH)),
            height=float(payload.get("height", NODE_HEIGHT)),
            header_font_size=int(payload.get("header_font_size", 8)),
            title_font_size=int(payload.get("title_font_size", 10)),
            body_font_size=int(payload.get("body_font_size", 8)),
            footer_font_size=int(payload.get("footer_font_size", 8)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanningEdge:
    """One directed, typed relationship between planning blocks."""

    edge_id: str
    source_id: str
    target_id: str
    relation: str = "Leads To"
    label: str = ""
    weight: float = 1.0
    route_x: float | None = None
    route_y: float | None = None

    def __post_init__(self) -> None:
        self.edge_id = _nonempty(self.edge_id, "edge_id")
        self.source_id = _nonempty(self.source_id, "source_id")
        self.target_id = _nonempty(self.target_id, "target_id")
        if self.source_id == self.target_id:
            raise ValueError("self-edges are not supported")
        if self.relation not in EDGE_TYPES:
            raise ValueError(f"unknown edge relation: {self.relation!r}")
        self.weight = float(self.weight)
        if (self.route_x is None) != (self.route_y is None):
            raise ValueError("route_x and route_y must both be set or both be None")
        if self.route_x is not None:
            self.route_x = snap_value(float(self.route_x))
            self.route_y = snap_value(float(self.route_y))

    @property
    def has_manual_route(self) -> bool:
        return self.route_x is not None and self.route_y is not None

    def clear_route(self) -> None:
        self.route_x = None
        self.route_y = None

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        target_id: str,
        relation: str = "Leads To",
        label: str = "",
        weight: float = 1.0,
    ) -> "PlanningEdge":
        return cls(
            edge_id=str(uuid4()),
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            label=label,
            weight=weight,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanningEdge":
        return cls(
            edge_id=str(payload["edge_id"]),
            source_id=str(payload["source_id"]),
            target_id=str(payload["target_id"]),
            relation=str(payload.get("relation", "Leads To")),
            label=str(payload.get("label", "")),
            weight=float(payload.get("weight", 1.0)),
            route_x=(
                None if payload.get("route_x") is None else float(payload["route_x"])
            ),
            route_y=(
                None if payload.get("route_y") is None else float(payload["route_y"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanningGroup:
    """Persistent membership plus an optional visual backdrop."""

    group_id: str
    title: str
    node_ids: list[str]
    backdrop: bool = True
    color: str = DEFAULT_GROUP_COLOR
    x: float = 0.0
    y: float = 0.0
    width: float = 400.0
    height: float = 250.0
    layer: int = 0

    def __post_init__(self) -> None:
        self.group_id = _nonempty(self.group_id, "group_id")
        self.title = _nonempty(self.title, "group title")
        self.node_ids = [str(node_id).strip() for node_id in self.node_ids]
        if len(self.node_ids) < 2:
            raise ValueError("a group must contain at least two nodes")
        if any(not node_id for node_id in self.node_ids):
            raise ValueError("group node IDs must not be empty")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("group node IDs must be unique")
        self.backdrop = bool(self.backdrop)
        self.color = normalize_hex_color(self.color)
        self.x = snap_value(float(self.x))
        self.y = snap_value(float(self.y))
        self.width = snap_dimension(float(self.width), 100.0)
        self.height = snap_dimension(float(self.height), 75.0)
        self.layer = int(self.layer)

    @classmethod
    def create(
        cls,
        *,
        title: str,
        node_ids: Iterable[str],
        x: float,
        y: float,
        width: float,
        height: float,
        backdrop: bool = True,
        color: str = DEFAULT_GROUP_COLOR,
        layer: int = 0,
    ) -> "PlanningGroup":
        return cls(
            group_id=str(uuid4()),
            title=title,
            node_ids=list(node_ids),
            backdrop=backdrop,
            color=color,
            x=x,
            y=y,
            width=width,
            height=height,
            layer=layer,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanningGroup":
        return cls(
            group_id=str(payload["group_id"]),
            title=str(payload.get("title", "Group")),
            node_ids=[str(node_id) for node_id in payload.get("node_ids", [])],
            backdrop=bool(payload.get("backdrop", True)),
            color=str(payload.get("color", DEFAULT_GROUP_COLOR)),
            x=float(payload.get("x", 0.0)),
            y=float(payload.get("y", 0.0)),
            width=float(payload.get("width", 400.0)),
            height=float(payload.get("height", 250.0)),
            layer=int(payload.get("layer", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanningDocument:
    """A complete planning graph document."""

    title: str = "Untitled Planning Graph"
    nodes: dict[str, PlanningNode] = field(default_factory=dict)
    edges: dict[str, PlanningEdge] = field(default_factory=dict)
    groups: dict[str, PlanningGroup] = field(default_factory=dict)
    version: int = DOCUMENT_VERSION
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.title = _nonempty(self.title, "document title")
        self.version = int(self.version)
        if self.version != DOCUMENT_VERSION:
            raise ValueError(
                f"unsupported planning document version {self.version}; "
                f"expected {DOCUMENT_VERSION}"
            )
        self.validate()

    def validate(self) -> None:
        if any(node_id != node.node_id for node_id, node in self.nodes.items()):
            raise ValueError("node dictionary keys must match node_id values")
        if any(edge_id != edge.edge_id for edge_id, edge in self.edges.items()):
            raise ValueError("edge dictionary keys must match edge_id values")
        if any(group_id != group.group_id for group_id, group in self.groups.items()):
            raise ValueError("group dictionary keys must match group_id values")

        known = set(self.nodes)
        dangling_edges = [
            edge.edge_id
            for edge in self.edges.values()
            if edge.source_id not in known or edge.target_id not in known
        ]
        if dangling_edges:
            raise ValueError(f"edges reference missing nodes: {dangling_edges}")

        memberships: dict[str, str] = {}
        for group in self.groups.values():
            missing = [node_id for node_id in group.node_ids if node_id not in known]
            if missing:
                raise ValueError(
                    f"group {group.group_id!r} references missing nodes: {missing}"
                )
            for node_id in group.node_ids:
                previous = memberships.get(node_id)
                if previous is not None:
                    raise ValueError(
                        f"node {node_id!r} belongs to multiple groups: "
                        f"{previous!r} and {group.group_id!r}"
                    )
                memberships[node_id] = group.group_id

    def add_node(self, node: PlanningNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node id: {node.node_id}")
        self.nodes[node.node_id] = node
        self.touch()

    def add_edge(self, edge: PlanningEdge) -> None:
        if edge.edge_id in self.edges:
            raise ValueError(f"duplicate edge id: {edge.edge_id}")
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError("edge endpoints must exist before the edge is added")
        self.edges[edge.edge_id] = edge
        self.touch()

    def add_group(self, group: PlanningGroup) -> None:
        if group.group_id in self.groups:
            raise ValueError(f"duplicate group id: {group.group_id}")
        missing = [node_id for node_id in group.node_ids if node_id not in self.nodes]
        if missing:
            raise ValueError(f"group references missing nodes: {missing}")
        occupied = {
            node_id
            for existing in self.groups.values()
            for node_id in existing.node_ids
        }
        overlap = sorted(occupied.intersection(group.node_ids))
        if overlap:
            raise ValueError(f"nodes already belong to another group: {overlap}")
        self.groups[group.group_id] = group
        self.touch()

    def remove_edge(self, edge_id: str) -> None:
        self.edges.pop(edge_id, None)
        self.touch()

    def remove_group(self, group_id: str) -> None:
        self.groups.pop(group_id, None)
        self.touch()

    def remove_group_and_nodes(self, group_id: str) -> list[str]:
        group = self.groups.get(group_id)
        if group is None:
            return []
        node_ids = list(group.node_ids)
        self.groups.pop(group_id, None)
        for node_id in node_ids:
            self.remove_node(node_id)
        self.touch()
        return node_ids

    def remove_node(self, node_id: str) -> list[str]:
        if node_id not in self.nodes:
            return []
        del self.nodes[node_id]
        removed_edges = [
            edge_id
            for edge_id, edge in self.edges.items()
            if node_id in (edge.source_id, edge.target_id)
        ]
        for edge_id in removed_edges:
            del self.edges[edge_id]

        groups_to_remove: list[str] = []
        for group in self.groups.values():
            if node_id in group.node_ids:
                group.node_ids.remove(node_id)
                if len(group.node_ids) < 2:
                    groups_to_remove.append(group.group_id)
        for group_id in groups_to_remove:
            del self.groups[group_id]
        self.touch()
        return removed_edges

    def group_for_node(self, node_id: str) -> PlanningGroup | None:
        for group in self.groups.values():
            if node_id in group.node_ids:
                return group
        return None

    def touch(self) -> None:
        self.updated_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": self.version,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "groups": [group.to_dict() for group in self.groups.values()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanningDocument":
        node_list = [PlanningNode.from_dict(item) for item in payload.get("nodes", [])]
        edge_list = [PlanningEdge.from_dict(item) for item in payload.get("edges", [])]
        group_list = [PlanningGroup.from_dict(item) for item in payload.get("groups", [])]
        return cls(
            title=str(payload.get("title", "Untitled Planning Graph")),
            nodes=_unique_map(node_list, key=lambda item: item.node_id, label="node"),
            edges=_unique_map(edge_list, key=lambda item: item.edge_id, label="edge"),
            groups=_unique_map(group_list, key=lambda item: item.group_id, label="group"),
            version=int(payload.get("version", DOCUMENT_VERSION)),
            created_at=str(payload.get("created_at", _utc_now())),
            updated_at=str(payload.get("updated_at", _utc_now())),
        )


def _unique_map(values: Iterable[Any], *, key, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        value_id = key(value)
        if value_id in result:
            raise ValueError(f"duplicate {label} id: {value_id}")
        result[value_id] = value
    return result
