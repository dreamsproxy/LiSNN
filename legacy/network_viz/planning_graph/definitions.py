"""Persistent block and status definitions for the planning graph.

The registry is intentionally independent from Qt. The GUI reads and writes the
single ``custom_blocks.json`` file beside this module so definitions survive
between application instances and remain easy to inspect or edit manually.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


DEFINITION_VERSION = 1
DEFAULT_DEFINITION_PATH = Path(__file__).with_name("custom_blocks.json")
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


DEFAULT_BLOCK_TYPES = (
    ("Idea", "#70A7FF"),
    ("Goal", "#63D297"),
    ("Question", "#B58BFF"),
    ("Task", "#F3C969"),
    ("Evidence", "#58C8D4"),
    ("Decision", "#FF956D"),
    ("Constraint", "#FF7890"),
    ("Result", "#93D36E"),
    ("Note", "#AAB4BE"),
)

DEFAULT_STATUSES = (
    ("None", "#40556A"),
    ("Backlog", "#94A3B2"),
    ("Active", "#4A9EFF"),
    ("Blocked", "#F06B6B"),
    ("Complete", "#63D297"),
    ("Rejected", "#A06A78"),
)


def normalize_color(value: str) -> str:
    normalized = str(value).strip().upper()
    if not _HEX_COLOR.fullmatch(normalized):
        raise ValueError("color must use #RRGGBB format")
    return normalized


def _name(value: str, field_name: str = "name") -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


@dataclass(frozen=True)
class VisualDefinition:
    name: str
    color: str
    builtin: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name))
        object.__setattr__(self, "color", normalize_color(self.color))
        object.__setattr__(self, "builtin", bool(self.builtin))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VisualDefinition":
        return cls(
            name=str(payload["name"]),
            color=str(payload["color"]),
            builtin=bool(payload.get("builtin", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "color": self.color, "builtin": self.builtin}


class DefinitionRegistry:
    """Mutable registry backed by one human-readable JSON file."""

    def __init__(
        self,
        *,
        block_types: Iterable[VisualDefinition],
        statuses: Iterable[VisualDefinition],
        path: str | Path = DEFAULT_DEFINITION_PATH,
    ) -> None:
        self.path = Path(path)
        self.block_types = self._unique(block_types, "block type")
        self.statuses = self._unique(statuses, "status")
        self._merge_defaults()

    @staticmethod
    def _unique(
        values: Iterable[VisualDefinition],
        label: str,
    ) -> dict[str, VisualDefinition]:
        result: dict[str, VisualDefinition] = {}
        for value in values:
            key = value.name.casefold()
            if key in result:
                raise ValueError(f"duplicate {label}: {value.name}")
            result[key] = value
        return result

    def _merge_defaults(self) -> None:
        for name, color in DEFAULT_BLOCK_TYPES:
            self.block_types.setdefault(
                name.casefold(),
                VisualDefinition(name=name, color=color, builtin=True),
            )
        for name, color in DEFAULT_STATUSES:
            self.statuses.setdefault(
                name.casefold(),
                VisualDefinition(name=name, color=color, builtin=True),
            )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_DEFINITION_PATH) -> "DefinitionRegistry":
        source = Path(path)
        if not source.exists():
            registry = cls.defaults(path=source)
            registry.save()
            return registry

        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        version = int(payload.get("version", DEFINITION_VERSION))
        if version != DEFINITION_VERSION:
            raise ValueError(
                f"unsupported definition version {version}; expected {DEFINITION_VERSION}"
            )
        registry = cls(
            block_types=[
                VisualDefinition.from_dict(item)
                for item in payload.get("block_types", [])
            ],
            statuses=[
                VisualDefinition.from_dict(item)
                for item in payload.get("statuses", [])
            ],
            path=source,
        )
        return registry

    @classmethod
    def defaults(
        cls,
        path: str | Path = DEFAULT_DEFINITION_PATH,
    ) -> "DefinitionRegistry":
        return cls(
            block_types=[
                VisualDefinition(name=name, color=color, builtin=True)
                for name, color in DEFAULT_BLOCK_TYPES
            ],
            statuses=[
                VisualDefinition(name=name, color=color, builtin=True)
                for name, color in DEFAULT_STATUSES
            ],
            path=path,
        )

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return self.path

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": DEFINITION_VERSION,
            "block_types": [item.to_dict() for item in self.block_types.values()],
            "statuses": [item.to_dict() for item in self.statuses.values()],
        }

    def add_block_type(self, name: str, color: str) -> VisualDefinition:
        definition = VisualDefinition(name=name, color=color, builtin=False)
        key = definition.name.casefold()
        if key in self.block_types:
            raise ValueError(f"block type already exists: {definition.name}")
        self.block_types[key] = definition
        self.save()
        return definition

    def add_status(self, name: str, color: str) -> VisualDefinition:
        definition = VisualDefinition(name=name, color=color, builtin=False)
        key = definition.name.casefold()
        if key in self.statuses:
            raise ValueError(f"status already exists: {definition.name}")
        self.statuses[key] = definition
        self.save()
        return definition

    def block_names(self) -> list[str]:
        return [item.name for item in self.block_types.values()]

    def status_names(self) -> list[str]:
        return [item.name for item in self.statuses.values()]

    def block_color(self, name: str, fallback: str = "#AAB4BE") -> str:
        definition = self.block_types.get(str(name).casefold())
        return fallback if definition is None else definition.color

    def status_color(self, name: str, fallback: str = "#40556A") -> str:
        definition = self.statuses.get(str(name).casefold())
        return fallback if definition is None else definition.color

    def has_block_type(self, name: str) -> bool:
        return str(name).casefold() in self.block_types

    def has_status(self, name: str) -> bool:
        return str(name).casefold() in self.statuses
