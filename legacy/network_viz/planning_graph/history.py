"""Undo/redo commands for complete planning document snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtGui import QUndoCommand

if TYPE_CHECKING:
    from .scene import PlanningScene


class DocumentStateCommand(QUndoCommand):
    """Restore a scene between two validated document snapshots.

    Snapshot commands keep every editing path reversible without duplicating
    mutation logic across blocks, edges, groups, inspectors, and serializers.
    Merge keys collapse continuous property typing into one undo operation.
    """

    def __init__(
        self,
        scene: "PlanningScene",
        before: dict[str, Any],
        after: dict[str, Any],
        text: str,
        *,
        merge_key: str | None = None,
        already_applied: bool = True,
    ) -> None:
        super().__init__(text)
        self.scene = scene
        self.before = before
        self.after = after
        self.merge_key = merge_key
        self._first_redo = already_applied

    def id(self) -> int:
        return 913 if self.merge_key is not None else -1

    def mergeWith(self, other: QUndoCommand) -> bool:
        if not isinstance(other, DocumentStateCommand):
            return False
        if self.scene is not other.scene or self.merge_key != other.merge_key:
            return False
        self.after = other.after
        self.setText(other.text())
        return True

    def undo(self) -> None:
        self.scene.restore_document_state(self.before)

    def redo(self) -> None:
        if self._first_redo:
            self._first_redo = False
            return
        self.scene.restore_document_state(self.after)
