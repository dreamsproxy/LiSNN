"""Robust zoom and pan behavior for the planning canvas.

The original view rejected a wheel step whenever the *next* scale remained below
0.15. Large documents fitted below that threshold therefore became trapped: the
user could neither zoom inward nor create scrollbar travel for panning.

This module keeps navigation independent from the selected rendering backend and
installs the behavior before the first PlanningView is instantiated.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtWidgets import QGraphicsView


MIN_ZOOM = 0.02
MAX_ZOOM = 8.0
ZOOM_STEP = 1.15

_INSTALLED = False


def clamp_zoom(value: float) -> float:
    """Clamp an absolute view scale to the supported navigation range."""

    return min(MAX_ZOOM, max(MIN_ZOOM, float(value)))


def install_navigation_patches() -> None:
    """Install backend-independent zoom and middle-button panning once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from .scene import PlanningView

    original_init = PlanningView.__init__
    original_fit_graph = PlanningView.fit_graph

    def view_init(self, scene) -> None:
        original_init(self, scene)
        self._panning = False
        self._pan_start = QPoint()
        self._pan_previous_drag_mode = self.dragMode()
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.viewport().setMouseTracking(True)

    def current_zoom(self) -> float:
        """Return the current uniform view scale."""

        return max(1.0e-12, abs(float(self.transform().m11())))

    def zoom_to(self, target: float, anchor: QPoint | None = None) -> float:
        """Set an absolute scale while preserving the scene point under anchor."""

        current = current_zoom(self)
        target = clamp_zoom(target)
        if math.isclose(current, target, rel_tol=1.0e-12, abs_tol=1.0e-12):
            return current

        anchor = anchor or self.viewport().rect().center()
        scene_before = self.mapToScene(anchor)
        previous_anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        try:
            ratio = target / current
            self.scale(ratio, ratio)
            scene_after = self.mapToScene(anchor)
            correction = scene_after - scene_before
            self.translate(correction.x(), correction.y())
        finally:
            self.setTransformationAnchor(previous_anchor)

        self.viewport().update()
        return current_zoom(self)

    def zoom_by(self, factor: float, anchor: QPoint | None = None) -> float:
        """Apply a multiplicative zoom step without creating a dead zone."""

        if factor <= 0.0:
            raise ValueError("zoom factor must be positive")
        return zoom_to(self, current_zoom(self) * float(factor), anchor)

    def wheel_event(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta == 0:
            event.ignore()
            return

        # Standard mouse wheels report 120 units per notch. High-resolution
        # trackpads may report smaller values, which remain proportional here.
        steps = float(delta) / 120.0
        factor = ZOOM_STEP ** steps
        zoom_by(self, factor, event.position().toPoint())
        event.accept()

    def pan_by_pixels(self, delta: QPoint) -> None:
        """Move the visible scene by a viewport-pixel drag delta."""

        if delta.isNull():
            return
        scale = current_zoom(self)
        center = self.mapToScene(self.viewport().rect().center())
        self.centerOn(
            center
            - QPointF(
                float(delta.x()) / scale,
                float(delta.y()) / scale,
            )
        )
        self.viewport().update()

    def mouse_press_event(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self._pan_previous_drag_mode = self.dragMode()
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            event.accept()
            return
        QGraphicsView.mousePressEvent(self, event)

    def mouse_move_event(self, event) -> None:
        if self._panning:
            position = event.position().toPoint()
            delta = position - self._pan_start
            self._pan_start = position
            pan_by_pixels(self, delta)
            event.accept()
            return
        QGraphicsView.mouseMoveEvent(self, event)

    def mouse_release_event(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.setDragMode(self._pan_previous_drag_mode)
            self.viewport().unsetCursor()
            event.accept()
            return
        QGraphicsView.mouseReleaseEvent(self, event)

    def fit_graph(self) -> None:
        # fitInView may legitimately produce a scale below the historical 0.15
        # threshold. The new zoom methods can always recover from that scale.
        original_fit_graph(self)
        self.viewport().update()

    PlanningView.__init__ = view_init
    PlanningView.current_zoom = current_zoom
    PlanningView.zoom_to = zoom_to
    PlanningView.zoom_by = zoom_by
    PlanningView.wheelEvent = wheel_event
    PlanningView.pan_by_pixels = pan_by_pixels
    PlanningView.mousePressEvent = mouse_press_event
    PlanningView.mouseMoveEvent = mouse_move_event
    PlanningView.mouseReleaseEvent = mouse_release_event
    PlanningView.fit_graph = fit_graph
    _INSTALLED = True
