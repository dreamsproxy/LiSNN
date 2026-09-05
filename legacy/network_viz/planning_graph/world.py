"""Dynamically expanding scene world for the planning canvas.

QGraphicsScene requires a finite sceneRect for scroll-bar mapping, but that rect
does not need to be a fixed project boundary. This module expands it in chunks
whenever the viewport or graph content approaches an edge. The result behaves
like an effectively unbounded diagramming canvas without assigning an enormous
initial rectangle that would reduce navigation precision.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPoint, QPointF, QRectF
from PySide6.QtWidgets import QGraphicsView


WORLD_CHUNK = 10_000.0
WORLD_MARGIN = 2_500.0
INITIAL_WORLD_HALF_EXTENT = 10_000.0

_INSTALLED = False


def _rounded_outward(rect: QRectF, chunk: float = WORLD_CHUNK) -> QRectF:
    """Round all sides outward to stable expansion chunks."""

    left = math.floor(rect.left() / chunk) * chunk
    top = math.floor(rect.top() / chunk) * chunk
    right = math.ceil(rect.right() / chunk) * chunk
    bottom = math.ceil(rect.bottom() / chunk) * chunk
    if right <= left:
        right = left + chunk
    if bottom <= top:
        bottom = top + chunk
    return QRectF(left, top, right - left, bottom - top)


def expanded_world_rect(
    current: QRectF,
    required: QRectF,
    *,
    margin: float = WORLD_MARGIN,
) -> QRectF:
    """Return ``current`` or an outward-rounded union containing ``required``."""

    padded = required.normalized().adjusted(-margin, -margin, margin, margin)
    if current.isValid() and not current.isEmpty() and current.contains(padded):
        return QRectF(current)
    if current.isValid() and not current.isEmpty():
        padded = current.united(padded)
    return _rounded_outward(padded)


def install_world_patches() -> None:
    """Install an effectively unbounded, dynamically expanding scene world."""

    global _INSTALLED
    if _INSTALLED:
        return

    from .scene import PlanningScene, PlanningView

    original_scene_init = PlanningScene.__init__
    original_load_document = PlanningScene.load_document
    original_create_node = PlanningScene.create_node
    original_create_edge = PlanningScene.create_edge
    original_add_group_item = PlanningScene._add_group_item
    original_node_geometry_live = PlanningScene.node_geometry_live
    original_group_geometry_live = PlanningScene.group_geometry_live
    original_edge_geometry_live = PlanningScene.edge_geometry_live

    original_view_init = PlanningView.__init__
    original_zoom_to = PlanningView.zoom_to
    original_fit_graph = PlanningView.fit_graph
    original_resize_event = PlanningView.resizeEvent

    def initial_world_rect() -> QRectF:
        half = INITIAL_WORLD_HALF_EXTENT
        return QRectF(-half, -half, half * 2.0, half * 2.0)

    def reset_world_to_content(self) -> QRectF:
        target = initial_world_rect()
        content = self.itemsBoundingRect()
        if content.isValid() and not content.isEmpty():
            target = expanded_world_rect(target, content)
        self.setSceneRect(target)
        return QRectF(target)

    def ensure_world_contains(
        self,
        required: QRectF,
        *,
        margin: float = WORLD_MARGIN,
    ) -> bool:
        current = self.sceneRect()
        expanded = expanded_world_rect(current, required, margin=margin)
        if expanded == current:
            return False
        self.setSceneRect(expanded)
        return True

    def ensure_item_world(self, item) -> None:
        if item is not None:
            ensure_world_contains(self, item.sceneBoundingRect())

    def scene_init(self, *args, **kwargs) -> None:
        original_scene_init(self, *args, **kwargs)
        reset_world_to_content(self)

    def load_document(self, document, *, clear_history: bool = False) -> None:
        original_load_document(self, document, clear_history=clear_history)
        reset_world_to_content(self)

    def create_node(self, *args, **kwargs):
        node = original_create_node(self, *args, **kwargs)
        ensure_item_world(self, self.node_items.get(node.node_id))
        return node

    def create_edge(self, *args, **kwargs):
        edge = original_create_edge(self, *args, **kwargs)
        ensure_item_world(self, self.edge_items.get(edge.edge_id))
        return edge

    def add_group_item(self, group):
        item = original_add_group_item(self, group)
        ensure_item_world(self, item)
        return item

    def node_geometry_live(self, node_id: str) -> None:
        original_node_geometry_live(self, node_id)
        ensure_item_world(self, self.node_items.get(node_id))
        for edge_item in self.edge_items.values():
            if node_id in (edge_item.model.source_id, edge_item.model.target_id):
                ensure_item_world(self, edge_item)

    def group_geometry_live(self, group_id: str) -> None:
        original_group_geometry_live(self, group_id)
        ensure_item_world(self, self.group_items.get(group_id))

    def edge_geometry_live(self, edge_id: str) -> None:
        original_edge_geometry_live(self, edge_id)
        ensure_item_world(self, self.edge_items.get(edge_id))

    def view_init(self, scene) -> None:
        original_view_init(self, scene)
        self._world_checking = False
        self.horizontalScrollBar().valueChanged.connect(
            lambda _value: self.ensure_visible_world()
        )
        self.verticalScrollBar().valueChanged.connect(
            lambda _value: self.ensure_visible_world()
        )
        self.ensure_visible_world()

    def visible_rect_for_center(self, center: QPointF) -> QRectF:
        scale = max(1.0e-12, float(self.current_zoom()))
        width = max(1.0, float(self.viewport().width()) / scale)
        height = max(1.0, float(self.viewport().height()) / scale)
        return QRectF(
            center.x() - width / 2.0,
            center.y() - height / 2.0,
            width,
            height,
        )

    def ensure_visible_world(self, center: QPointF | None = None) -> None:
        if getattr(self, "_world_checking", False):
            return
        scene = self.scene()
        if scene is None or not hasattr(scene, "ensure_world_contains"):
            return
        self._world_checking = True
        try:
            center = center or self.mapToScene(self.viewport().rect().center())
            visible = visible_rect_for_center(self, center)
            # Two viewport spans of breathing room let a drag continue smoothly
            # before another expansion is needed.
            margin = max(WORLD_MARGIN, visible.width(), visible.height())
            scene.ensure_world_contains(visible, margin=margin)
        finally:
            self._world_checking = False

    def pan_by_pixels(self, delta: QPoint) -> None:
        if delta.isNull():
            return
        scale = max(1.0e-12, float(self.current_zoom()))
        center = self.mapToScene(self.viewport().rect().center())
        target = center - QPointF(
            float(delta.x()) / scale,
            float(delta.y()) / scale,
        )
        self.ensure_visible_world(target)
        self.centerOn(target)
        self.viewport().update()

    def zoom_to(self, target: float, anchor: QPoint | None = None) -> float:
        result = original_zoom_to(self, target, anchor)
        self.ensure_visible_world()
        return result

    def fit_graph(self) -> None:
        original_fit_graph(self)
        self.ensure_visible_world()

    def resize_event(self, event) -> None:
        original_resize_event(self, event)
        self.ensure_visible_world()

    PlanningScene.__init__ = scene_init
    PlanningScene.load_document = load_document
    PlanningScene.create_node = create_node
    PlanningScene.create_edge = create_edge
    PlanningScene._add_group_item = add_group_item
    PlanningScene.node_geometry_live = node_geometry_live
    PlanningScene.group_geometry_live = group_geometry_live
    PlanningScene.edge_geometry_live = edge_geometry_live
    PlanningScene.reset_world_to_content = reset_world_to_content
    PlanningScene.ensure_world_contains = ensure_world_contains

    PlanningView.__init__ = view_init
    PlanningView.visible_rect_for_center = visible_rect_for_center
    PlanningView.ensure_visible_world = ensure_visible_world
    PlanningView.pan_by_pixels = pan_by_pixels
    PlanningView.zoom_to = zoom_to
    PlanningView.fit_graph = fit_graph
    PlanningView.resizeEvent = resize_event

    _INSTALLED = True
