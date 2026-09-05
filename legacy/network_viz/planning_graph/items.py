"""Graphics items for blocks, groups, and routed connections."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

from .definitions import DefinitionRegistry
from .grid import (
    GROUP_MIN_HEIGHT,
    GROUP_MIN_WIDTH,
    NODE_MIN_HEIGHT,
    NODE_MIN_WIDTH,
    snap_dimension,
    snap_value,
    snap_xy,
)
from .models import PlanningEdge, PlanningGroup, PlanningNode
from .theme import (
    EDGE_COLORS,
    MUTED_TEXT,
    NODE_BODY,
    NODE_BORDER,
    NODE_HEADER,
    SELECTION,
    TEXT,
)

if TYPE_CHECKING:
    from .scene import PlanningScene


HEADER_MIN_HEIGHT = 34.0
STATUS_BAR_HEIGHT = 6.0
PORT_RADIUS = 5.0
PORT_HIT_RADIUS = 13.0
RESIZE_HANDLE_SIZE = 12.0
GROUP_LABEL_HEIGHT = 26.0
GROUP_Z_BASE = -1000.0
GROUP_Z_STEP = 0.1
EDGE_ROUTE_HANDLE_RADIUS = 7.0


def _font(size: int, *, bold: bool = False) -> QFont:
    font = QFont()
    font.setPointSize(int(size))
    font.setBold(bold)
    return font


def _corner_handles(rect: QRectF) -> dict[str, QRectF]:
    half = RESIZE_HANDLE_SIZE / 2.0
    return {
        "top_left": QRectF(
            rect.left() - half,
            rect.top() - half,
            RESIZE_HANDLE_SIZE,
            RESIZE_HANDLE_SIZE,
        ),
        "top_right": QRectF(
            rect.right() - half,
            rect.top() - half,
            RESIZE_HANDLE_SIZE,
            RESIZE_HANDLE_SIZE,
        ),
        "bottom_left": QRectF(
            rect.left() - half,
            rect.bottom() - half,
            RESIZE_HANDLE_SIZE,
            RESIZE_HANDLE_SIZE,
        ),
        "bottom_right": QRectF(
            rect.right() - half,
            rect.bottom() - half,
            RESIZE_HANDLE_SIZE,
            RESIZE_HANDLE_SIZE,
        ),
    }


def _handle_at(handles: dict[str, QRectF], point: QPointF) -> str | None:
    return next((name for name, rect in handles.items() if rect.contains(point)), None)


def _set_resize_cursor(item: QGraphicsObject, handle: str | None) -> None:
    if handle in ("top_left", "bottom_right"):
        item.setCursor(Qt.CursorShape.SizeFDiagCursor)
    elif handle in ("top_right", "bottom_left"):
        item.setCursor(Qt.CursorShape.SizeBDiagCursor)
    else:
        item.unsetCursor()


class GraphNodeItem(QGraphicsObject):
    """Resizable, grid-aligned planning block."""

    def __init__(self, model: PlanningNode, registry: DefinitionRegistry) -> None:
        super().__init__()
        self.model = model
        self.registry = registry
        self.hovered = False
        self._resizing = False
        self._resize_handle: str | None = None
        self._resize_origin: tuple[float, float, float, float] | None = None
        self._gesture_before: dict[str, Any] | None = None
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.model.x, self.model.y = snap_xy(model.x, model.y)
        self.model.width = snap_dimension(model.width, NODE_MIN_WIDTH)
        self.model.height = snap_dimension(model.height, NODE_MIN_HEIGHT)
        self.setPos(self.model.x, self.model.y)

    def content_rect(self) -> QRectF:
        return QRectF(
            -self.model.width / 2.0,
            -self.model.height / 2.0,
            self.model.width,
            self.model.height,
        )

    def boundingRect(self) -> QRectF:
        padding = max(RESIZE_HANDLE_SIZE, PORT_HIT_RADIUS) / 2.0 + 3.0
        return self.content_rect().adjusted(-padding, -padding, padding, padding)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.content_rect()
        selected = self.isSelected()
        outline = QColor(SELECTION if selected else NODE_BORDER)
        accent = QColor(self.registry.block_color(self.model.kind))
        status_color = QColor(self.registry.status_color(self.model.status, NODE_HEADER))
        header_height = max(HEADER_MIN_HEIGHT, float(self.model.header_font_size + 20))

        painter.setPen(QPen(outline, 2.5 if selected else 1.4))
        painter.setBrush(QBrush(QColor(NODE_BODY)))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        header = QRectF(rect.left(), rect.top(), rect.width(), header_height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(NODE_HEADER)))
        painter.drawRoundedRect(header, 10.0, 10.0)
        painter.drawRect(QRectF(header.left(), header.bottom() - 10.0, header.width(), 10.0))

        painter.setBrush(QBrush(accent))
        painter.drawRoundedRect(QRectF(rect.left(), rect.top(), 6.0, rect.height()), 3.0, 3.0)
        painter.setBrush(QBrush(status_color))
        painter.drawRoundedRect(
            QRectF(rect.left(), rect.bottom() - STATUS_BAR_HEIGHT, rect.width(), STATUS_BAR_HEIGHT),
            3.0,
            3.0,
        )

        painter.setPen(QPen(QColor(TEXT), 1.0))
        painter.setFont(_font(self.model.header_font_size, bold=True))
        painter.drawText(
            QRectF(header.left() + 14.0, header.top(), header.width() * 0.56, header.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self.model.kind.upper(),
        )
        painter.drawText(
            QRectF(
                header.left() + header.width() * 0.56,
                header.top(),
                header.width() * 0.39,
                header.height(),
            ),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            self.model.status,
        )

        title_top = header.bottom() + 8.0
        title_height = max(38.0, float(self.model.title_font_size * 3.0))
        painter.setFont(_font(self.model.title_font_size, bold=True))
        painter.drawText(
            QRectF(rect.left() + 14.0, title_top, rect.width() - 28.0, title_height),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            self.model.title,
        )

        footer_height = max(18.0, float(self.model.footer_font_size + 9))
        body_top = title_top + title_height + 4.0
        body_bottom = rect.bottom() - footer_height - STATUS_BAR_HEIGHT - 5.0
        painter.setFont(_font(self.model.body_font_size))
        painter.setPen(QPen(QColor(MUTED_TEXT), 1.0))
        painter.drawText(
            QRectF(
                rect.left() + 14.0,
                body_top,
                rect.width() - 28.0,
                max(0.0, body_bottom - body_top),
            ),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            self.model.body,
        )

        footer: list[str] = []
        if self.model.priority:
            footer.append(f"P{self.model.priority}")
        if self.model.tags:
            footer.append("#" + " #".join(self.model.tags[:4]))
        painter.setFont(_font(self.model.footer_font_size))
        painter.drawText(
            QRectF(
                rect.left() + 14.0,
                rect.bottom() - footer_height - STATUS_BAR_HEIGHT,
                rect.width() - 28.0,
                footer_height,
            ),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "  ".join(footer),
        )

        if self.hovered or selected:
            painter.setPen(QPen(QColor(SELECTION if selected else TEXT), 1.0))
            painter.setBrush(QBrush(QColor(NODE_BODY)))
            for point in self.port_positions().values():
                painter.drawEllipse(point, PORT_RADIUS, PORT_RADIUS)
        if selected:
            painter.setPen(QPen(QColor(SELECTION), 1.0))
            for handle in _corner_handles(rect).values():
                painter.drawRect(handle)

    def port_positions(self) -> dict[str, QPointF]:
        rect = self.content_rect()
        return {
            "top": QPointF(0.0, rect.top()),
            "right": QPointF(rect.right(), 0.0),
            "bottom": QPointF(0.0, rect.bottom()),
            "left": QPointF(rect.left(), 0.0),
        }

    def best_port_name(self, toward: QPointF) -> str:
        return min(
            self.port_positions(),
            key=lambda name: (
                self.mapToScene(self.port_positions()[name]).x() - toward.x()
            ) ** 2
            + (
                self.mapToScene(self.port_positions()[name]).y() - toward.y()
            ) ** 2,
        )

    def best_port_scene(self, toward: QPointF) -> QPointF:
        return self.mapToScene(self.port_positions()[self.best_port_name(toward)])

    def port_at(self, point: QPointF) -> str | None:
        for name, port in self.port_positions().items():
            if math.hypot(point.x() - port.x(), point.y() - port.y()) <= PORT_HIT_RADIUS:
                return name
        return None

    def hoverEnterEvent(self, event) -> None:
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event) -> None:
        handle = _handle_at(_corner_handles(self.content_rect()), event.pos()) if self.isSelected() else None
        _set_resize_cursor(self, handle)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.hovered = False
        self.unsetCursor()
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        scene = self.scene()
        if event.button() == Qt.MouseButton.LeftButton:
            handle = _handle_at(_corner_handles(self.content_rect()), event.pos()) if self.isSelected() else None
            if handle is not None:
                self._gesture_before = scene.snapshot() if hasattr(scene, "snapshot") else None
                self._resizing = True
                self._resize_handle = handle
                rect = self.content_rect()
                top_left = self.mapToScene(rect.topLeft())
                bottom_right = self.mapToScene(rect.bottomRight())
                self._resize_origin = (
                    top_left.x(),
                    top_left.y(),
                    bottom_right.x(),
                    bottom_right.y(),
                )
                event.accept()
                return
            port = self.port_at(event.pos())
            if port is not None and hasattr(scene, "start_connection"):
                scene.start_connection(self, self.mapToScene(self.port_positions()[port]))
                event.accept()
                return
            self._gesture_before = scene.snapshot() if hasattr(scene, "snapshot") else None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        scene = self.scene()
        if hasattr(scene, "connection_active") and scene.connection_active:
            scene.update_connection(event.scenePos())
            event.accept()
            return
        if self._resizing and self._resize_origin and self._resize_handle:
            left, top, right, bottom = self._resize_origin
            x = snap_value(event.scenePos().x())
            y = snap_value(event.scenePos().y())
            if "left" in self._resize_handle:
                left = min(x, right - NODE_MIN_WIDTH)
            else:
                right = max(x, left + NODE_MIN_WIDTH)
            if "top" in self._resize_handle:
                top = min(y, bottom - NODE_MIN_HEIGHT)
            else:
                bottom = max(y, top + NODE_MIN_HEIGHT)
            self.set_scene_rect(left, top, right - left, bottom - top)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        scene = self.scene()
        if hasattr(scene, "connection_active") and scene.connection_active:
            scene.finish_connection(event.scenePos())
            event.accept()
            return
        was_resizing = self._resizing
        if was_resizing:
            self._resizing = False
            self._resize_handle = None
            self._resize_origin = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)
        if self._gesture_before is not None and hasattr(scene, "commit_external_change"):
            scene.commit_external_change(
                "Resize block" if was_resizing else "Move block",
                self._gesture_before,
            )
        self._gesture_before = None

    def mouseDoubleClickEvent(self, event) -> None:
        scene = self.scene()
        if hasattr(scene, "request_edit_node"):
            scene.request_edit_node(self.model.node_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def set_scene_rect(self, x: float, y: float, width: float, height: float) -> None:
        width = snap_dimension(width, NODE_MIN_WIDTH)
        height = snap_dimension(height, NODE_MIN_HEIGHT)
        left = snap_value(x)
        top = snap_value(y)
        center_x = snap_value(left + width / 2.0)
        center_y = snap_value(top + height / 2.0)
        self.prepareGeometryChange()
        self.model.width = width
        self.model.height = height
        self.setPos(center_x, center_y)
        self.model.x = center_x
        self.model.y = center_y
        self.update()
        scene = self.scene()
        if hasattr(scene, "node_geometry_live"):
            scene.node_geometry_live(self.model.node_id)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            value = QPointF(snap_value(value.x()), snap_value(value.y()))
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.model.x = float(self.pos().x())
            self.model.y = float(self.pos().y())
            scene = self.scene()
            if hasattr(scene, "node_geometry_live"):
                scene.node_geometry_live(self.model.node_id)
        return result


class GraphGroupItem(QGraphicsObject):
    """Movable and resizable visual group frame."""

    def __init__(self, model: PlanningGroup) -> None:
        super().__init__()
        self.model = model
        self._resizing = False
        self._resize_handle: str | None = None
        self._resize_origin: tuple[float, float, float, float] | None = None
        self._gesture_before: dict[str, Any] | None = None
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setPos(snap_value(model.x), snap_value(model.y))
        self.model.x = float(self.pos().x())
        self.model.y = float(self.pos().y())
        self.model.width = snap_dimension(model.width, GROUP_MIN_WIDTH)
        self.model.height = snap_dimension(model.height, GROUP_MIN_HEIGHT)
        self.refresh_layer()

    def content_rect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.model.width, self.model.height)

    def boundingRect(self) -> QRectF:
        padding = RESIZE_HANDLE_SIZE / 2.0 + 3.0
        return self.content_rect().adjusted(-padding, -padding, padding, padding)

    def refresh_layer(self) -> None:
        self.setZValue(GROUP_Z_BASE + float(self.model.layer) * GROUP_Z_STEP)
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.content_rect()
        color = QColor(self.model.color)
        fill = QColor(color)
        fill.setAlpha(58 if self.model.backdrop else 0)
        outline = QColor(SELECTION if self.isSelected() else color)
        outline.setAlpha(255 if self.isSelected() else (205 if self.model.backdrop else 130))
        pen = QPen(outline, 2.2 if self.isSelected() else 1.5)
        if not self.model.backdrop:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(fill))
        painter.drawRoundedRect(rect, 12.0, 12.0)

        painter.setFont(_font(9, bold=True))
        width = min(
            max(110.0, float(painter.fontMetrics().horizontalAdvance(self.model.title) + 24)),
            max(110.0, self.model.width - 16.0),
        )
        label_rect = QRectF(8.0, 8.0, width, GROUP_LABEL_HEIGHT)
        label_fill = QColor(color)
        label_fill.setAlpha(225)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(label_fill))
        painter.drawRoundedRect(label_rect, 6.0, 6.0)
        painter.setPen(QPen(QColor(TEXT), 1.0))
        painter.drawText(
            label_rect.adjusted(9.0, 0.0, -9.0, 0.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self.model.title,
        )
        if self.isSelected():
            painter.setPen(QPen(QColor(SELECTION), 1.0))
            painter.setBrush(QBrush(QColor(NODE_BODY)))
            for handle in _corner_handles(rect).values():
                painter.drawRect(handle)

    def hoverMoveEvent(self, event) -> None:
        handle = _handle_at(_corner_handles(self.content_rect()), event.pos()) if self.isSelected() else None
        _set_resize_cursor(self, handle)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        scene = self.scene()
        if event.button() == Qt.MouseButton.LeftButton:
            self._gesture_before = scene.snapshot() if hasattr(scene, "snapshot") else None
            handle = _handle_at(_corner_handles(self.content_rect()), event.pos()) if self.isSelected() else None
            if handle is not None:
                self._resizing = True
                self._resize_handle = handle
                rect = self.content_rect()
                top_left = self.mapToScene(rect.topLeft())
                bottom_right = self.mapToScene(rect.bottomRight())
                self._resize_origin = (
                    top_left.x(),
                    top_left.y(),
                    bottom_right.x(),
                    bottom_right.y(),
                )
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._resizing and self._resize_origin and self._resize_handle:
            left, top, right, bottom = self._resize_origin
            x = snap_value(event.scenePos().x())
            y = snap_value(event.scenePos().y())
            if "left" in self._resize_handle:
                left = min(x, right - GROUP_MIN_WIDTH)
            else:
                right = max(x, left + GROUP_MIN_WIDTH)
            if "top" in self._resize_handle:
                top = min(y, bottom - GROUP_MIN_HEIGHT)
            else:
                bottom = max(y, top + GROUP_MIN_HEIGHT)
            self.set_scene_rect(left, top, right - left, bottom - top)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        was_resizing = self._resizing
        if was_resizing:
            self._resizing = False
            self._resize_handle = None
            self._resize_origin = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)
        scene = self.scene()
        if self._gesture_before is not None and hasattr(scene, "commit_external_change"):
            scene.commit_external_change(
                "Resize group" if was_resizing else "Move group",
                self._gesture_before,
            )
        self._gesture_before = None

    def mouseDoubleClickEvent(self, event) -> None:
        scene = self.scene()
        if hasattr(scene, "request_edit_group"):
            scene.request_edit_group(self.model.group_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def set_scene_rect(self, x: float, y: float, width: float, height: float) -> None:
        x = snap_value(x)
        y = snap_value(y)
        width = snap_dimension(width, GROUP_MIN_WIDTH)
        height = snap_dimension(height, GROUP_MIN_HEIGHT)
        self.prepareGeometryChange()
        resizing = self._resizing
        self._resizing = True
        self.setPos(x, y)
        self.model.x = x
        self.model.y = y
        self.model.width = width
        self.model.height = height
        self._resizing = resizing
        self.update()
        scene = self.scene()
        if hasattr(scene, "group_geometry_live"):
            scene.group_geometry_live(self.model.group_id)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            value = QPointF(snap_value(value.x()), snap_value(value.y()))
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            x = float(self.pos().x())
            y = float(self.pos().y())
            dx = x - self.model.x
            dy = y - self.model.y
            self.model.x = x
            self.model.y = y
            scene = self.scene()
            if not self._resizing and (dx or dy) and hasattr(scene, "move_group_nodes"):
                scene.move_group_nodes(self.model.group_id, dx, dy)
            if hasattr(scene, "group_geometry_live"):
                scene.group_geometry_live(self.model.group_id)
        return result


class GraphEdgeItem(QGraphicsObject):
    """Directed connection with automatic lanes and a draggable route handle."""

    def __init__(
        self,
        model: PlanningEdge,
        source: GraphNodeItem,
        target: GraphNodeItem,
    ) -> None:
        super().__init__()
        self.model = model
        self.source = source
        self.target = target
        self.path = QPainterPath()
        self.arrow = QPolygonF()
        self.label_point = QPointF()
        self.route_point = QPointF()
        self.lane_offset = 0.0
        self._dragging_route = False
        self._route_moved = False
        self._gesture_before: dict[str, Any] | None = None
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(-1.0)
        self.update_geometry()

    def set_lane_offset(self, value: float) -> None:
        self.lane_offset = float(value)
        if not self.model.has_manual_route:
            self.update_geometry()

    def update_geometry(self) -> None:
        self.prepareGeometryChange()
        start = self.source.best_port_scene(self.target.scenePos())
        end = self.target.best_port_scene(self.source.scenePos())
        if self.model.has_manual_route:
            route = QPointF(float(self.model.route_x), float(self.model.route_y))
        else:
            midpoint = QPointF((start.x() + end.x()) / 2.0, (start.y() + end.y()) / 2.0)
            dx = end.x() - start.x()
            dy = end.y() - start.y()
            length = max(1.0, math.hypot(dx, dy))
            route = QPointF(
                midpoint.x() + (-dy / length) * self.lane_offset,
                midpoint.y() + (dx / length) * self.lane_offset,
            )
        self.route_point = route
        path = QPainterPath(start)
        path.quadTo(route, end)
        self.path = path
        self.label_point = path.pointAtPercent(0.5)

        tangent = path.pointAtPercent(0.965)
        angle = math.atan2(end.y() - tangent.y(), end.x() - tangent.x())
        arrow_size = 13.0 if self.model.relation == "Leads To" else 10.0
        self.arrow = QPolygonF(
            [
                end,
                QPointF(
                    end.x() - arrow_size * math.cos(angle - math.pi / 6),
                    end.y() - arrow_size * math.sin(angle - math.pi / 6),
                ),
                QPointF(
                    end.x() - arrow_size * math.cos(angle + math.pi / 6),
                    end.y() - arrow_size * math.sin(angle + math.pi / 6),
                ),
            ]
        )
        self.update()

    def boundingRect(self) -> QRectF:
        handle_rect = QRectF(
            self.route_point.x() - EDGE_ROUTE_HANDLE_RADIUS,
            self.route_point.y() - EDGE_ROUTE_HANDLE_RADIUS,
            EDGE_ROUTE_HANDLE_RADIUS * 2,
            EDGE_ROUTE_HANDLE_RADIUS * 2,
        )
        return self.path.boundingRect().united(handle_rect).adjusted(-28.0, -28.0, 28.0, 28.0)

    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(16.0)
        return stroker.createStroke(self.path)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(EDGE_COLORS.get(self.model.relation, MUTED_TEXT))
        painter.setPen(QPen(color, 3.0 if self.isSelected() else 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self.path)
        painter.setBrush(QBrush(color))
        painter.drawPolygon(self.arrow)

        label = self.model.label.strip() or self.model.relation
        painter.setFont(_font(8, bold=self.isSelected()))
        label_rect = painter.fontMetrics().boundingRect(label).adjusted(-6, -4, 6, 4)
        label_rect.moveCenter(self.label_point.toPoint())
        painter.setPen(Qt.PenStyle.NoPen)
        fill = QColor(NODE_BODY)
        fill.setAlpha(235)
        painter.setBrush(QBrush(fill))
        painter.drawRoundedRect(QRectF(label_rect), 4.0, 4.0)
        painter.setPen(QPen(color, 1.0))
        painter.drawText(QRectF(label_rect), Qt.AlignmentFlag.AlignCenter, label)

        if self.isSelected():
            painter.setPen(QPen(QColor(SELECTION), 2.0))
            painter.setBrush(QBrush(QColor(NODE_BODY)))
            painter.drawEllipse(
                self.route_point,
                EDGE_ROUTE_HANDLE_RADIUS,
                EDGE_ROUTE_HANDLE_RADIUS,
            )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSelected(True)
            scene = self.scene()
            self._gesture_before = scene.snapshot() if hasattr(scene, "snapshot") else None
            self._dragging_route = True
            self._route_moved = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging_route:
            self._route_moved = True
            self.model.route_x = snap_value(event.scenePos().x())
            self.model.route_y = snap_value(event.scenePos().y())
            self.update_geometry()
            scene = self.scene()
            if hasattr(scene, "edge_geometry_live"):
                scene.edge_geometry_live(self.model.edge_id)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging_route:
            self._dragging_route = False
            scene = self.scene()
            if self._route_moved and self._gesture_before is not None and hasattr(scene, "commit_external_change"):
                scene.commit_external_change("Route connection", self._gesture_before)
            self._gesture_before = None
            self._route_moved = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        scene = self.scene()
        if hasattr(scene, "request_edit_edge"):
            scene.request_edit_edge(self.model.edge_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)
