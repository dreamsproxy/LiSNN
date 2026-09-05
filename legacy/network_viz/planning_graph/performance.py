"""Adaptive rendering and OpenGL acceleration for the planning graph.

The graph document and editing model remain independent from this module. It
patches the existing QGraphicsView classes at application startup so large
planning documents can use progressive level-of-detail rendering without
changing their serialized representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPolygonF,
    QSurfaceFormat,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QStyleOptionGraphicsItem,
)

from .grid import GRID_SIZE
from .theme import GRID_DOT, MUTED_TEXT, NODE_BODY, NODE_BORDER, NODE_HEADER, SELECTION, TEXT


@dataclass(frozen=True)
class RenderProfile:
    """Centralized level-of-detail and grid thresholds."""

    name: str
    full_detail_lod: float
    compact_lod: float
    title_lod: float
    group_label_lod: float
    edge_label_lod: float
    interaction_lod: float
    arrow_lod: float
    grid_close_lod: float
    grid_medium_lod: float
    grid_far_lod: float
    max_grid_points: int


BALANCED_PROFILE = RenderProfile(
    name="Balanced",
    full_detail_lod=0.70,
    compact_lod=0.40,
    title_lod=0.20,
    group_label_lod=0.30,
    edge_label_lod=0.68,
    interaction_lod=0.55,
    arrow_lod=0.20,
    grid_close_lod=0.75,
    grid_medium_lod=0.42,
    grid_far_lod=0.24,
    max_grid_points=18_000,
)

PERFORMANCE_PROFILE = RenderProfile(
    name="Performance",
    full_detail_lod=0.90,
    compact_lod=0.52,
    title_lod=0.27,
    group_label_lod=0.38,
    edge_label_lod=0.88,
    interaction_lod=0.65,
    arrow_lod=0.27,
    grid_close_lod=0.92,
    grid_medium_lod=0.55,
    grid_far_lod=0.30,
    max_grid_points=10_000,
)

_SOFTWARE_VALUES = {"1", "true", "yes", "on", "software", "raster"}
_OPENGL_ENABLED = (
    os.environ.get("LISNN_RENDERER", "opengl").strip().lower()
    not in _SOFTWARE_VALUES
)
_PERFORMANCE_MODE = (
    os.environ.get("LISNN_PERFORMANCE_MODE", "0").strip().lower()
    in _SOFTWARE_VALUES
)
_PATCHED = False


def configure_runtime(
    *,
    software_rendering: bool | None = None,
    performance_mode: bool | None = None,
) -> None:
    """Set renderer preferences before the first PlanningView is created."""

    global _OPENGL_ENABLED, _PERFORMANCE_MODE
    if software_rendering is not None:
        _OPENGL_ENABLED = not bool(software_rendering)
    if performance_mode is not None:
        _PERFORMANCE_MODE = bool(performance_mode)


def active_profile() -> RenderProfile:
    return PERFORMANCE_PROFILE if _PERFORMANCE_MODE else BALANCED_PROFILE


def set_performance_mode(enabled: bool) -> None:
    global _PERFORMANCE_MODE
    _PERFORMANCE_MODE = bool(enabled)


def performance_mode_enabled() -> bool:
    return _PERFORMANCE_MODE


def detail_tier(lod: float, profile: RenderProfile | None = None) -> str:
    profile = profile or active_profile()
    if lod >= profile.full_detail_lod:
        return "full"
    if lod >= profile.compact_lod:
        return "compact"
    if lod >= profile.title_lod:
        return "title"
    return "silhouette"


def grid_step_for_lod(
    lod: float,
    profile: RenderProfile | None = None,
) -> float | None:
    """Return visual grid spacing; snapping always remains GRID_SIZE."""

    profile = profile or active_profile()
    if lod >= profile.grid_close_lod:
        return GRID_SIZE
    if lod >= profile.grid_medium_lod:
        return GRID_SIZE * 4.0
    if lod >= profile.grid_far_lod:
        return GRID_SIZE * 10.0
    return None


def _lod_from_painter(painter: QPainter) -> float:
    try:
        value = QStyleOptionGraphicsItem.levelOfDetailFromTransform(
            painter.worldTransform()
        )
    except (AttributeError, TypeError):
        value = abs(float(painter.worldTransform().m11()))
    return max(0.0001, float(value))


def _font(size: int, *, bold: bool = False) -> QFont:
    font = QFont()
    font.setPointSize(max(1, int(size)))
    font.setBold(bold)
    return font


def _projected_readable(
    scene_units: float,
    lod: float,
    minimum_pixels: float,
) -> bool:
    return scene_units * lod >= minimum_pixels


def _arrow_for_line(start: QPointF, end: QPointF, size: float) -> QPolygonF:
    angle = math.atan2(end.y() - start.y(), end.x() - start.x())
    return QPolygonF(
        [
            end,
            QPointF(
                end.x() - size * math.cos(angle - math.pi / 6.0),
                end.y() - size * math.sin(angle - math.pi / 6.0),
            ),
            QPointF(
                end.x() - size * math.cos(angle + math.pi / 6.0),
                end.y() - size * math.sin(angle + math.pi / 6.0),
            ),
        ]
    )


def _find_menu(window: Any, title: str):
    normalized = title.replace("&", "")
    for action in window.menuBar().actions():
        if action.text().replace("&", "") == normalized:
            return action.menu()
    return None


def install_runtime_patches() -> None:
    """Install OpenGL, adaptive grid, LOD item painting, and UI controls once."""

    global _PATCHED
    if _PATCHED:
        return

    from . import items as item_module
    from .items import GraphEdgeItem, GraphGroupItem, GraphNodeItem
    from .scene import PlanningScene, PlanningView
    from .window import PlanningGraphWindow

    original_view_init = PlanningView.__init__
    original_node_init = GraphNodeItem.__init__
    original_node_paint = GraphNodeItem.paint
    original_window_init = PlanningGraphWindow.__init__

    def view_init(self, scene):
        original_view_init(self, scene)
        self.renderer_backend = "Software Raster"
        self.renderer_error = ""
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate
        )

        platform = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
        allow_opengl = _OPENGL_ENABLED and platform not in {
            "offscreen",
            "minimal",
            "vnc",
        }
        if allow_opengl:
            try:
                from PySide6.QtOpenGLWidgets import QOpenGLWidget

                surface_format = QSurfaceFormat()
                surface_format.setSamples(4)
                surface_format.setSwapBehavior(
                    QSurfaceFormat.SwapBehavior.DoubleBuffer
                )
                viewport = QOpenGLWidget()
                viewport.setFormat(surface_format)
                self.setViewport(viewport)
                self.setViewportUpdateMode(
                    QGraphicsView.ViewportUpdateMode.FullViewportUpdate
                )
                self.renderer_backend = "OpenGL"
            except Exception as exc:
                self.renderer_error = str(exc)
                self.renderer_backend = "Software Raster"

    def view_set_performance_mode(self, enabled: bool) -> None:
        set_performance_mode(enabled)
        scene = self.scene()
        if scene is not None:
            scene.update()
        self.viewport().update()

    def node_init(self, model, registry):
        original_node_init(self, model, registry)
        # DeviceCoordinateCache is rebuilt at every zoom and fights LOD changes.
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)

    def scene_draw_background(
        self,
        painter: QPainter,
        rect: QRectF,
    ) -> None:
        QGraphicsScene.drawBackground(self, painter, rect)
        lod = _lod_from_painter(painter)
        profile = active_profile()
        step = grid_step_for_lod(lod, profile)
        if step is None:
            return

        columns = max(0, int(math.floor(rect.width() / step)) + 2)
        rows = max(0, int(math.floor(rect.height() / step)) + 2)
        while columns * rows > profile.max_grid_points:
            step *= 2.0
            columns = max(0, int(math.floor(rect.width() / step)) + 2)
            rows = max(0, int(math.floor(rect.height() / step)) + 2)

        left = math.floor(rect.left() / step) * step
        top = math.floor(rect.top() / step) * step
        points = QPolygonF()
        x = left
        while x <= rect.right():
            y = top
            while y <= rect.bottom():
                points.append(QPointF(x, y))
                y += step
            x += step

        pen = QPen(QColor(*GRID_DOT))
        pen.setWidthF(2.0 if step <= GRID_SIZE else 1.5)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.save()
        painter.setPen(pen)
        painter.drawPoints(points)
        painter.restore()

    def node_paint(self, painter: QPainter, option, widget=None) -> None:
        lod = _lod_from_painter(painter)
        tier = detail_tier(lod)
        if tier == "full":
            original_node_paint(self, painter, option, widget)
            return

        profile = active_profile()
        rect = self.content_rect()
        selected = self.isSelected()
        outline = QColor(SELECTION if selected else NODE_BORDER)
        accent = QColor(self.registry.block_color(self.model.kind))
        status_color = QColor(
            self.registry.status_color(self.model.status, NODE_HEADER)
        )
        header_height = max(
            item_module.HEADER_MIN_HEIGHT,
            float(self.model.header_font_size + 20),
        )
        header = QRectF(rect.left(), rect.top(), rect.width(), header_height)

        painter.save()
        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing,
            lod >= profile.title_lod,
        )
        painter.setPen(QPen(outline, 2.5 if selected else 1.2))
        painter.setBrush(QBrush(QColor(NODE_BODY)))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(NODE_HEADER)))
        painter.drawRoundedRect(header, 10.0, 10.0)
        painter.drawRect(
            QRectF(
                header.left(),
                header.bottom() - 10.0,
                header.width(),
                10.0,
            )
        )
        painter.setBrush(QBrush(accent))
        painter.drawRoundedRect(
            QRectF(rect.left(), rect.top(), 6.0, rect.height()),
            3.0,
            3.0,
        )
        painter.setBrush(QBrush(status_color))
        painter.drawRoundedRect(
            QRectF(
                rect.left(),
                rect.bottom() - item_module.STATUS_BAR_HEIGHT,
                rect.width(),
                item_module.STATUS_BAR_HEIGHT,
            ),
            3.0,
            3.0,
        )

        if tier == "compact":
            painter.setPen(QPen(QColor(TEXT), 1.0))
            if _projected_readable(self.model.header_font_size, lod, 4.0):
                painter.setFont(_font(self.model.header_font_size, bold=True))
                kind = painter.fontMetrics().elidedText(
                    self.model.kind.upper(),
                    Qt.TextElideMode.ElideRight,
                    max(1, int(header.width() * 0.52)),
                )
                status = painter.fontMetrics().elidedText(
                    self.model.status,
                    Qt.TextElideMode.ElideRight,
                    max(1, int(header.width() * 0.34)),
                )
                painter.drawText(
                    QRectF(
                        header.left() + 14.0,
                        header.top(),
                        header.width() * 0.55,
                        header.height(),
                    ),
                    Qt.AlignmentFlag.AlignVCenter
                    | Qt.AlignmentFlag.AlignLeft,
                    kind,
                )
                painter.drawText(
                    QRectF(
                        header.left() + header.width() * 0.58,
                        header.top(),
                        header.width() * 0.36,
                        header.height(),
                    ),
                    Qt.AlignmentFlag.AlignVCenter
                    | Qt.AlignmentFlag.AlignRight,
                    status,
                )
            if _projected_readable(self.model.title_font_size, lod, 4.5):
                painter.setFont(_font(self.model.title_font_size, bold=True))
                title = painter.fontMetrics().elidedText(
                    self.model.title,
                    Qt.TextElideMode.ElideRight,
                    max(1, int(rect.width() - 28.0)),
                )
                painter.drawText(
                    QRectF(
                        rect.left() + 14.0,
                        header.bottom() + 5.0,
                        rect.width() - 28.0,
                        38.0,
                    ),
                    Qt.AlignmentFlag.AlignLeft
                    | Qt.AlignmentFlag.AlignVCenter,
                    title,
                )
        elif tier == "title" and _projected_readable(
            rect.width(), lod, 45.0
        ):
            painter.setPen(QPen(QColor(TEXT), 1.0))
            painter.setFont(
                _font(max(7, self.model.title_font_size), bold=True)
            )
            title = painter.fontMetrics().elidedText(
                self.model.title,
                Qt.TextElideMode.ElideRight,
                max(1, int(rect.width() - 24.0)),
            )
            painter.drawText(
                rect.adjusted(
                    12.0,
                    header_height,
                    -12.0,
                    -item_module.STATUS_BAR_HEIGHT,
                ),
                Qt.AlignmentFlag.AlignCenter,
                title,
            )

        if lod >= profile.interaction_lod and (self.hovered or selected):
            painter.setPen(
                QPen(QColor(SELECTION if selected else TEXT), 1.0)
            )
            painter.setBrush(QBrush(QColor(NODE_BODY)))
            for point in self.port_positions().values():
                painter.drawEllipse(
                    point,
                    item_module.PORT_RADIUS,
                    item_module.PORT_RADIUS,
                )
        if lod >= profile.interaction_lod and selected:
            painter.setPen(QPen(QColor(SELECTION), 1.0))
            for handle in item_module._corner_handles(rect).values():
                painter.drawRect(handle)
        painter.restore()

    def group_paint(self, painter: QPainter, option, widget=None) -> None:
        lod = _lod_from_painter(painter)
        profile = active_profile()
        rect = self.content_rect()
        color = QColor(self.model.color)
        fill = QColor(color)
        fill.setAlpha(58 if self.model.backdrop else 0)
        outline = QColor(SELECTION if self.isSelected() else color)
        outline.setAlpha(
            255
            if self.isSelected()
            else (205 if self.model.backdrop else 130)
        )
        pen = QPen(outline, 2.2 if self.isSelected() else 1.5)
        if not self.model.backdrop:
            pen.setStyle(Qt.PenStyle.DashLine)

        painter.save()
        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing,
            lod >= profile.title_lod,
        )
        painter.setPen(pen)
        painter.setBrush(QBrush(fill))
        painter.drawRoundedRect(rect, 12.0, 12.0)

        if lod >= profile.group_label_lod and _projected_readable(
            self.model.width,
            lod,
            90.0,
        ):
            painter.setFont(_font(9, bold=True))
            available = max(90, int(self.model.width - 34.0))
            title = painter.fontMetrics().elidedText(
                self.model.title,
                Qt.TextElideMode.ElideRight,
                available,
            )
            width = min(
                max(
                    110.0,
                    float(
                        painter.fontMetrics().horizontalAdvance(title) + 24
                    ),
                ),
                max(110.0, self.model.width - 16.0),
            )
            label_rect = QRectF(
                8.0,
                8.0,
                width,
                item_module.GROUP_LABEL_HEIGHT,
            )
            label_fill = QColor(color)
            label_fill.setAlpha(225)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(label_fill))
            painter.drawRoundedRect(label_rect, 6.0, 6.0)
            painter.setPen(QPen(QColor(TEXT), 1.0))
            painter.drawText(
                label_rect.adjusted(9.0, 0.0, -9.0, 0.0),
                Qt.AlignmentFlag.AlignVCenter
                | Qt.AlignmentFlag.AlignLeft,
                title,
            )

        if self.isSelected() and lod >= profile.interaction_lod:
            painter.setPen(QPen(QColor(SELECTION), 1.0))
            painter.setBrush(QBrush(QColor(NODE_BODY)))
            for handle in item_module._corner_handles(rect).values():
                painter.drawRect(handle)
        painter.restore()

    def edge_paint(self, painter: QPainter, option, widget=None) -> None:
        lod = _lod_from_painter(painter)
        profile = active_profile()
        tier = detail_tier(lod, profile)
        color = QColor(
            item_module.EDGE_COLORS.get(self.model.relation, MUTED_TEXT)
        )
        selected = self.isSelected()
        start = self.path.pointAtPercent(0.0)
        end = self.path.pointAtPercent(1.0)

        painter.save()
        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing,
            lod >= profile.title_lod,
        )
        pen = QPen(color, 3.0 if selected else 2.0)
        if tier == "silhouette":
            pen.setWidthF(2.0 if selected else 1.2)
            pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if tier == "silhouette":
            painter.drawLine(start, end)
        else:
            painter.drawPath(self.path)

        if lod >= profile.arrow_lod:
            arrow = (
                self.arrow
                if tier != "silhouette"
                else _arrow_for_line(
                    start,
                    end,
                    11.0 if self.model.relation == "Leads To" else 8.0,
                )
            )
            painter.setBrush(QBrush(color))
            painter.drawPolygon(arrow)

        label = self.model.label.strip() or self.model.relation
        estimated_label_width = max(18.0, len(label) * 7.0 + 12.0)
        if (
            lod >= profile.edge_label_lod
            and _projected_readable(
                estimated_label_width,
                lod,
                28.0,
            )
        ):
            painter.setFont(_font(8, bold=selected))
            label_rect = painter.fontMetrics().boundingRect(label).adjusted(
                -6,
                -4,
                6,
                4,
            )
            label_rect.moveCenter(self.label_point.toPoint())
            painter.setPen(Qt.PenStyle.NoPen)
            label_fill = QColor(NODE_BODY)
            label_fill.setAlpha(235)
            painter.setBrush(QBrush(label_fill))
            painter.drawRoundedRect(QRectF(label_rect), 4.0, 4.0)
            painter.setPen(QPen(color, 1.0))
            painter.drawText(
                QRectF(label_rect),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )

        if selected and lod >= profile.interaction_lod:
            painter.setPen(QPen(QColor(SELECTION), 2.0))
            painter.setBrush(QBrush(QColor(NODE_BODY)))
            painter.drawEllipse(
                self.route_point,
                item_module.EDGE_ROUTE_HANDLE_RADIUS,
                item_module.EDGE_ROUTE_HANDLE_RADIUS,
            )
        painter.restore()

    def window_init(self, *args, **kwargs):
        original_window_init(self, *args, **kwargs)
        self.performance_action = QAction("Performance Mode", self)
        self.performance_action.setCheckable(True)
        self.performance_action.setShortcut("Ctrl+Shift+P")
        self.performance_action.setChecked(performance_mode_enabled())
        self.performance_action.setToolTip(
            "Use more aggressive level-of-detail thresholds for very large graphs"
        )

        self.renderer_status_label = QLabel()
        self.renderer_status_label.setMinimumWidth(235)
        self.statusBar().addPermanentWidget(self.renderer_status_label)

        def refresh_status() -> None:
            profile = active_profile()
            backend = getattr(
                self.view,
                "renderer_backend",
                "Software Raster",
            )
            self.renderer_status_label.setText(
                f"Renderer: {backend} | LOD: {profile.name}"
            )
            error = getattr(self.view, "renderer_error", "")
            if error:
                self.renderer_status_label.setToolTip(
                    "OpenGL initialization failed; software fallback is active.\n"
                    + error
                )
            else:
                self.renderer_status_label.setToolTip(
                    "Rendering backend and adaptive level-of-detail profile"
                )

        def toggle_performance(enabled: bool) -> None:
            self.view.set_performance_mode(enabled)
            refresh_status()
            self.statusBar().showMessage(
                f"{active_profile().name} rendering profile enabled.",
                3000,
            )

        self.performance_action.toggled.connect(toggle_performance)
        view_menu = _find_menu(self, "View")
        if view_menu is not None:
            view_menu.addSeparator()
            view_menu.addAction(self.performance_action)
        refresh_status()

    PlanningView.__init__ = view_init
    PlanningView.set_performance_mode = view_set_performance_mode
    PlanningScene.drawBackground = scene_draw_background
    GraphNodeItem.__init__ = node_init
    GraphNodeItem.paint = node_paint
    GraphGroupItem.paint = group_paint
    GraphEdgeItem.paint = edge_paint
    PlanningGraphWindow.__init__ = window_init
    _PATCHED = True
