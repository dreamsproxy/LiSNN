"""Adaptive, explicit group-title rendering and editing flow.

Groups already persist a title in the document model and expose it in the live
inspector. The normal label is intentionally culled by the LOD renderer at
large overviews, however, which makes architectural layers difficult to
identify. This module adds a screen-readable title banner only when the normal
label has been culled and the projected group is large enough to label.

Newly created groups also emit the existing edit request immediately, placing
keyboard focus in the live title field without a modal dialog or extra undo
entry.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen


MIN_PROJECTED_GROUP_WIDTH = 72.0
TARGET_TITLE_PIXELS = 13.0
TARGET_BANNER_PIXELS = 26.0
MAX_SCENE_FONT_PIXELS = 640

_INSTALLED = False


def overview_group_title_visible(lod: float, group_width: float) -> bool:
    """Return whether a culled group title is still useful at this zoom."""

    return lod > 0.0 and float(group_width) * float(lod) >= MIN_PROJECTED_GROUP_WIDTH


def adaptive_title_font_pixels(lod: float) -> int:
    """Return a scene-space font size targeting a stable screen size."""

    lod = max(1.0e-4, float(lod))
    return max(9, min(MAX_SCENE_FONT_PIXELS, round(TARGET_TITLE_PIXELS / lod)))


def install_group_title_patches() -> None:
    """Install overview titles and first-class group-title editing once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from .items import GraphGroupItem
    from .performance import _lod_from_painter, active_profile
    from .scene import PlanningScene
    from .theme import TEXT

    original_paint = GraphGroupItem.paint
    original_group_selected_nodes = PlanningScene.group_selected_nodes

    def paint(self, painter: QPainter, option, widget=None) -> None:
        original_paint(self, painter, option, widget)

        lod = _lod_from_painter(painter)
        profile = active_profile()
        if lod >= profile.group_label_lod:
            return
        if not overview_group_title_visible(lod, self.model.width):
            return

        rect = self.content_rect()
        scene_padding = max(5.0, 7.0 / lod)
        banner_height = min(
            rect.height(),
            max(24.0, TARGET_BANNER_PIXELS / lod),
        )
        available_width = max(1.0, rect.width() - scene_padding * 2.0)

        font = QFont()
        font.setPixelSize(adaptive_title_font_pixels(lod))
        font.setBold(True)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setFont(font)
        title = painter.fontMetrics().elidedText(
            self.model.title,
            Qt.TextElideMode.ElideRight,
            max(1, int(available_width - scene_padding * 2.0)),
        )
        text_width = float(painter.fontMetrics().horizontalAdvance(title))
        banner_width = min(
            available_width,
            max(banner_height * 2.5, text_width + scene_padding * 2.0),
        )
        banner = QRectF(
            rect.left() + scene_padding,
            rect.top() + scene_padding,
            banner_width,
            banner_height,
        )

        fill = QColor(self.model.color)
        fill.setAlpha(232)
        painter.setPen(QPen(QColor(self.model.color), max(1.0, 1.0 / lod)))
        painter.setBrush(QBrush(fill))
        radius = min(banner_height * 0.22, 8.0 / lod)
        painter.drawRoundedRect(banner, radius, radius)
        painter.setPen(QPen(QColor(TEXT), 1.0))
        painter.drawText(
            banner.adjusted(scene_padding, 0.0, -scene_padding, 0.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            title,
        )
        painter.restore()

    def group_selected_nodes(self):
        group = original_group_selected_nodes(self)
        if group is not None:
            self.edit_group_requested.emit(group.group_id)
        return group

    GraphGroupItem.paint = paint
    PlanningScene.group_selected_nodes = group_selected_nodes
    _INSTALLED = True
