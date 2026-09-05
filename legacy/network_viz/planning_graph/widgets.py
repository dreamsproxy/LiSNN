"""Palette, definition dialog, and live inspector widgets."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QColor, QDrag, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .definitions import DefinitionRegistry, normalize_color
from .grid import (
    GRID_SIZE,
    GROUP_MIN_HEIGHT_CELLS,
    GROUP_MIN_WIDTH_CELLS,
    NODE_MIN_HEIGHT_CELLS,
    NODE_MIN_WIDTH_CELLS,
)
from .models import EDGE_TYPES, PlanningEdge, PlanningGroup, PlanningNode
from .scene import MIME_NODE_KIND
from .theme import EDGE_COLORS, TEXT


CUSTOM_KIND_ENTRY = "Custom..."


def color_icon(color: str, size: int = 14) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


def set_color_button(button: QPushButton, color: str) -> str:
    normalized = QColor(color).name().upper()
    button.setText(normalized)
    button.setIcon(color_icon(normalized, 18))
    button.setStyleSheet(
        f"background: {normalized}; color: {TEXT}; font-weight: 700; "
        "border: 1px solid #8292A3; padding: 6px;"
    )
    return normalized


class NodePalette(QListWidget):
    """Drag and double-click source for configured planning block types."""

    create_requested = Signal(str)

    def __init__(self, registry: DefinitionRegistry) -> None:
        super().__init__()
        self.registry = registry
        self.setDragEnabled(True)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setSpacing(3)
        self.itemDoubleClicked.connect(self._double_clicked)
        self.refresh()

    def refresh(self) -> None:
        self.clear()
        for name in self.registry.block_names():
            item = QListWidgetItem(color_icon(self.registry.block_color(name)), name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setToolTip(
                f"Drag {name} onto the canvas or double-click to place it in the visible center"
            )
            self.addItem(item)

    def _double_clicked(self, item: QListWidgetItem) -> None:
        self.create_requested.emit(str(item.data(Qt.ItemDataRole.UserRole)))

    def startDrag(self, supported_actions) -> None:
        item = self.currentItem()
        if item is None:
            return
        kind = str(item.data(Qt.ItemDataRole.UserRole))
        mime = QMimeData()
        mime.setData(MIME_NODE_KIND, kind.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class NewBlockDialog(QDialog):
    """Create one persistent block definition in ``custom_blocks.json``."""

    def __init__(self, registry: DefinitionRegistry, parent=None) -> None:
        super().__init__(parent)
        self.registry = registry
        self.color_value = "#70A7FF"
        self.setWindowTitle("New Planning Block Type")
        self.setMinimumWidth(380)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Example: Experiment, Dataset, Risk")
        self.color_button = QPushButton()
        set_color_button(self.color_button, self.color_value)
        self.color_button.clicked.connect(self._choose_color)
        self.preview = QLabel("New custom block")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(54)
        self._refresh_preview()

        form = QFormLayout()
        form.addRow("Block type", self.name_edit)
        form.addRow("Accent color", self.color_button)
        form.addRow("Preview", self.preview)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _choose_color(self) -> None:
        color = QColorDialog.getColor(QColor(self.color_value), self, "Block Accent Color")
        if color.isValid():
            self.color_value = set_color_button(self.color_button, color.name())
            self._refresh_preview()

    def _refresh_preview(self) -> None:
        self.preview.setStyleSheet(
            f"background: #252C34; color: {TEXT}; border-left: 8px solid {self.color_value}; "
            "border-radius: 5px; font-weight: 700;"
        )

    def values(self) -> tuple[str, str]:
        return self.name_edit.text().strip(), normalize_color(self.color_value)


class InspectorPanel(QWidget):
    """Immediately emits every property edit; no Apply button is required."""

    node_changed = Signal(str, dict)
    edge_changed = Signal(str, dict)
    group_changed = Signal(str, dict)

    def __init__(self, registry: DefinitionRegistry) -> None:
        super().__init__()
        self.registry = registry
        self.current_node_id: str | None = None
        self.current_edge_id: str | None = None
        self.current_group_id: str | None = None
        self.group_color_value = "#3B5B78"
        self._loading = False

        title = QLabel("Properties")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        self.stack = QStackedWidget()
        self.empty_page = self._build_empty_page()
        self.node_page = self._build_node_page()
        self.edge_page = self._build_edge_page()
        self.group_page = self._build_group_page()
        for page in (self.empty_page, self.node_page, self.edge_page, self.group_page):
            self.stack.addWidget(page)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.stack, 1)
        self.refresh_registry()
        self._connect_live_signals()
        self.show_empty()

    def _build_empty_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        message = QLabel(
            "Select one block, connection, or group. Property changes are applied immediately and can be undone with Ctrl+Z."
        )
        message.setWordWrap(True)
        message.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(message)
        layout.addStretch(1)
        return page

    def _build_node_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.node_kind = QComboBox()
        self.node_custom_kind = QLineEdit()
        self.node_custom_kind.setPlaceholderText("Custom block kind")
        self.node_title = QLineEdit()
        self.node_status = QComboBox()
        self.node_priority = QSpinBox()
        self.node_priority.setRange(0, 5)
        self.node_tags = QLineEdit()
        self.node_tags.setPlaceholderText("comma, separated, tags")
        self.node_body = QTextEdit()
        self.node_body.setMinimumHeight(150)
        self.node_body.setPlaceholderText("Reasoning, details, assumptions, or notes...")
        self.node_width_cells = QSpinBox()
        self.node_width_cells.setRange(NODE_MIN_WIDTH_CELLS, 400)
        self.node_height_cells = QSpinBox()
        self.node_height_cells.setRange(NODE_MIN_HEIGHT_CELLS, 400)
        self.node_header_font = QSpinBox()
        self.node_title_font = QSpinBox()
        self.node_body_font = QSpinBox()
        self.node_footer_font = QSpinBox()
        for control in (
            self.node_header_font,
            self.node_title_font,
            self.node_body_font,
            self.node_footer_font,
        ):
            control.setRange(6, 72)
            control.setSuffix(" pt")

        form.addRow("Kind", self.node_kind)
        form.addRow("Custom kind", self.node_custom_kind)
        form.addRow("Title", self.node_title)
        form.addRow("Status", self.node_status)
        form.addRow("Priority", self.node_priority)
        form.addRow("Tags", self.node_tags)
        form.addRow("Context", self.node_body)
        form.addRow("Width (cells)", self.node_width_cells)
        form.addRow("Height (cells)", self.node_height_cells)
        form.addRow("Header font", self.node_header_font)
        form.addRow("Title font", self.node_title_font)
        form.addRow("Context font", self.node_body_font)
        form.addRow("Footer font", self.node_footer_font)
        return page

    def _build_edge_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.edge_relation = QComboBox()
        for relation in EDGE_TYPES:
            self.edge_relation.addItem(
                color_icon(EDGE_COLORS.get(relation, "#94A3B2")),
                relation,
            )
        self.edge_label = QLineEdit()
        self.edge_label.setPlaceholderText("Optional human-readable label")
        self.edge_weight = QDoubleSpinBox()
        self.edge_weight.setRange(-1_000_000.0, 1_000_000.0)
        self.edge_weight.setDecimals(4)
        self.edge_route_mode = QLabel()
        self.edge_route_mode.setWordWrap(True)
        form.addRow("Relation", self.edge_relation)
        form.addRow("Label", self.edge_label)
        form.addRow("Weight", self.edge_weight)
        form.addRow("Routing", self.edge_route_mode)
        return page

    def _build_group_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.group_title = QLineEdit()
        self.group_backdrop = QCheckBox("Show colored backdrop")
        self.group_color_button = QPushButton()
        self.group_color_button.clicked.connect(self._choose_group_color)
        self.group_layer = QSpinBox()
        self.group_layer.setRange(-1000, 1000)
        self.group_width_cells = QSpinBox()
        self.group_width_cells.setRange(GROUP_MIN_WIDTH_CELLS, 400)
        self.group_height_cells = QSpinBox()
        self.group_height_cells.setRange(GROUP_MIN_HEIGHT_CELLS, 400)
        self.group_members = QLabel()
        self.group_members.setWordWrap(True)
        form.addRow("Title", self.group_title)
        form.addRow("Backdrop", self.group_backdrop)
        form.addRow("Color", self.group_color_button)
        form.addRow("Layer", self.group_layer)
        form.addRow("Width (cells)", self.group_width_cells)
        form.addRow("Height (cells)", self.group_height_cells)
        form.addRow("Members", self.group_members)
        return page

    def refresh_registry(self) -> None:
        current_kind = self.node_kind.currentText() if hasattr(self, "node_kind") else ""
        current_status = self.node_status.currentText() if hasattr(self, "node_status") else ""
        if hasattr(self, "node_kind"):
            self.node_kind.blockSignals(True)
            self.node_kind.clear()
            for name in self.registry.block_names():
                self.node_kind.addItem(color_icon(self.registry.block_color(name)), name)
            self.node_kind.addItem(CUSTOM_KIND_ENTRY)
            self.node_kind.setCurrentText(current_kind or self.registry.block_names()[0])
            self.node_kind.blockSignals(False)
        if hasattr(self, "node_status"):
            self.node_status.blockSignals(True)
            self.node_status.clear()
            for name in self.registry.status_names():
                self.node_status.addItem(color_icon(self.registry.status_color(name)), name)
            self.node_status.setCurrentText(current_status or "None")
            self.node_status.blockSignals(False)

    def _connect_live_signals(self) -> None:
        self.node_kind.currentTextChanged.connect(self._kind_changed)
        self.node_custom_kind.textEdited.connect(self._emit_node)
        self.node_title.textEdited.connect(self._emit_node)
        self.node_status.currentTextChanged.connect(self._emit_node)
        self.node_priority.valueChanged.connect(self._emit_node)
        self.node_tags.textEdited.connect(self._emit_node)
        self.node_body.textChanged.connect(self._emit_node)
        self.node_width_cells.valueChanged.connect(self._emit_node)
        self.node_height_cells.valueChanged.connect(self._emit_node)
        self.node_header_font.valueChanged.connect(self._emit_node)
        self.node_title_font.valueChanged.connect(self._emit_node)
        self.node_body_font.valueChanged.connect(self._emit_node)
        self.node_footer_font.valueChanged.connect(self._emit_node)

        self.edge_relation.currentTextChanged.connect(self._emit_edge)
        self.edge_label.textEdited.connect(self._emit_edge)
        self.edge_weight.valueChanged.connect(self._emit_edge)

        self.group_title.textEdited.connect(self._emit_group)
        self.group_backdrop.toggled.connect(self._emit_group)
        self.group_layer.valueChanged.connect(self._emit_group)
        self.group_width_cells.valueChanged.connect(self._emit_group)
        self.group_height_cells.valueChanged.connect(self._emit_group)

    def _kind_changed(self, value: str) -> None:
        self.node_custom_kind.setVisible(value == CUSTOM_KIND_ENTRY)
        self._emit_node()

    def show_empty(self) -> None:
        self.current_node_id = None
        self.current_edge_id = None
        self.current_group_id = None
        self.stack.setCurrentWidget(self.empty_page)

    def show_node(self, node: PlanningNode) -> None:
        self._loading = True
        try:
            self.current_node_id = node.node_id
            self.current_edge_id = None
            self.current_group_id = None
            if self.registry.has_block_type(node.kind):
                self.node_kind.setCurrentText(node.kind)
                self.node_custom_kind.clear()
            else:
                self.node_kind.setCurrentText(CUSTOM_KIND_ENTRY)
                self.node_custom_kind.setText(node.kind)
            self.node_custom_kind.setVisible(self.node_kind.currentText() == CUSTOM_KIND_ENTRY)
            self.node_title.setText(node.title)
            if self.registry.has_status(node.status):
                self.node_status.setCurrentText(node.status)
            else:
                self.node_status.setCurrentText("None")
            self.node_priority.setValue(node.priority)
            self.node_tags.setText(", ".join(node.tags))
            self.node_body.setPlainText(node.body)
            self.node_width_cells.setValue(round(node.width / GRID_SIZE))
            self.node_height_cells.setValue(round(node.height / GRID_SIZE))
            self.node_header_font.setValue(node.header_font_size)
            self.node_title_font.setValue(node.title_font_size)
            self.node_body_font.setValue(node.body_font_size)
            self.node_footer_font.setValue(node.footer_font_size)
            self.stack.setCurrentWidget(self.node_page)
        finally:
            self._loading = False

    def show_edge(self, edge: PlanningEdge) -> None:
        self._loading = True
        try:
            self.current_edge_id = edge.edge_id
            self.current_node_id = None
            self.current_group_id = None
            self.edge_relation.setCurrentText(edge.relation)
            self.edge_label.setText(edge.label)
            self.edge_weight.setValue(edge.weight)
            self.edge_route_mode.setText(
                "Manual route: drag the blue route handle. Right-click the connection to reset."
                if edge.has_manual_route
                else "Automatic radial route. Drag the connection to define a manual route."
            )
            self.stack.setCurrentWidget(self.edge_page)
        finally:
            self._loading = False

    def show_group(self, group: PlanningGroup, member_titles: list[str]) -> None:
        self._loading = True
        try:
            self.current_group_id = group.group_id
            self.current_node_id = None
            self.current_edge_id = None
            self.group_title.setText(group.title)
            self.group_backdrop.setChecked(group.backdrop)
            self.group_layer.setValue(group.layer)
            self.group_width_cells.setValue(round(group.width / GRID_SIZE))
            self.group_height_cells.setValue(round(group.height / GRID_SIZE))
            self.group_members.setText("\n".join(member_titles))
            self.group_color_value = set_color_button(self.group_color_button, group.color)
            self.stack.setCurrentWidget(self.group_page)
        finally:
            self._loading = False

    def focus_title(self) -> None:
        page = self.stack.currentWidget()
        if page is self.node_page:
            self.node_title.setFocus()
            self.node_title.selectAll()
        elif page is self.edge_page:
            self.edge_label.setFocus()
            self.edge_label.selectAll()
        elif page is self.group_page:
            self.group_title.setFocus()
            self.group_title.selectAll()

    def _choose_group_color(self) -> None:
        color = QColorDialog.getColor(QColor(self.group_color_value), self, "Backdrop Color")
        if color.isValid():
            self.group_color_value = set_color_button(self.group_color_button, color.name())
            self._emit_group()

    def _emit_node(self, *args) -> None:
        if self._loading or self.current_node_id is None:
            return
        kind = self.node_kind.currentText()
        if kind == CUSTOM_KIND_ENTRY:
            kind = self.node_custom_kind.text().strip() or "Custom"
        payload: dict[str, Any] = {
            "kind": kind,
            "title": self.node_title.text().strip() or "Untitled",
            "status": self.node_status.currentText() or "None",
            "priority": self.node_priority.value(),
            "tags": [tag.strip() for tag in self.node_tags.text().split(",") if tag.strip()],
            "body": self.node_body.toPlainText(),
            "width": self.node_width_cells.value() * GRID_SIZE,
            "height": self.node_height_cells.value() * GRID_SIZE,
            "header_font_size": self.node_header_font.value(),
            "title_font_size": self.node_title_font.value(),
            "body_font_size": self.node_body_font.value(),
            "footer_font_size": self.node_footer_font.value(),
        }
        self.node_changed.emit(self.current_node_id, payload)

    def _emit_edge(self, *args) -> None:
        if self._loading or self.current_edge_id is None:
            return
        self.edge_changed.emit(
            self.current_edge_id,
            {
                "relation": self.edge_relation.currentText(),
                "label": self.edge_label.text(),
                "weight": self.edge_weight.value(),
            },
        )

    def _emit_group(self, *args) -> None:
        if self._loading or self.current_group_id is None:
            return
        self.group_changed.emit(
            self.current_group_id,
            {
                "title": self.group_title.text().strip() or "Group",
                "backdrop": self.group_backdrop.isChecked(),
                "color": self.group_color_value,
                "layer": self.group_layer.value(),
                "width": self.group_width_cells.value() * GRID_SIZE,
                "height": self.group_height_cells.value() * GRID_SIZE,
            },
        )
