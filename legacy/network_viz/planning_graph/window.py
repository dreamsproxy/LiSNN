"""Main window for the LiSNN U1 planning graph editor."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .definitions import DefinitionRegistry
from .items import GraphEdgeItem, GraphGroupItem, GraphNodeItem
from .models import EDGE_TYPES, PlanningDocument
from .scene import PlanningScene, PlanningView
from .serialization import SQLITE_SUFFIXES, load_document, save_document
from .theme import EDGE_COLORS
from .widgets import InspectorPanel, NewBlockDialog, NodePalette, color_icon


GRAPH_FILTER = (
    "Planning Graph JSON (*.json);;"
    "Planning Graph SQLite (*.db *.sqlite *.sqlite3);;"
    "All Supported Graphs (*.json *.db *.sqlite *.sqlite3)"
)


class PlanningGraphWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LiSNN Planning Graph - U1 Alpha")
        self.resize(1540, 940)
        self.current_path: Path | None = None
        self.dirty = False

        self.registry = DefinitionRegistry.load()
        self.scene = PlanningScene(self.registry)
        self.view = PlanningView(self.scene)
        self.palette = NodePalette(self.registry)
        self.inspector = InspectorPanel(self.registry)

        self._build_central_layout()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._connect_signals()
        self.statusBar().showMessage(
            "Drag or double-click blocks. Drag connections to route them. All edits are immediate and undoable."
        )
        self._update_title()

    def _build_central_layout(self) -> None:
        palette_panel = QWidget()
        palette_layout = QVBoxLayout(palette_panel)
        palette_title = QLabel("Planning Blocks")
        palette_title.setStyleSheet("font-size: 15px; font-weight: 700;")
        help_label = QLabel(
            "Drag onto the canvas, or double-click to place a block at the visible center."
        )
        help_label.setWordWrap(True)
        self.new_block_button = QPushButton("+ New Block")
        self.new_block_button.setToolTip(
            "Create a persistent custom block type in planning_graph/custom_blocks.json"
        )
        palette_layout.addWidget(palette_title)
        palette_layout.addWidget(help_label)
        palette_layout.addWidget(self.palette, 1)
        palette_layout.addWidget(self.new_block_button)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(palette_panel)
        splitter.addWidget(self.view)
        splitter.addWidget(self.inspector)
        splitter.setSizes([230, 980, 360])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    def _build_actions(self) -> None:
        self.new_action = QAction("New", self)
        self.new_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_action.triggered.connect(self.new_document)

        self.open_action = QAction("Open...", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_document)

        self.save_action = QAction("Save", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self.save_document)

        self.save_as_action = QAction("Save As...", self)
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(self.save_document_as)

        self.load_example_action = QAction("Load LiSNN Roadmap Example", self)
        self.load_example_action.triggered.connect(self.load_lisnn_example)

        self.undo_action = self.scene.undo_stack.createUndoAction(self, "Undo")
        self.undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        self.redo_action = self.scene.undo_stack.createRedoAction(self, "Redo")
        self.redo_action.setShortcut(QKeySequence("Ctrl+Y"))

        self.delete_action = QAction("Delete Selected", self)
        self.delete_action.setShortcut(QKeySequence.StandardKey.Delete)
        self.delete_action.triggered.connect(self.scene.delete_selected)

        self.duplicate_action = QAction("Duplicate Block", self)
        self.duplicate_action.setShortcut("Ctrl+D")
        self.duplicate_action.triggered.connect(self.scene.duplicate_selected_node)

        self.group_action = QAction("Group Selected Blocks", self)
        self.group_action.setShortcut("Ctrl+G")
        self.group_action.triggered.connect(self.scene.group_selected_nodes)

        self.ungroup_action = QAction("Ungroup Selected", self)
        self.ungroup_action.setShortcut("Ctrl+Shift+G")
        self.ungroup_action.triggered.connect(self._ungroup_selected)

        self.invert_action = QAction("Invert Selection", self)
        self.invert_action.setShortcut("Ctrl+Shift+I")
        self.invert_action.triggered.connect(self.scene.invert_selection)

        self.fit_action = QAction("Fit Graph", self)
        self.fit_action.setShortcut("F")
        self.fit_action.triggered.connect(self.view.fit_graph)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.new_action)
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.load_example_action)

        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.duplicate_action)
        edit_menu.addAction(self.group_action)
        edit_menu.addAction(self.ungroup_action)
        edit_menu.addAction(self.invert_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.delete_action)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.fit_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Planning Graph")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction(self.new_action)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_action)
        toolbar.addSeparator()
        toolbar.addAction(self.undo_action)
        toolbar.addAction(self.redo_action)
        toolbar.addSeparator()
        toolbar.addAction(self.duplicate_action)
        toolbar.addAction(self.group_action)
        toolbar.addAction(self.delete_action)
        toolbar.addAction(self.fit_action)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("New connection: "))
        self.relation_combo = QComboBox()
        for relation in EDGE_TYPES:
            self.relation_combo.addItem(
                color_icon(EDGE_COLORS.get(relation, "#94A3B2")),
                relation,
            )
        self.relation_combo.setCurrentText("Leads To")
        self.relation_combo.currentTextChanged.connect(self._set_default_relation)
        toolbar.addWidget(self.relation_combo)

    def _connect_signals(self) -> None:
        self.scene.document_changed.connect(self._mark_dirty)
        self.scene.selectionChanged.connect(self._selection_changed)
        self.scene.edit_node_requested.connect(self._focus_node)
        self.scene.edit_edge_requested.connect(self._focus_edge)
        self.scene.edit_group_requested.connect(self._focus_group)
        self.scene.status_message.connect(
            lambda message: self.statusBar().showMessage(message, 5000)
        )
        self.inspector.node_changed.connect(self.scene.update_node_properties)
        self.inspector.edge_changed.connect(self.scene.update_edge_properties)
        self.inspector.group_changed.connect(self.scene.update_group_properties)
        self.palette.create_requested.connect(self._create_block_at_center)
        self.new_block_button.clicked.connect(self._create_custom_block_type)
        self.scene.undo_stack.cleanChanged.connect(self._history_clean_changed)

    def _set_default_relation(self, relation: str) -> None:
        self.scene.default_relation = relation

    def _mark_dirty(self) -> None:
        self.dirty = True
        self._update_title()

    def _history_clean_changed(self, clean: bool) -> None:
        if clean:
            self.dirty = False
            self._update_title()

    def _update_title(self) -> None:
        name = self.current_path.name if self.current_path else self.scene.document.title
        marker = " *" if self.dirty else ""
        self.setWindowTitle(f"LiSNN Planning Graph - {name}{marker}")

    def _selection_changed(self) -> None:
        self._refresh_selected_inspector()

    def _refresh_selected_inspector(self) -> None:
        selected = self.scene.selectedItems()
        if len(selected) != 1:
            self.inspector.show_empty()
            return
        item = selected[0]
        if isinstance(item, GraphNodeItem):
            current = self.scene.document.nodes.get(item.model.node_id)
            if current is not None:
                self.inspector.show_node(current)
        elif isinstance(item, GraphEdgeItem):
            current = self.scene.document.edges.get(item.model.edge_id)
            if current is not None:
                self.inspector.show_edge(current)
        elif isinstance(item, GraphGroupItem):
            current = self.scene.document.groups.get(item.model.group_id)
            if current is not None:
                titles = [
                    self.scene.document.nodes[node_id].title
                    for node_id in current.node_ids
                    if node_id in self.scene.document.nodes
                ]
                self.inspector.show_group(current, titles)
        else:
            self.inspector.show_empty()

    def _focus_node(self, node_id: str) -> None:
        item = self.scene.node_items.get(node_id)
        if item is None:
            return
        self.scene.clearSelection()
        item.setSelected(True)
        self.inspector.show_node(item.model)
        self.inspector.focus_title()

    def _focus_edge(self, edge_id: str) -> None:
        item = self.scene.edge_items.get(edge_id)
        if item is None:
            return
        self.scene.clearSelection()
        item.setSelected(True)
        self.inspector.show_edge(item.model)
        self.inspector.focus_title()

    def _focus_group(self, group_id: str) -> None:
        item = self.scene.group_items.get(group_id)
        if item is None:
            return
        self.scene.clearSelection()
        item.setSelected(True)
        titles = [
            self.scene.document.nodes[node_id].title
            for node_id in item.model.node_ids
            if node_id in self.scene.document.nodes
        ]
        self.inspector.show_group(item.model, titles)
        self.inspector.focus_title()

    def _ungroup_selected(self) -> None:
        group_ids = [
            item.model.group_id
            for item in self.scene.selectedItems()
            if isinstance(item, GraphGroupItem)
        ]
        for group_id in group_ids:
            self.scene.ungroup(group_id)

    def _create_block_at_center(self, kind: str) -> None:
        node = self.scene.create_node(kind, self.view.visible_scene_center())
        self.statusBar().showMessage(f"Created {node.kind}: {node.title}", 3000)

    def _create_custom_block_type(self) -> None:
        dialog = NewBlockDialog(self.registry, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        name, color = dialog.values()
        try:
            definition = self.registry.add_block_type(name, color)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot create block type", str(exc))
            return
        self.palette.refresh()
        self.inspector.refresh_registry()
        self.scene.refresh_registry()
        self._create_block_at_center(definition.name)
        self.statusBar().showMessage(
            f"Saved custom block type {definition.name} to {self.registry.path.name}",
            5000,
        )

    def new_document(self) -> None:
        if not self._confirm_discard_changes():
            return
        self.current_path = None
        self.dirty = False
        self.scene.load_document(PlanningDocument(), clear_history=True)
        self.scene.undo_stack.setClean()
        self.inspector.show_empty()
        self._update_title()

    def open_document(self) -> None:
        if not self._confirm_discard_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Planning Graph",
            "",
            GRAPH_FILTER,
        )
        if path:
            self._load_path(Path(path))

    def save_document(self) -> bool:
        if self.current_path is None:
            return self.save_document_as()
        try:
            save_document(self.scene.document, self.current_path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        self.dirty = False
        self.scene.undo_stack.setClean()
        self._update_title()
        self.statusBar().showMessage(f"Saved {self.current_path}", 4000)
        return True

    def save_document_as(self) -> bool:
        suggested = self.current_path.name if self.current_path else "planning_graph.json"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Planning Graph",
            suggested,
            GRAPH_FILTER,
        )
        if not path:
            return False
        destination = Path(path)
        if not destination.suffix:
            destination = destination.with_suffix(
                ".db" if "SQLite" in selected_filter else ".json"
            )
        elif destination.suffix.lower() not in SQLITE_SUFFIXES | {".json"}:
            destination = destination.with_suffix(".json")
        self.current_path = destination
        return self.save_document()

    def load_lisnn_example(self) -> None:
        if not self._confirm_discard_changes():
            return
        example = Path(__file__).resolve().parent / "examples" / "lisnn_roadmap.json"
        self._load_path(example, treat_as_unsaved=True)

    def _load_path(self, path: Path, *, treat_as_unsaved: bool = False) -> None:
        try:
            document = load_document(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self.scene.load_document(document, clear_history=True)
        self.current_path = None if treat_as_unsaved else path
        self.dirty = treat_as_unsaved
        if not treat_as_unsaved:
            self.scene.undo_stack.setClean()
        self.inspector.show_empty()
        self._update_title()
        self.view.fit_graph()
        self.statusBar().showMessage(f"Loaded {path.name}", 4000)

    def _confirm_discard_changes(self) -> bool:
        if not self.dirty:
            return True
        choice = QMessageBox.question(
            self,
            "Unsaved changes",
            "Save changes before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Save:
            return self.save_document()
        return choice == QMessageBox.StandardButton.Discard

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._confirm_discard_changes():
            event.accept()
        else:
            event.ignore()
