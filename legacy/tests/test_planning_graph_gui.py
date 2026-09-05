"""Offscreen smoke tests for the PySide6 planning graph alpha."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, QRectF
from PySide6.QtWidgets import QApplication

from planning_graph.definitions import DefinitionRegistry
from planning_graph.group_titles import (
    adaptive_title_font_pixels,
    overview_group_title_visible,
)
from planning_graph.rendering import (
    configure_runtime,
    detail_tier,
    grid_step_for_lod,
    install_runtime_patches,
)
from planning_graph.scene import PlanningScene
from planning_graph.window import PlanningGraphWindow
from planning_graph.world import expanded_world_rect


configure_runtime(software_rendering=True, performance_mode=False)
install_runtime_patches()
_APP = QApplication.instance() or QApplication([])


class PlanningGraphGuiTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_runtime(software_rendering=True, performance_mode=False)
        self.temporary_directory = tempfile.TemporaryDirectory()
        path = Path(self.temporary_directory.name) / "custom_blocks.json"
        self.registry = DefinitionRegistry.defaults(path)
        self.registry.save()
        self.scene = PlanningScene(self.registry)

    def tearDown(self) -> None:
        self.scene.clear()
        self.temporary_directory.cleanup()

    def test_scene_defaults_to_leads_to(self) -> None:
        self.assertEqual(self.scene.default_relation, "Leads To")

    def test_create_undo_redo_round_trip(self) -> None:
        self.scene.create_node("Idea", QPointF(12.0, 13.0))
        self.assertEqual(len(self.scene.document.nodes), 1)
        self.scene.undo_stack.undo()
        self.assertEqual(len(self.scene.document.nodes), 0)
        self.scene.undo_stack.redo()
        self.assertEqual(len(self.scene.document.nodes), 1)

    def test_live_property_edits_merge_into_one_undo(self) -> None:
        node = self.scene.create_node("Idea", QPointF())
        self.scene.update_node_properties(node.node_id, {"title": "A"})
        self.scene.update_node_properties(node.node_id, {"title": "AB"})
        self.scene.update_node_properties(node.node_id, {"title": "ABC"})
        self.assertEqual(self.scene.document.nodes[node.node_id].title, "ABC")
        self.scene.undo_stack.undo()
        self.assertEqual(
            self.scene.document.nodes[node.node_id].title,
            "New Idea",
        )

    def test_live_block_dimensions_update_item_geometry(self) -> None:
        node = self.scene.create_node("Idea", QPointF())
        self.scene.update_node_properties(
            node.node_id,
            {"width": 275.0, "height": 225.0},
        )
        current = self.scene.document.nodes[node.node_id]
        item = self.scene.node_items[node.node_id]
        self.assertEqual(current.width, 300.0)
        self.assertEqual(current.height, 250.0)
        self.assertEqual(item.content_rect().width(), current.width)
        self.assertEqual(item.content_rect().height(), current.height)

    def test_parallel_departures_receive_distinct_lanes(self) -> None:
        source = self.scene.create_node("Idea", QPointF(0.0, 0.0))
        first = self.scene.create_node("Task", QPointF(400.0, -150.0))
        second = self.scene.create_node("Task", QPointF(400.0, 150.0))
        edge_a = self.scene.create_edge(source.node_id, first.node_id)
        edge_b = self.scene.create_edge(source.node_id, second.node_id)
        offsets = {
            self.scene.edge_items[edge_a.edge_id].lane_offset,
            self.scene.edge_items[edge_b.edge_id].lane_offset,
        }
        self.assertEqual(len(offsets), 2)
        self.assertNotEqual(offsets, {0.0})

    def test_manual_route_can_reset_to_automatic(self) -> None:
        source = self.scene.create_node("Idea", QPointF())
        target = self.scene.create_node("Task", QPointF(400.0, 0.0))
        edge = self.scene.create_edge(source.node_id, target.node_id)
        edge.route_x = 100.0
        edge.route_y = 100.0
        edge.__post_init__()
        self.scene.edge_items[edge.edge_id].update_geometry()
        self.assertTrue(edge.has_manual_route)
        self.scene.reset_edge_route(edge.edge_id)
        self.assertFalse(
            self.scene.document.edges[edge.edge_id].has_manual_route
        )

    def test_balanced_lod_tiers_are_progressive(self) -> None:
        self.assertEqual(detail_tier(0.80), "full")
        self.assertEqual(detail_tier(0.50), "compact")
        self.assertEqual(detail_tier(0.30), "title")
        self.assertEqual(detail_tier(0.10), "silhouette")

    def test_grid_density_adapts_and_disappears_at_overview(self) -> None:
        self.assertEqual(grid_step_for_lod(1.00), 25.0)
        self.assertEqual(grid_step_for_lod(0.50), 100.0)
        self.assertEqual(grid_step_for_lod(0.30), 250.0)
        self.assertIsNone(grid_step_for_lod(0.10))

    def test_overview_group_titles_remain_readable_for_large_groups(self) -> None:
        self.assertTrue(overview_group_title_visible(0.05, 3600.0))
        self.assertFalse(overview_group_title_visible(0.05, 400.0))
        self.assertGreater(adaptive_title_font_pixels(0.05), 100)

    def test_world_rect_expands_outward_in_chunks(self) -> None:
        current = QRectF(-10_000.0, -10_000.0, 20_000.0, 20_000.0)
        expanded = expanded_world_rect(
            current,
            QRectF(48_000.0, -200.0, 400.0, 400.0),
        )
        self.assertLessEqual(expanded.left(), current.left())
        self.assertGreaterEqual(expanded.right(), 50_000.0)
        self.assertEqual(expanded.right() % 10_000.0, 0.0)

    def test_creating_distant_block_expands_scene_world(self) -> None:
        before = QRectF(self.scene.sceneRect())
        node = self.scene.create_node("Idea", QPointF(75_000.0, 0.0))
        item = self.scene.node_items[node.node_id]
        after = self.scene.sceneRect()
        self.assertGreater(after.right(), before.right())
        self.assertTrue(after.contains(item.sceneBoundingRect()))

    def test_zoom_recovers_from_scale_below_old_floor(self) -> None:
        window = PlanningGraphWindow()
        try:
            view = window.view
            view.resetTransform()
            view.scale(0.05, 0.05)
            before = view.current_zoom()
            after = view.zoom_by(1.15)
            self.assertAlmostEqual(before, 0.05, places=6)
            self.assertAlmostEqual(after, 0.0575, places=6)
            self.assertGreater(after, before)
            self.assertLess(after, 0.15)
        finally:
            window.close()

    def test_pan_by_pixels_changes_visible_scene_center(self) -> None:
        window = PlanningGraphWindow()
        try:
            window.show()
            _APP.processEvents()
            view = window.view
            view.resetTransform()
            view.zoom_to(1.0)
            view.centerOn(0.0, 0.0)
            _APP.processEvents()
            before = view.visible_scene_center()
            view.pan_by_pixels(QPoint(120, 80))
            _APP.processEvents()
            after = view.visible_scene_center()
            self.assertLess(after.x(), before.x())
            self.assertLess(after.y(), before.y())
        finally:
            window.close()

    def test_pan_beyond_current_boundary_expands_world(self) -> None:
        window = PlanningGraphWindow()
        try:
            window.show()
            _APP.processEvents()
            view = window.view
            view.resetTransform()
            view.zoom_to(1.0)
            before = QRectF(window.scene.sceneRect())
            view.pan_by_pixels(QPoint(-60_000, 0))
            _APP.processEvents()
            after = window.scene.sceneRect()
            self.assertGreater(after.right(), before.right())
            self.assertGreater(view.visible_scene_center().x(), before.right())
        finally:
            window.close()

    def test_window_constructs_with_renderer_controls(self) -> None:
        window = PlanningGraphWindow()
        try:
            self.assertEqual(window.scene.default_relation, "Leads To")
            self.assertEqual(window.relation_combo.currentText(), "Leads To")
            self.assertGreater(window.palette.count(), 0)
            self.assertEqual(window.view.renderer_backend, "Software Raster")
            self.assertIn(
                "Renderer: Software Raster",
                window.renderer_status_label.text(),
            )
            self.assertIsNotNone(window.rendering_menu)
            self.assertFalse(window.performance_action.isChecked())
            window.performance_action.setChecked(True)
            self.assertIn("LOD: Performance", window.renderer_status_label.text())
            window.performance_action.setChecked(False)
            self.assertIn("LOD: Balanced", window.renderer_status_label.text())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
