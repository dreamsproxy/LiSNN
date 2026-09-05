"""Tests for the GUI-independent planning graph alpha model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from planning_graph.definitions import DefinitionRegistry
from planning_graph.grid import (
    GRID_SIZE,
    NODE_HEIGHT,
    NODE_WIDTH,
    is_grid_aligned,
    snap_dimension,
    snap_rect_outward,
    snap_value,
)
from planning_graph.models import (
    PlanningDocument,
    PlanningEdge,
    PlanningGroup,
    PlanningNode,
)
from planning_graph.serialization import load_document, save_document


class PlanningGraphModelTests(unittest.TestCase):
    def make_two_node_document(self) -> tuple[PlanningDocument, PlanningNode, PlanningNode]:
        document = PlanningDocument(title="Test Plan")
        first = PlanningNode.create(kind="Idea", title="Candidate mechanism")
        second = PlanningNode.create(kind="Task", title="Run the ablation")
        document.add_node(first)
        document.add_node(second)
        return document, first, second

    def test_json_round_trip_preserves_complete_graph(self) -> None:
        document, first, second = self.make_two_node_document()
        edge = PlanningEdge.create(
            source_id=first.node_id,
            target_id=second.node_id,
            relation="Leads To",
            label="test with",
        )
        edge.route_x = 125.0
        edge.route_y = -75.0
        edge.__post_init__()
        document.add_edge(edge)
        document.add_group(
            PlanningGroup.create(
                title="Investigation",
                node_ids=[first.node_id, second.node_id],
                x=-100.0,
                y=-100.0,
                width=500.0,
                height=300.0,
                color="#3B5B78",
                layer=4,
            )
        )
        decoded = PlanningDocument.from_dict(json.loads(json.dumps(document.to_dict())))
        self.assertEqual(decoded.to_dict(), document.to_dict())

    def test_default_connection_is_leads_to(self) -> None:
        edge = PlanningEdge.create(source_id="N1", target_id="N2")
        self.assertEqual(edge.relation, "Leads To")

    def test_manual_route_requires_both_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            PlanningEdge(
                edge_id="E1",
                source_id="N1",
                target_id="N2",
                route_x=25.0,
                route_y=None,
            )

    def test_manual_route_is_grid_aligned(self) -> None:
        edge = PlanningEdge(
            edge_id="E1",
            source_id="N1",
            target_id="N2",
            route_x=37.0,
            route_y=-38.0,
        )
        self.assertEqual((edge.route_x, edge.route_y), (25.0, -50.0))

    def test_custom_block_kind_is_allowed(self) -> None:
        node = PlanningNode.create(kind="Experiment", title="Ablation 7")
        self.assertEqual(node.kind, "Experiment")

    def test_none_is_default_status(self) -> None:
        node = PlanningNode.create(kind="Idea", title="Candidate")
        self.assertEqual(node.status, "None")

    def test_node_geometry_and_fonts_round_trip(self) -> None:
        node = PlanningNode.create(
            kind="Note",
            title="Readable",
            width=337.0,
            height=211.0,
            header_font_size=9,
            title_font_size=14,
            body_font_size=11,
            footer_font_size=7,
        )
        restored = PlanningNode.from_dict(node.to_dict())
        self.assertEqual(restored.to_dict(), node.to_dict())
        self.assertTrue(is_grid_aligned(node.width / 2.0))
        self.assertTrue(is_grid_aligned(node.height / 2.0))

    def test_dimensions_use_even_grid_cell_counts(self) -> None:
        width = snap_dimension(275.0, 150.0)
        height = snap_dimension(125.0, 100.0)
        self.assertEqual(round(width / GRID_SIZE) % 2, 0)
        self.assertEqual(round(height / GRID_SIZE) % 2, 0)
        self.assertTrue(is_grid_aligned(width / 2.0))
        self.assertTrue(is_grid_aligned(height / 2.0))

    def test_default_dimensions_remain_grid_aligned(self) -> None:
        self.assertTrue(is_grid_aligned(NODE_WIDTH / 2.0))
        self.assertTrue(is_grid_aligned(NODE_HEIGHT / 2.0))

    def test_rejects_dangling_edge(self) -> None:
        payload = {
            "version": 1,
            "title": "Invalid",
            "nodes": [{"node_id": "N1", "kind": "Idea", "title": "Known"}],
            "edges": [
                {
                    "edge_id": "E1",
                    "source_id": "N1",
                    "target_id": "N2",
                    "relation": "Related",
                }
            ],
        }
        with self.assertRaises(ValueError):
            PlanningDocument.from_dict(payload)

    def test_remove_node_cascades_connections(self) -> None:
        document, first, second = self.make_two_node_document()
        edge = PlanningEdge.create(
            source_id=first.node_id,
            target_id=second.node_id,
            relation="Depends On",
        )
        document.add_edge(edge)
        removed = document.remove_node(first.node_id)
        self.assertEqual(removed, [edge.edge_id])
        self.assertFalse(document.edges)

    def test_self_edge_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PlanningEdge.create(source_id="N1", target_id="N1")

    def test_snap_value_uses_grid(self) -> None:
        self.assertEqual(snap_value(36.0), 25.0)
        self.assertEqual(snap_value(38.0), 50.0)
        self.assertEqual(snap_value(-38.0), -50.0)

    def test_snap_rect_expands_outward(self) -> None:
        self.assertEqual(
            snap_rect_outward(1.0, 24.0, 51.0, 76.0),
            (0.0, 0.0, 75.0, 100.0),
        )

    def test_group_rejects_unknown_members(self) -> None:
        document = PlanningDocument()
        first = PlanningNode.create(kind="Idea", title="First")
        document.add_node(first)
        group = PlanningGroup.create(
            title="Invalid",
            node_ids=[first.node_id, "missing"],
            x=0.0,
            y=0.0,
            width=250.0,
            height=150.0,
        )
        with self.assertRaises(ValueError):
            document.add_group(group)

    def test_node_cannot_belong_to_multiple_groups(self) -> None:
        document = PlanningDocument()
        nodes = [PlanningNode.create(kind="Idea", title=f"N{i}") for i in range(3)]
        for node in nodes:
            document.add_node(node)
        document.add_group(
            PlanningGroup.create(
                title="One",
                node_ids=[nodes[0].node_id, nodes[1].node_id],
                x=0.0,
                y=0.0,
                width=250.0,
                height=150.0,
            )
        )
        with self.assertRaises(ValueError):
            document.add_group(
                PlanningGroup.create(
                    title="Two",
                    node_ids=[nodes[1].node_id, nodes[2].node_id],
                    x=0.0,
                    y=0.0,
                    width=250.0,
                    height=150.0,
                )
            )

    def test_removing_member_removes_small_group(self) -> None:
        document, first, second = self.make_two_node_document()
        group = PlanningGroup.create(
            title="Temporary",
            node_ids=[first.node_id, second.node_id],
            x=0.0,
            y=0.0,
            width=250.0,
            height=150.0,
        )
        document.add_group(group)
        document.remove_node(first.node_id)
        self.assertNotIn(group.group_id, document.groups)

    def test_delete_group_and_nodes_cascades(self) -> None:
        document, first, second = self.make_two_node_document()
        group = PlanningGroup.create(
            title="Disposable",
            node_ids=[first.node_id, second.node_id],
            x=0.0,
            y=0.0,
            width=250.0,
            height=150.0,
        )
        document.add_group(group)
        removed = document.remove_group_and_nodes(group.group_id)
        self.assertEqual(set(removed), {first.node_id, second.node_id})
        self.assertFalse(document.nodes)
        self.assertFalse(document.groups)

    def test_legacy_document_without_new_fields_loads(self) -> None:
        payload = {
            "version": 1,
            "title": "Legacy",
            "nodes": [{"node_id": "N1", "kind": "Idea", "title": "Old"}],
            "edges": [],
        }
        document = PlanningDocument.from_dict(payload)
        node = document.nodes["N1"]
        self.assertEqual(document.groups, {})
        self.assertEqual(node.width, NODE_WIDTH)
        self.assertEqual(node.status, "None")

    def test_json_file_save_and_load(self) -> None:
        document = PlanningDocument(title="Persistent")
        document.add_node(PlanningNode.create(kind="Note", title="Remember this"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            save_document(document, path)
            restored = load_document(path)
        self.assertEqual(restored.to_dict(), document.to_dict())

    def test_sqlite_save_and_load_matches_json_model(self) -> None:
        document, first, second = self.make_two_node_document()
        document.add_edge(PlanningEdge.create(source_id=first.node_id, target_id=second.node_id))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.db"
            save_document(document, path)
            restored = load_document(path)
            # Windows refuses this unlink while any sqlite3.Connection remains open.
            path.unlink()
            self.assertFalse(path.exists())
        self.assertEqual(restored.to_dict(), document.to_dict())

    def test_definition_registry_persists_custom_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom_blocks.json"
            registry = DefinitionRegistry.defaults(path)
            registry.save()
            registry.add_block_type("Experiment", "#123ABC")
            restored = DefinitionRegistry.load(path)
        self.assertTrue(restored.has_block_type("Experiment"))
        self.assertEqual(restored.block_color("Experiment"), "#123ABC")
        self.assertIn("None", restored.status_names())


if __name__ == "__main__":
    unittest.main()
