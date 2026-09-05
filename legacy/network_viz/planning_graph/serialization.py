"""JSON and SQLite persistence for planning graph documents."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3

from .models import PlanningDocument, PlanningEdge, PlanningGroup, PlanningNode


SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def save_document(document: PlanningDocument, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.touch()
    if destination.suffix.lower() in SQLITE_SUFFIXES:
        return _save_sqlite(document, destination)
    return _save_json(document, destination)


def load_document(path: str | Path) -> PlanningDocument:
    source = Path(path)
    if source.suffix.lower() in SQLITE_SUFFIXES:
        return _load_sqlite(source)
    return _load_json(source)


def _save_json(document: PlanningDocument, destination: Path) -> Path:
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(document.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return destination


def _load_json(source: Path) -> PlanningDocument:
    with source.open("r", encoding="utf-8") as handle:
        return PlanningDocument.from_dict(json.load(handle))


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL,
            tags_json TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            width REAL NOT NULL,
            height REAL NOT NULL,
            header_font_size INTEGER NOT NULL,
            title_font_size INTEGER NOT NULL,
            body_font_size INTEGER NOT NULL,
            footer_font_size INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS edges (
            edge_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            target_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            relation TEXT NOT NULL,
            label TEXT NOT NULL,
            weight REAL NOT NULL,
            route_x REAL,
            route_y REAL
        );
        CREATE TABLE IF NOT EXISTS groups_table (
            group_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            backdrop INTEGER NOT NULL,
            color TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            width REAL NOT NULL,
            height REAL NOT NULL,
            layer INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS group_nodes (
            group_id TEXT NOT NULL REFERENCES groups_table(group_id) ON DELETE CASCADE,
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            member_order INTEGER NOT NULL,
            PRIMARY KEY (group_id, node_id)
        );
        """
    )


def _save_sqlite(document: PlanningDocument, destination: Path) -> Path:
    # sqlite3.Connection.__enter__ manages commit/rollback but deliberately does
    # not close the connection. Use closing() as the outer lifetime boundary so
    # Windows releases the database file before this function returns.
    with closing(_connect(destination)) as connection:
        with connection:
            _create_schema(connection)
            connection.executescript(
                """
                DELETE FROM group_nodes;
                DELETE FROM edges;
                DELETE FROM groups_table;
                DELETE FROM nodes;
                DELETE FROM metadata;
                """
            )
            metadata = {
                "version": str(document.version),
                "title": document.title,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                metadata.items(),
            )
            connection.executemany(
                """
                INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        node.node_id,
                        node.kind,
                        node.title,
                        node.body,
                        node.status,
                        node.priority,
                        json.dumps(node.tags, ensure_ascii=False),
                        node.x,
                        node.y,
                        node.width,
                        node.height,
                        node.header_font_size,
                        node.title_font_size,
                        node.body_font_size,
                        node.footer_font_size,
                    )
                    for node in document.nodes.values()
                ],
            )
            connection.executemany(
                "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        edge.edge_id,
                        edge.source_id,
                        edge.target_id,
                        edge.relation,
                        edge.label,
                        edge.weight,
                        edge.route_x,
                        edge.route_y,
                    )
                    for edge in document.edges.values()
                ],
            )
            connection.executemany(
                "INSERT INTO groups_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        group.group_id,
                        group.title,
                        int(group.backdrop),
                        group.color,
                        group.x,
                        group.y,
                        group.width,
                        group.height,
                        group.layer,
                    )
                    for group in document.groups.values()
                ],
            )
            connection.executemany(
                "INSERT INTO group_nodes VALUES (?, ?, ?)",
                [
                    (group.group_id, node_id, index)
                    for group in document.groups.values()
                    for index, node_id in enumerate(group.node_ids)
                ],
            )
    return destination


def _load_sqlite(source: Path) -> PlanningDocument:
    # Explicitly close before returning. This is required on Windows, where an
    # open SQLite handle prevents TemporaryDirectory cleanup and file replacement.
    with closing(_connect(source)) as connection:
        _create_schema(connection)
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        nodes = {
            row[0]: PlanningNode(
                node_id=row[0],
                kind=row[1],
                title=row[2],
                body=row[3],
                status=row[4],
                priority=row[5],
                tags=json.loads(row[6]),
                x=row[7],
                y=row[8],
                width=row[9],
                height=row[10],
                header_font_size=row[11],
                title_font_size=row[12],
                body_font_size=row[13],
                footer_font_size=row[14],
            )
            for row in connection.execute(
                """
                SELECT node_id, kind, title, body, status, priority, tags_json,
                       x, y, width, height, header_font_size, title_font_size,
                       body_font_size, footer_font_size
                FROM nodes ORDER BY rowid
                """
            )
        }
        edges = {
            row[0]: PlanningEdge(
                edge_id=row[0],
                source_id=row[1],
                target_id=row[2],
                relation=row[3],
                label=row[4],
                weight=row[5],
                route_x=row[6],
                route_y=row[7],
            )
            for row in connection.execute(
                """
                SELECT edge_id, source_id, target_id, relation, label, weight,
                       route_x, route_y FROM edges ORDER BY rowid
                """
            )
        }
        group_members: dict[str, list[str]] = {}
        for group_id, node_id in connection.execute(
            "SELECT group_id, node_id FROM group_nodes ORDER BY group_id, member_order"
        ):
            group_members.setdefault(group_id, []).append(node_id)
        groups = {
            row[0]: PlanningGroup(
                group_id=row[0],
                title=row[1],
                node_ids=group_members.get(row[0], []),
                backdrop=bool(row[2]),
                color=row[3],
                x=row[4],
                y=row[5],
                width=row[6],
                height=row[7],
                layer=row[8],
            )
            for row in connection.execute(
                """
                SELECT group_id, title, backdrop, color, x, y, width, height, layer
                FROM groups_table ORDER BY rowid
                """
            )
        }

    if not metadata:
        raise ValueError("SQLite file does not contain a planning graph document")
    return PlanningDocument(
        title=metadata.get("title", "Untitled Planning Graph"),
        nodes=nodes,
        edges=edges,
        groups=groups,
        version=int(metadata.get("version", 1)),
        created_at=metadata.get("created_at", ""),
        updated_at=metadata.get("updated_at", ""),
    )
