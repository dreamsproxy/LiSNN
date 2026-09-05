# LiSNN Planning Graph - U1 Alpha

U1 is a PySide6 graph editor for planning, reasoning, research decomposition, project mapping, and human-readable process specification. U2 will reuse the same document model for LiSNN workflow visualization.

## Launch

From the LiSNN repository root:

```powershell
python3 -m pip install -r planning_graph\requirements.txt
python3 -m unittest discover -s tests -v
python3 main.py
```

The current alpha branch passes 42 tests, including offscreen PySide6 scene and window tests.

Use `python3`, not the Windows `py` launcher. On the current development system, `py` selects free-threaded Python 3.14 by default.

## Alpha interaction summary

- Drag a block type from the left palette onto the canvas.
- Double-click a left-palette block to create it at the center of the visible canvas.
- Click **+ New Block** to create a persistent custom type and accent color.
- Drag from a block port to another block to create a connection.
- New connections default to **Leads To** and display a directional triangle.
- Select and drag a connection to create a persistent manual route.
- Right-click a manually routed connection and choose **Reset Automatic Route**.
- Select a block or group to reveal four grid-snapped resize handles.
- Edit selected object properties in the right panel; changes appear immediately.
- `Ctrl+Z` undoes and `Ctrl+Y` redoes graph mutations.
- Mouse wheel zooms; middle-mouse drag pans.
- Left-drag on empty canvas creates a rubber-band selection.
- `Delete` removes the selection.
- `Ctrl+D` duplicates one selected block.
- `Ctrl+G` groups selected blocks.
- `Ctrl+Shift+G` ungroups selected groups.
- `Ctrl+Shift+I` inverts selection.
- `F` fits the full graph into view.

## Visual language

- Near-black canvas with half-alpha white dots at 25-unit intersections.
- Dark gray block context, blue-gray headers, white primary text, muted context text.
- Left block edge indicates block type color.
- Bottom block edge indicates status color.
- Status **None** uses the normal header color, producing no additional semantic emphasis.
- Blocks and group frames resize in even grid-cell increments so centers, sides, and ports remain aligned.
- Connections are colored by relationship type.
- Automatic connections sharing the same source block and departure side use stable radial lane spacing instead of occupying the same curve.
- Arbitrary graph-wide crossing avoidance is not automatic; drag a connection to place a persistent manual route when a local layout needs further separation.
- Group backdrops use low-alpha configurable colors and independent layers.
- Every color selector shows the rendered color and its `#RRGGBB` code.

## Configurable block types

All block types and statuses are defined in:

```text
planning_graph/custom_blocks.json
```

The built-in block types are:

- Idea
- Goal
- Question
- Task
- Evidence
- Decision
- Constraint
- Result
- Note

The GUI **+ New Block** dialog adds custom definitions to the same master file. Existing and new graph files refer to block types by name, so custom types remain available across application instances.

A block whose kind is not currently registered is still loadable. The inspector displays it through the **Custom...** entry and exposes its literal kind text.

## Block properties

The live block inspector exposes:

- kind or custom kind;
- title;
- status;
- priority;
- tags;
- context/body;
- width and height in grid cells;
- header font size;
- title font size;
- context font size;
- footer font size.

## Directed connection types

- Related
- Supports
- Contradicts
- Depends On
- Leads To
- Part Of
- Refines
- Blocks

Connections store source, target, type, label, weight, and an optional manual route point. When no manual point exists, the scene calculates a radial automatic route.

## Groups and backdrops

A group is persistent membership of two or more blocks. Its backdrop is optional and does not define membership.

Right-clicking a group provides:

- Select Group Members
- Ungroup
- Delete Group and Blocks
- Add Backdrop / Remove Backdrop
- Backdrop Color presets and custom color picker
- Bring to Front
- Bring Forward
- Send Backward
- Send to Back

Ungrouping preserves blocks and connections. **Delete Group and Blocks** removes the group, all member blocks, and all attached connections. Removing a backdrop preserves the group as a selectable dashed frame.

The right inspector edits group title, backdrop visibility, rendered color, layer, width, and height.

## Undo and redo

Graph mutations are stored as validated document snapshots. This covers:

- creation and deletion;
- live property edits;
- movement and resizing;
- grouping and ungrouping;
- backdrop and layer changes;
- connection creation and routing;
- route reset.

Continuous property typing is merged into one undo entry per selected object. Selection changes, zoom, and pan are not document mutations and are not added to history.

## Saving and loading

The default format is one human-readable `.json` file.

The same graph can also be saved as one SQLite file using:

- `.db`
- `.sqlite`
- `.sqlite3`

**Save As** selects either format. **Open** loads either format. JSON and SQLite preserve the same document state:

- blocks and all visual properties;
- connections and manual routes;
- groups, membership, backdrops, colors, geometry, and layers;
- document metadata.

The SQLite format uses structured `metadata`, `nodes`, `edges`, `groups_table`, and `group_nodes` tables. Existing v1 JSON files without groups, dimensions, fonts, statuses, or route points remain loadable through defaults.

## Module layout

```text
main.py                          GUI launcher
planning_graph/custom_blocks.json master block/status definitions
planning_graph/definitions.py    persistent definition registry
planning_graph/models.py         GUI-independent graph model
planning_graph/grid.py           shared snapping and geometry
planning_graph/history.py        undo/redo snapshot commands
planning_graph/theme.py          dark theme and relation colors
planning_graph/serialization.py  JSON and SQLite persistence
planning_graph/items.py          blocks, groups, and routed connections
planning_graph/scene.py          canvas, routing, grouping, context menus
planning_graph/widgets.py        palette, new-block dialog, live inspector
planning_graph/window.py         menus, toolbar, file workflow
planning_graph/app.py            QApplication bootstrap
planning_graph/examples/         example graphs
```

## Scope boundary

U1 alpha does not yet:

- execute a plan;
- infer blocks or relationships automatically;
- visualize live LiSNN neuron or hypothesis activity;
- perform automatic whole-graph layout;
- synchronize with external task systems;
- support nested or overlapping membership groups.

Those remain later features. U1 alpha establishes the visual planning surface and portable graph format first.
