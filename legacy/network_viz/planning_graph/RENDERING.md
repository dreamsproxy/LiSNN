# U1 Rendering and Performance

The planning graph uses the existing `QGraphicsScene` document and interaction model with an adaptive rendering layer applied at application startup.

## Backends

Default launch:

```powershell
python3 main.py
```

This requests a `QOpenGLWidget` viewport. The status bar reports:

```text
Renderer: OpenGL | LOD: Balanced
```

Use the software raster viewport when testing driver problems, remote desktop sessions, or platform-specific OpenGL issues:

```powershell
python3 main.py --software-rendering
```

OpenGL initialization failures retain a usable software viewport and expose the fallback in the status-bar indicator.

## Rendering profiles

Balanced mode is the default. Toggle **View -> Performance Mode** or press `Ctrl+Shift+P` for more aggressive culling. Start directly in that mode with:

```powershell
python3 main.py --performance-mode
```

The toggle changes only painting thresholds. It does not alter graph content, coordinates, snapping, selection state, JSON, or SQLite data.

## Level-of-detail tiers

### Balanced profile

| Effective zoom | Block rendering |
|---:|---|
| `>= 0.70` | Full header, title, body, footer, ports, and handles |
| `0.40-0.70` | Header, title, semantic strips; no body/footer |
| `0.20-0.40` | Title and semantic silhouette |
| `< 0.20` | Block silhouette only |

Performance mode raises these thresholds to `0.90`, `0.52`, and `0.27`.

Group labels, connection labels, arrows, ports, block handles, group handles, and route handles use their own centralized readability thresholds in `planning_graph/performance.py`.

## Adaptive grid

Grid snapping always remains at 25 scene units. Only visual density changes:

| Balanced effective zoom | Visible dots |
|---:|---|
| `>= 0.75` | every 25 units |
| `0.42-0.75` | every 100 units |
| `0.24-0.42` | every 250 units |
| `< 0.24` | hidden |

A hard point-count limit automatically increases visual spacing when an unusually large visible rectangle would still produce too many dots.

## Connection rendering

- Full and compact tiers retain routed quadratic paths.
- Overview silhouettes use inexpensive straight segments.
- Direction arrows remain visible above the configured arrow threshold.
- Labels are skipped until their projected screen size is readable.
- Manual route coordinates remain unchanged at every tier.

## Interaction geometry

Low-detail painting does not change serialized geometry. Blocks, connections, and groups remain selectable through their normal scene shapes. Ports and resize/route handles are painted only at working zoom levels so overview rendering does not spend time drawing unusable controls.

## Central configuration

All thresholds and point limits are defined by:

```text
planning_graph/performance.py
```

The two profiles are:

```text
BALANCED_PROFILE
PERFORMANCE_PROFILE
```

This keeps performance tuning separate from graph models, persistence, and editing behavior.
