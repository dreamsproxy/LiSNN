# LiNN Migration Inventory

Status: Phase 1 complete for planning purposes. No LiNN source has been imported into LiSNN.

## Source repository

- Repository: `dreamsproxy/LiNN`
- Default branch: `main`
- Visibility: private
- Approximate repository size reported by GitHub: 42,862 KB
- Current inspected head: `6e304b3dff413e892c20dca5b6e55653467c07cc`

## Known source groups

### Root-level recurrent/Oja experiments

| Path | Classification | Planned disposition | Notes |
|---|---|---|---|
| `README.md` | Documentation | Import | Minimal project description. |
| `OjaNet.py` | Source | Import | Recurrent Oja-style network with input, recurrent, and readout matrices. |
| `activations.py` | Source | Import | Numba-compiled activation functions and derivatives. |
| `data_loader.py` | Source | Import | Image/MNIST and audio/CQT data loading. |

### Spatial/emergence experiments

| Path | Classification | Planned disposition | Notes |
|---|---|---|---|
| `Emergence/connectivity.py` | Source | Import | Union-find, A* pathfinding, and connected-path analysis. |
| `Emergence/*` | Source and generated artifacts | Review before import | Contains spatial reservoir/emergence PoCs, including local propagation, refractory state, stability, and path analysis. Exact file list must be enumerated before Phase 4. |
| `space-time-constant.py` | Source | Import | Historical experiment identified from commit history; exact current path must be verified before copying. |
| `vision.py` | Source | Import | Historical vision experiment identified from commit history. |
| `vision_test.py` | Source | Import | Historical vision test identified from commit history. |

## Known dependencies

- `numpy`
- `numba`
- `opencv-python`
- `tqdm`
- `matplotlib`
- `librosa`
- Python standard library modules including `glob`, `collections`, `random`, and `heapq`

## Exclusion candidates requiring review

The following classes must not be copied automatically during Phase 4:

- Generated sample frames and rendered image sequences
- Training outputs and plots
- Audio/video assets
- Dataset directories
- Weight arrays and checkpoints
- Cache directories such as `__pycache__`
- Virtual environments and IDE-local settings
- Machine-local paths or environment-specific files

## Current classification policy

| Class | Default action |
|---|---|
| Python source and text documentation | Import under `legacy/LiNN/` |
| Small assets required to understand or execute a PoC | Review individually |
| Large binary assets | No automatic import |
| Generated outputs | Exclude unless they provide unique evidence |
| Caches, environments, temporary files | Exclude |

## Inventory limitation

The available GitHub connector exposes repository metadata, commit history, known file fetches, and code search, but did not return a complete recursive tree for this private repository. Therefore this Phase 1 inventory is a planning inventory rather than a byte-complete manifest.

Before Phase 4, the source tree must be enumerated exactly and converted into a final path-by-path import/exclusion manifest. No donor code should be copied until that final enumeration is reviewed.
