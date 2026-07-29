# Legacy and donor implementations

Repositories under `legacy/` are preserved as pinned Git submodules for complete provenance, reproducibility, and later component extraction. They are not part of the active LiSNN API and must not be imported by production modules or collected by the default test suite.

## Consolidated roster

| Snapshot | Donor | Frozen commit | Role | Active API |
|---|---|---|---|---|
| `legacy/LiNN_snapshot` | `dreamsproxy/LiNN` | `6e304b3dff413e892c20dca5b6e55653467c07cc` | continuous/liquid reference and ablation lineage | excluded |
| `legacy/BioEmulator_snapshot` | `dreamsproxy/BioEmulator` | `d9d31d04ff860643ec4b3b9650f00718f86af15a` | EMNET, historical SNN variants, and bio-inspired experiments | excluded |
| `legacy/VolumetricSNN_snapshot` | `dreamsproxy/VolumetricSNN` | `2f1b7c7d0d0b8e1e87d2652212dacd3d1e893c1d` | spatial and volumetric reconstruction experiments | excluded |

D-LiNN is deliberately excluded from the merge roster. PySES is unrelated to LiSNN consolidation. MycoNet currently contains no repository payload to preserve.

## Checkout

```bash
git clone --recurse-submodules https://github.com/dreamsproxy/LiSNN.git
```

For an existing clone:

```bash
git submodule update --init --recursive
```

`LiNN_snapshot` is private and therefore requires GitHub credentials with access to both LiSNN and LiNN. BioEmulator and VolumetricSNN are public.

Each gitlink pins an exact donor commit. Changes to a donor default branch do not update LiSNN automatically.

## Isolation policy

- Active LiSNN code must not import from any `legacy/*_snapshot` path.
- Donor scripts may contain top-level execution, hardcoded paths, incomplete experiments, old environments, large binary assets, and generated outputs.
- Donor dependencies are not promoted into LiSNN's core requirements.
- Behavioral extraction must occur through a reviewed adapter or rewrite outside the snapshots.
- Snapshot revisions may advance only through an explicit provenance update and validation commit.
