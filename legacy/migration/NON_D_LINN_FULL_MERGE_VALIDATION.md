# Non-D-LiNN full merge validation

Status: complete.

## Merge roster

The complete external donor roster for this consolidation is:

1. `dreamsproxy/LiNN` -> `legacy/LiNN_snapshot`
2. `dreamsproxy/BioEmulator` -> `legacy/BioEmulator_snapshot`
3. `dreamsproxy/VolumetricSNN` -> `legacy/VolumetricSNN_snapshot`

D-LiNN is explicitly excluded. PySES is unrelated. MycoNet has no repository payload at the time of consolidation.

## Frozen revisions

| Donor | Commit |
|---|---|
| LiNN | `6e304b3dff413e892c20dca5b6e55653467c07cc` |
| BioEmulator | `d9d31d04ff860643ec4b3b9650f00718f86af15a` |
| VolumetricSNN | `2f1b7c7d0d0b8e1e87d2652212dacd3d1e893c1d` |

## Fidelity

PASS.

Each destination is a Git submodule whose gitlink points directly to the frozen donor commit. No donor file is decoded, filtered, renamed, normalized, or rewritten. Donor history and binary assets remain in their source repositories.

## Isolation

PASS.

- Active LiSNN source is unchanged.
- `timecode_frame_baseline/` is unchanged.
- `substrates/` and `cognitive/` are unchanged.
- The existing `pytest.ini` excludes `legacy` from recursive collection.
- No active module imports from a donor snapshot.
- Donor dependencies are not promoted to LiSNN requirements.

## Resulting architecture

```text
LiSNN active repository
+-- timecode_frame_baseline/       working temporal-association baseline
+-- cognitive/                     shared cognitive contracts
+-- substrates/                    active hypothesis substrates
+-- legacy/
    +-- LiNN_snapshot              continuous/liquid donor
    +-- BioEmulator_snapshot       EMNET and historical neural variants
    +-- VolumetricSNN_snapshot     spatial/volumetric donor
```

The merge is source-complete but behaviorally isolated. Future integration must extract or rewrite selected components behind explicit interfaces rather than import donor internals directly.

## Next development boundary

The repository is now ready for a new active component based on the working timecode system. That work should begin outside `legacy/` and treat the snapshots as references only.
