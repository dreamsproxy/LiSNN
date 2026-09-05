# LiNN Source Snapshot

This document freezes the donor revision approved for migration planning. It does not authorize or perform the source import.

## Donor

- Repository: `dreamsproxy/LiNN`
- Repository ID: `843133979`
- Default branch: `main`
- Visibility at inspection: private
- Frozen commit: `6e304b3dff413e892c20dca5b6e55653467c07cc`
- Commit message: `Create connectivity.py`
- Commit timestamp: `2025-05-19T18:59:31Z`

## Destination

- Repository: `dreamsproxy/LiSNN`
- Repository ID: `499461389`
- Default branch: `main`
- Planned destination path: `legacy/LiNN/`
- Migration branch: `merge/linn-legacy-import`
- Branch base commit: `f5a25359e8b0e5fd7eb8cab3e3b18fecdd6651b5`

## Lineage note

LiNN was created as an independent reimplementation after the earlier D-LiNN repository was forgotten. D-LiNN was later renamed LiSNN, meaning LiNN and LiSNN represent parallel experimental lineages rather than a normal fork relationship.

The intended migration preserves LiNN as a donor/reference snapshot inside LiSNN. It does not make LiNN the canonical active implementation and does not merge unrelated Git histories.

## Current authorization boundary

Approved:

1. Inventory LiNN.
2. Freeze this donor revision.
3. Create the LiSNN migration branch.

Not approved yet:

- Copying LiNN source into LiSNN
- Copying datasets, outputs, audio, video, or weights
- Editing LiNN algorithms or imports
- Refactoring LiNN into LiSNN modules
- Opening or merging a pull request into `main`

## Source integrity rule

If LiNN changes after the frozen commit, those changes are outside this migration snapshot unless this document is deliberately updated before Phase 4.
