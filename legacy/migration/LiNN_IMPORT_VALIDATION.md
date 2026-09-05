# LiNN import validation

Status: Phases 4-7 completed on `merge/linn-snapshot-import` using a pinned Git submodule.

## Validated source

- Donor repository: `dreamsproxy/LiNN`
- Frozen commit: `6e304b3dff413e892c20dca5b6e55653467c07cc`
- Donor commit message: `Create connectivity.py`
- Donor repository ID: `843133979`

## Destination representation

- Destination repository: `dreamsproxy/LiSNN`
- Path: `legacy/LiNN_snapshot`
- Representation: Git submodule/gitlink
- `.gitmodules` URL: `https://github.com/dreamsproxy/LiNN.git`

## Fidelity result

PASS.

The destination gitlink points directly to the frozen donor commit. No LiNN file was decoded, rewritten, renamed, normalized, filtered, or copied into LiSNN's object database. Therefore the checked-out submodule tree is the exact LiNN repository state at the frozen commit, including source, documentation, binary assets, and commit history reachable from that revision.

## Isolation result

PASS with documented private-repository requirement.

- Active LiSNN source files are unchanged.
- `timecode_frame_baseline/` is unchanged.
- No active LiSNN module imports from `legacy/LiNN_snapshot`.
- A root `pytest.ini` excludes `legacy` from recursive test collection.
- LiNN dependencies remain inside the donor repository and are not added to LiSNN's primary dependency declaration.
- LiNN top-level scripts, hardcoded paths, and output behavior remain confined to the submodule.

## Checkout caveat

LiNN is private. A user must authenticate with credentials that can read both LiSNN and LiNN before running:

```bash
git submodule update --init --recursive
```

A clone without those permissions will retain the gitlink metadata but cannot populate the submodule working tree.

## Repository-size result

PASS.

LiSNN stores only the gitlink, `.gitmodules`, and migration documentation. LiNN binary assets such as `The Fool.wav`, `sample.png`, and `Experiments/checkerboard.png` are not duplicated into the LiSNN object database.

## Scope boundary

No Phase 8 component classification, shared-interface design, adapter extraction, behavioral rewrite, or hybrid-network work was performed.

## Future update rule

Advancing `legacy/LiNN_snapshot` to another LiNN commit requires:

1. A deliberate source-revision decision.
2. An updated provenance document.
3. A new validation report.
4. Review that no active LiSNN code begins depending directly on the legacy tree.
