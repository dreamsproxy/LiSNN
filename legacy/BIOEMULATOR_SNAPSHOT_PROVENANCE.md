# BioEmulator snapshot provenance

## Source

- Repository: `dreamsproxy/BioEmulator`
- Visibility at import: public
- Default branch: `main`
- Frozen commit: `d9d31d04ff860643ec4b3b9650f00718f86af15a`
- Commit message: `Audio and uncommited changes`
- Destination: `legacy/BioEmulator_snapshot`
- Representation: pinned Git submodule/gitlink

## Why the full repository is retained

BioEmulator contains the EMNET lineage under `EMNET/`, multiple historical SNN implementations, Oja-network experiments, diffusion-growth work, audio experiments, datasets, generated assets, and older rewrites. Retaining only the `EMNET/` directory would erase the experimental context and relationships among those variants.

The snapshot preserves the donor repository exactly without copying its large binary assets into LiSNN's object database.

## Intended LiSNN role

EMNET remains an explicit engram and episodic-memory subsystem. It is not an implementation of `CognitiveSubstrate` and must later communicate through a separate memory interface or adapter.

No active LiSNN module imports from this snapshot. No BioEmulator dependency is added to LiSNN's primary environment.

## Licensing note

No top-level `LICENSE` file was found at the frozen donor revision. The snapshot is retained as a repository reference rather than copied into active LiSNN source. Licensing and redistribution terms must be resolved before extracting donor code into the active package.

## Update rule

Advancing this snapshot requires a deliberate donor revision, updated provenance, validation of active-code isolation, and review of any newly introduced assets or dependencies.
