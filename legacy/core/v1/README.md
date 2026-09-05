# Language timecode core v1

This directory freezes the verified absolute-position language timecode baseline.

```text
absolute token position -> stored token identity
```

For a corpus with `T` token positions and vocabulary size `V`, v1 contains `V + T` neurons. Training presents a token and its one-hot absolute position code together. Recall presents only the position code and reconstructs the stored token sequence.

Run the frozen version from the repository root:

```bash
python -m core.v1.main --dataset-dir datasets --max-tokens 256
```

The original v1 outputs are written with `_v1` suffixes by its frozen CLI so they do not overwrite v2 checkpoints or generations.

V1 preserves non-binary LIF spikes, dense recurrent propagation, combined top-k STDP and Hebbian plasticity, clipping, pruning, repeated simulation ticks, exact tokenization, timecode-only recall, evaluation, and checkpoint save/load.

The root `core/` package now contains v2, which replaces absolute position recall with relative-time next-token prediction.
