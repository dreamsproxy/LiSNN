# Language timecode core

This package rebuilds the preserved `timecode_frame_baseline` as a language-native associative memory.

The direct mapping is:

```text
video frame                    -> language token
flattened image neuron field   -> vocabulary neuron field
one-hot frame/time code        -> one-hot token-position timecode
recalled 32 x 32 frame         -> recalled token sequence
```

For a corpus with `T` token positions and vocabulary size `V`, the network contains `V + T` neurons. The first `V` neurons represent token identity. The final `T` neurons represent absolute sequence positions. Training presents a token and its position code together for multiple recurrent ticks. Recall presents only a position code and decodes the strongest accumulated token-neuron activity.

## What was preserved

- Non-binary spike magnitudes.
- Dense recurrent propagation.
- Combined top-k STDP and Hebbian plasticity.
- Recurrent state carried across repeated ticks and token positions.
- Periodic clipping and pruning.
- Timecode-only associative recall with accumulated spikes.

## Intentional repairs

The archived frame baseline contained several mechanical behaviors that obscured its intended mechanism. The rebuild keeps the learning architecture while repairing them:

- Timecode neurons now integrate their external code through the same LIF dynamics as token neurons. In the archived loop they were assigned their threshold and then threshold-subtracted to zero.
- LIF state preserves valid negative membrane potentials instead of replacing every negative value with `1e-16`.
- Constant-vector normalization returns zeros instead of dividing by zero.
- Error suppression uses the actual matching indices and suppresses them to zero.
- The Hebbian trace starts at zero and updates only connections from active presynaptic neurons; the archived random Hebbian matrix injected unrelated associations before training.
- Initial timecode connectivity is applied to presynaptic timecode columns, matching the matrix propagation convention.
- STDP time constants broadcast along their intended pre/post axes.
- Weight normalization handles zero-norm rows safely.
- Pruning removes weights by absolute magnitude.

These are semantic repairs of the intended timecode association, not a new learning objective.

## Usage

Place UTF-8 `.txt` files in `datasets/`, then run from the repository root:

```bash
python -m core.main --dataset-dir datasets --max-tokens 256
```

Word-preserving tokenization is also available:

```bash
python -m core.main --tokenizer word --max-tokens 256
```

The command writes:

- `language_timecode_recall.txt`: sequence reconstructed from timecodes only.
- `language_timecode_model.npz`: complete model checkpoint.

## Scalability boundary

This first implementation intentionally retains the baseline's dense recurrent and Hebbian matrices. Memory therefore scales as `O((V + T)^2)`. Two float64 plasticity matrices require approximately:

```text
16 * (V + T)^2 bytes
```

Use short corpus slices while validating behavior. Sparse connectivity, shared/relative timecodes, context-conditioned prediction, and free-running generation belong to the next stage after this parity baseline is measured.
