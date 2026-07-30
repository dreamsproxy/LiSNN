# Language trajectory core (v2)

The root `core/` package is LiSNN's second language baseline:

```text
recent token-time trajectory -> probable next token
```

The verified absolute-position reconstruction baseline is frozen under `core/v1/` and remains runnable with:

```bash
python -m core.v1.main --dataset-dir datasets --max-tokens 256
```

## Architecture

For vocabulary size `V` and context length `C`, v2 creates:

- `V` token neurons.
- `C` relative-time neurons, from oldest to newest context slot.
- `V * C` conjunctive token-at-slot neurons.
- one prediction-query neuron.

A context such as `A B A` is represented as an ordered trajectory:

```text
A@slot(C-3) -> B@slot(C-2) -> A@slot(C-1) -> QUERY
```

The recurrent field retains the non-binary LIF, dense propagation, STDP, Hebbian plasticity, clipping, pruning, and repeated simulation ticks inherited from v1.

The next-token target is never injected into that recurrent field. Instead, the accumulated context trajectory is passed to a probabilistic readout. Its update is local in sign:

- Hebbian strengthening for the target token.
- Anti-Hebbian weakening for competing tokens.

A softmax converts the resulting token logits into a probability distribution.

## Training and evaluation

```bash
python -m core.main \
    --dataset-dir datasets \
    --max-tokens 256 \
    --context-length 8 \
    --epochs 3
```

The CLI reports teacher-forced next-token accuracy, mean top-1 confidence, cross-entropy, and perplexity. It then generates a continuation from either the first context window or an explicit prompt:

```bash
python -m core.main \
    --dataset-dir datasets \
    --prompt "The Seed" \
    --generate-tokens 128
```

Use `--sample` to draw from the predicted distribution instead of greedy argmax decoding.

Outputs:

- `language_next_token_generation.txt`
- `language_trajectory_model.npz`

## Meaning of the version boundary

`core/v1/` answers: "Which stored token belongs to this absolute position?"

The root v2 answers: "Given these recent tokens and their relative ordering, which token is most probable next?"

V2 no longer needs one neuron for every corpus position. Its time representation is reused across all windows. The current conjunctive binding bank still scales with `V * C`; sparse or distributed bindings are a later optimization, not silently mixed into this behavioral milestone.
