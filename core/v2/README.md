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

For vocabulary size `V`, context length `C`, and optional hidden population `H`, v2 creates:

- `V` token neurons.
- `C` relative-time neurons, from oldest to newest context slot.
- `V * C` conjunctive token-at-slot neurons.
- one prediction-query neuron.
- `H` recurrent-only hidden neurons.

A context such as `A B A` is represented as an ordered trajectory:

```text
A@slot(C-3) -> B@slot(C-2) -> A@slot(C-1) -> QUERY
```

The recurrent field retains the non-binary LIF, dense propagation, STDP, Hebbian plasticity, clipping, pruning, and repeated simulation ticks inherited from v1.

## Recurrent-only hidden neurons

Use either spelling:

```bash
python -m core.main --hidden_neurons 64
python -m core.main --hidden-neurons 64
```

Hidden neurons:

- receive no direct token, timecode, binding, or query input;
- have no direct connection to the probabilistic token readout;
- connect recurrently with every I/O neuron in both directions;
- also participate in hidden-to-hidden recurrence;
- use thresholds shifted 2 mV lower and membrane time constants scaled to 90%, giving them a slightly higher firing rate.

Their effect on prediction must therefore pass through recurrent changes in the observable I/O population.

## Fixed inhibitory/excitatory neuron types

The inhibitory share is controlled by `--ei_ratio` or `--ei-ratio`:

```bash
python -m core.main --ei_ratio 0.5
```

`--ei_ratio 0.5` means approximately 50% inhibitory and 50% excitatory neurons. The default is `0.5`. The ratio is applied separately to the I/O population and hidden population so changing the hidden count does not perturb the requested I/O split.

The recurrent matrix follows the propagation convention `signals = W @ spikes`, so each matrix column contains the outgoing synapses of one presynaptic neuron:

- excitatory-neuron columns are constrained to non-negative weights;
- inhibitory-neuron columns are constrained to non-positive weights.

STDP, Hebbian updates, clipping, normalization, and pruning preserve that sign constraint. A neuron cannot change type during training, and the assignments are stored in checkpoints.

## Controlled hidden-population ablations

I/O and hidden populations use independent deterministic random streams. For a fixed corpus, context length, seed, and E/I ratio, changing only `hidden_neurons` preserves the initial:

- I/O neuron-type assignments;
- I/O LIF parameters and thresholds;
- I/O time constants;
- I/O-to-I/O recurrent weight block.

The H0 and H8 runs therefore differ only by the added hidden population and the recurrent blocks involving it. After training begins, their I/O weights may diverge because the H8 hidden cells alter recurrent activity; that divergence is the effect being measured.

Example controlled pair:

```bash
python -m core.main \
    --dataset-dir datasets \
    --max-tokens 256 \
    --context-length 8 \
    --hidden_neurons 0 \
    --ei_ratio 0.5 \
    --epochs 10 \
    --checkpoint trajectory-ei50-h0.npz \
    --output generation-ei50-h0.txt

python -m core.main \
    --dataset-dir datasets \
    --max-tokens 256 \
    --context-length 8 \
    --hidden_neurons 8 \
    --ei_ratio 0.5 \
    --epochs 10 \
    --checkpoint trajectory-ei50-h8.npz \
    --output generation-ei50-h8.txt
```

## Next-token readout

The next-token target is never injected into the recurrent field. Instead, the accumulated observable I/O trajectory is passed to a probabilistic readout. Hidden-neuron dimensions are excluded from that classifier.

Its update is local in sign:

- Hebbian strengthening for the target token.
- Anti-Hebbian weakening for competing tokens.

A softmax converts the resulting token logits into a probability distribution.

## Training and evaluation

```bash
python -m core.main \
    --dataset-dir datasets \
    --max-tokens 256 \
    --context-length 8 \
    --hidden_neurons 64 \
    --ei_ratio 0.5 \
    --epochs 3
```

The CLI reports the requested inhibitory ratio, separate I/O and hidden inhibitory counts, total neuron count, dense-state estimate, teacher-forced next-token accuracy, mean top-1 confidence, cross-entropy, and perplexity.

It then generates a continuation from either the first context window or an explicit prompt:

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
