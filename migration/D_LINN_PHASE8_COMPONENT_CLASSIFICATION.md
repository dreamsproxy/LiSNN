# D-LiNN Phase 8 component classification

Status: Phase 8 source audit completed for the original `dreamsproxy/D-LiNN` lineage.

This document classifies D-LiNN components for migration into LiSNN. It does not import D-LiNN source, change an active runtime, or add a dependency.

## Audited source

- Source repository: `dreamsproxy/D-LiNN`
- Source visibility: private
- Source default branch: `main`
- Audited head: `3628ebd7517253f6906015a5522c70aba8b3a244`
- Head commit: `Minor Testing & Debugging`
- Head date: 2022-01-26
- Initial commit: `a75ed9860808fcbeb9cce5c099efdd5ecd8abbcd`
- Commits after the initial commit: 13
- Repository license file: GNU GPL version 3

D-LiNN is the earlier original lineage. Its most important contribution is not a complete spiking-neuron implementation. It is the architectural attempt to represent a neural system as an explicitly identified, persistable graph of independently addressable neuron state.

## Architectural summary

D-LiNN explored four related ideas:

1. Generate globally unique neuron identifiers.
2. Store each neuron's membrane value and weighted outgoing links as JSON.
3. Maintain a cortex-level manifest containing the neuron identifiers.
4. Explore local multiprocessing and network messaging as ways to execute independently addressable neurons.

The repository stops at proof-of-concept scaffolding. The local and decentralized runtimes are incomplete, and several files contain syntax errors, undefined variables, placeholder functions, or copied external experiments.

## Component classification

| Source component | Original role | Decision | LiSNN treatment |
|---|---|---|---|
| `Data-Pipeline/PoC/linn.py` | Generate neuron IDs, initial membrane values, dense random connectivity, per-neuron JSON, and a cortex manifest | **Adapt concept; rewrite implementation** | Preserve explicit identity, graph-state, and persistence semantics. Do not reuse the procedural implementation directly. |
| `Data-Pipeline/PoC/nodeDB/*.json` | One file per neuron containing `ID`, `mV`, and `Nodes Linked` | **Reference; optional migration reader** | Keep as historical schema. A future importer may read it, but active LiSNN must use consolidated, versioned state rather than one file per neuron. |
| `Data-Pipeline/PoC/nodeDB/cortext_data.json` | Cortex-level list of neuron IDs | **Adapt concept; rename and redesign** | Replace with a versioned network manifest or state envelope. Correct the historical `cortext` naming in new code. |
| `Data-Pipeline/PoC/local/cortex.py` | Intended local process coordinator | **Retain role; rewrite** | The cortex/orchestrator boundary remains useful, but execution should coordinate populations or shards rather than one operating-system process per neuron. |
| `Data-Pipeline/PoC/local/node.py` | Intended neuron worker with load, parse, input, processing, and output boundaries | **Reference only** | Preserve the conceptual lifecycle when defining runtime interfaces. Do not import the incomplete worker code. |
| `Data-Pipeline/PoC/local/optimizers.py` | Standalone Adam implementation | **Reject** | It is unrelated to the current local-plasticity path and contains an incorrect second-moment bias update. Use tested framework or project-native optimization code where required. |
| `Data-Pipeline/PoC/decentralized/*` | Early ZeroMQ cortex/node experiment | **Retain architecture note; reject code** | Distributed execution is out of the current core scope. Any later backend should shard populations, use typed messages, and avoid raw Python-object transport. |
| `Data-Pipeline/Testing/ZMQ Testing/*` | PUB/SUB, RADIO/DISH, UDP, and request/reply experiments | **Reject** | These are incomplete transport tests, not LiSNN algorithms. |
| `Data-Pipeline/Testing/mp1.py`, `mp2.py` | Minimal multiprocessing pipe test | **Reject** | Standard library demonstration only. |
| `Data-Pipeline/PoC/hebb.py` | Elementary Hebbian truth-table demonstration | **Reject** | It contributes no behavior beyond existing LiSNN Hebbian/STDP work. |
| `BindsNET_TEST/main.py` | Copied BindsNET Conv1D MNIST example | **Do not import** | Third-party reference code is explicitly identified as copied. It must remain isolated from LiSNN migration work. |
| Generated JSON and miscellaneous JSON tests | Debugging and generated sample state | **Retain only in source snapshot** | Do not copy into active LiSNN. |
| D-LiNN `requirements.txt` | 2021 environment pins including `pyzmq`, `numpy`, and `key-generator` | **Reject** | Do not promote these pins or add ZeroMQ/key-generator to LiSNN core dependencies. |

## Source-to-target semantic map

| D-LiNN field or role | LiSNN equivalent |
|---|---|
| Neuron `ID` | Optional stable neuron ID, population ID, or hypothesis ID, depending on substrate granularity |
| `mV` | Membrane-state vector entry |
| `Nodes Linked` | Dense or sparse connectivity representation |
| Per-neuron JSON file | Entry inside one versioned substrate snapshot |
| Cortex ID list | Network manifest and schema metadata |
| Cortex coordinator | Substrate runtime or execution backend |
| Node input/process/output boundary | Tick lifecycle and substrate `step()` implementation |
| Save/load intent | `CognitiveSubstrate.export_state()` and `import_state()` plus a durable state-store layer |

## Integration decision

D-LiNN should not become a functional substrate adapter. It does not contain a complete dynamical model with a stable `step()` behavior to adapt.

Its viable contribution is a persistence and topology specification layered below current substrates:

```text
CognitiveSubstrate
    -> export_state() / import_state()
    -> versioned state envelope
    -> durable state store
```

The current `LiSNNHypothesisSubstrate` already exports and imports complete runtime state. D-LiNN therefore supplies the missing historical design intent for making that state durable and explicitly identifiable, not a replacement neuron implementation.

## Recommended state envelope

A future serializer should use one atomic, versioned snapshot rather than one JSON file per neuron.

```json
{
  "schema_version": 1,
  "substrate_type": "lisnn_hypothesis",
  "created_at": "ISO-8601 timestamp",
  "config": {},
  "manifest": {
    "population_ids": [],
    "neuron_ids": null
  },
  "state": {},
  "connectivity": {
    "format": "dense-or-sparse",
    "payload": null
  },
  "integrity": {
    "algorithm": "sha256",
    "digest": null
  }
}
```

The exact format should remain substrate-neutral. Large arrays should later support a binary representation while keeping a small JSON manifest.

## Explicit rejections

The following D-LiNN decisions must not be carried into the active LiSNN runtime:

- one operating-system process per neuron;
- one mutable file per neuron;
- fully connected JSON adjacency for every neuron;
- arbitrary globally unique strings as the only array index;
- direct dependency on ZeroMQ in the core package;
- `send_pyobj` or other unsafe language-object transport;
- copied BindsNET example code;
- untested custom Adam code;
- generated sample state committed as active model state.

## Proposed follow-on commits

### Commit A: preserve exact D-LiNN provenance

Add D-LiNN as a pinned private submodule under `legacy/D-LiNN_snapshot`, with provenance and validation documentation equivalent to the existing LiNN snapshot process.

### Commit B: add a durable substrate state store

Add a small storage interface around existing substrate `export_state()` and `import_state()` methods. The first backend should provide atomic local-file writes, schema versioning, validation, and deterministic round-trip tests.

### Commit C: add optional D-LiNN schema inspection

Add an offline migration utility that can inspect a D-LiNN `nodeDB` directory and convert its graph into the new manifest/connectivity envelope. It should not be imported by active substrate modules.

### Commit D: evaluate graph identity requirements

Determine whether stable identity is needed at the individual-neuron level, population level, or only at the cognitive hypothesis level. Do not add per-neuron UUID overhead until a benchmark requires it.

## Phase 8 outcome

D-LiNN is classified as:

- **historically authoritative for explicit neural graph identity and persistence intent**;
- **not suitable as an executable LiSNN substrate**;
- **not suitable for direct source merging**;
- **valuable as provenance and as design input for a versioned state-store layer**.

No active LiSNN module should import directly from D-LiNN or a future D-LiNN legacy snapshot.