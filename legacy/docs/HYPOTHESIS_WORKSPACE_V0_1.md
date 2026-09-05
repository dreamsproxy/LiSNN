# LiSNN Hypothesis Workspace v0.1

## Objective

Build the smallest LiSNN system that reproduces one recognizable fragment of Alan's research behavior:

> Maintain several competing explanations for an observed failure, preserve them over time, and change their relative priority when new evidence arrives.

This milestone is intentionally narrower than a synthetic Alan. It establishes the first falsifiable cognitive mechanism required for one.

## Initial Domain

Use historical AFGAN critic-diagnosis episodes because they provide:

- explicit observations;
- competing mechanistic explanations;
- sequential experiments;
- numerical and visual outcomes;
- repeated belief revision;
- both publication-constrained and personal-use branches.

A suitable first episode is the OEP instability investigation.

```text
Observation:
OEP remains numerically stable near 1, but visible artifacts continue.

Candidate hypotheses:
H1: OEP measures the wrong behavior.
H2: OEP transformations are mutually inconsistent.
H3: OEP interacts badly with mixed-batch LP/LV regularization.
H4: The generator architecture is the primary cause and OEP is incidental.
```

Events arrive sequentially:

```text
t0: OEP remains near 1.
t1: Artifacts remain.
t2: Disabling RGB skip does not remove the artifacts.
t3: Rotation and flip assumptions are found to be inconsistent.
t4: Removing rotations changes training behavior.
```

The system must maintain and update the competing hypotheses without collapsing them into one undifferentiated state.

## Phase 1: Structured Research Episodes

Create 20 manually curated research episodes in YAML or JSON before attempting large-scale training.

Example:

```yaml
episode_id: afgan_oep_001
project: AFGAN

hypotheses:
  - id: H1
    text: OEP is not diagnostically meaningful
  - id: H2
    text: Rotation and flip assumptions are inconsistent
  - id: H3
    text: Mixed-batch regularizer interaction causes instability
  - id: H4
    text: Generator architecture is the primary cause

events:
  - step: 0
    observation: OEP remains near 1
    supports: [H1]
    weakens: []

  - step: 1
    observation: Removing RGB skip does not remove artifacts
    supports: [H1, H2, H3]
    weakens: [H4]

  - step: 2
    observation: OEP transformations were not composed consistently
    supports: [H2]
    weakens: []

  - step: 3
    observation: Removing rotations improves stability
    supports: [H2]
    weakens: [H4]

expected_final_ranking:
  - H2
  - H3
  - H1
  - H4
```

The first dataset should prioritize conceptual clarity over scale.

## Phase 2: Rule-Based Baseline

Implement a simple explicit hypothesis tracker before LiSNN.

```python
class HypothesisTracker:
    def __init__(self, hypothesis_ids):
        self.scores = {hypothesis_id: 0.0 for hypothesis_id in hypothesis_ids}

    def apply_event(
        self,
        supports,
        weakens,
        support_gain=1.0,
        weaken_gain=1.0,
    ):
        for hypothesis_id in supports:
            self.scores[hypothesis_id] += support_gain

        for hypothesis_id in weakens:
            self.scores[hypothesis_id] -= weaken_gain
```

This baseline is intentionally limited. LiSNN must later justify its complexity through capabilities the rule table handles poorly:

- delayed evidence;
- partial forgetting;
- contextual recall;
- nonlinear accumulation;
- reactivation after long gaps;
- coexistence of competing hypotheses;
- interaction among related hypotheses.

Without a baseline, interesting spike patterns cannot be distinguished from useful cognitive behavior.

## Phase 3: Common Substrate Interface

Define a common interface before integrating specific neuron implementations.

```python
from dataclasses import dataclass
from typing import Mapping


@dataclass
class CognitiveEvent:
    features: Mapping[str, float]
    context_id: str
    timestamp: int


@dataclass
class HypothesisState:
    activation: float
    confidence: float
    persistence: float


class CognitiveSubstrate:
    def reset(self) -> None:
        raise NotImplementedError

    def register_hypotheses(self, hypothesis_ids: list[str]) -> None:
        raise NotImplementedError

    def step(self, event: CognitiveEvent) -> dict[str, HypothesisState]:
        raise NotImplementedError

    def consolidate(self) -> None:
        raise NotImplementedError

    def export_state(self) -> dict:
        raise NotImplementedError

    def import_state(self, state: dict) -> None:
        raise NotImplementedError
```

Implement this contract for:

- `RuleBaseline`;
- `GRUBaseline` or `LSTMBaseline`;
- `LiNNSubstrate`;
- `LiSNNSubstrate`.

EMNET should later implement a separate episodic-memory interface.

## Phase 4: Minimal LiSNN Workspace

Do not begin with thousands of neurons.

Suggested initial topology:

```text
4 hypothesis populations
16-32 neurons per hypothesis population
1 evidence input population
1 context population
1 inhibitory competition population
1 slow-persistence population
```

Expected total size: approximately 100-200 neurons.

Each hypothesis population should contain:

- fast activation neurons;
- slow persistence neurons;
- local recurrent excitation;
- lateral inhibition against competing hypotheses;
- context-sensitive input pathways.

The first required behavior is:

```text
Evidence for H2 arrives.
-> H2 becomes active.
-> H2 remains recoverable after the direct input disappears.
-> Contradictory evidence weakens H2 without erasing it instantly.
-> A related cue can reactivate H2 later.
-> H1 and H3 may coexist when evidence remains ambiguous.
```

## Phase 5: Explicit Cognitive Modes

Add externally controllable divergence and convergence modulators.

```python
mode = {
    "divergence": 0.0,
    "convergence": 0.0,
    "incubation": 0.0,
}
```

### Divergence

- lower lateral inhibition;
- reduce activation thresholds;
- increase weak associative excitation;
- broaden contextual retrieval;
- preserve low-confidence alternatives.

### Convergence

- increase evidence weighting;
- strengthen lateral inhibition;
- reduce unsupported persistence;
- sharpen hypothesis ranking.

### Incubation

- reduce direct external drive;
- replay unresolved hypotheses;
- consolidate delayed associations;
- decay irrelevant activation.

Mode switching should initially remain explicit and inspectable rather than learned.

## Phase 6: Metrics

### Ranking accuracy

Does the final hypothesis ordering match the expected ordering?

### Persistence

Can a supported hypothesis remain recoverable after 50-500 unrelated steps?

### Contradiction handling

Does contradictory evidence reduce confidence without instantly deleting the hypothesis?

### Multi-hypothesis retention

Can two or more incompatible explanations remain simultaneously recoverable?

### Reactivation

Can a dormant hypothesis return when a related cue appears?

### Context separation

Can the system distinguish similar instability concepts belonging to different projects or mechanisms?

### Mode sensitivity

Does divergence broaden the active hypothesis set, while convergence sharpens selection using the same evidence?

### Baseline advantage

Does LiSNN provide measurable benefits over rule-based and conventional recurrent baselines?

## Required Logging

Record at every evaluation step:

```text
population firing rates
membrane potentials
adaptation currents
synaptic weights
inhibitory activity
mode values
hypothesis activation
hypothesis confidence
memory retrieval strength
context identifier
event identifier
```

Required plots:

- hypothesis activation over time;
- confidence over time;
- inhibitory activity over time;
- event markers;
- mode transitions;
- final ranking comparison across substrates.

The internal trajectory should remain interpretable enough to identify when a hypothesis persisted, weakened, reactivated, or was suppressed.

## Proposed Repository Layout

```text
LiSNN/
+-- cognitive/
|   +-- schemas.py
|   +-- interfaces.py
|   +-- events.py
|   +-- modes.py
|
+-- substrates/
|   +-- rule_baseline.py
|   +-- gru_baseline.py
|   +-- linn_adapter.py
|   +-- lisnn_adapter.py
|
+-- memory/
|   +-- interface.py
|   +-- episodic_store.py
|   +-- emnet_adapter.py
|
+-- benchmarks/
|   +-- hypothesis_tracking/
|       +-- episodes/
|       |   +-- afgan_oep_001.yaml
|       +-- evaluator.py
|       +-- metrics.py
|       +-- run_benchmark.py
|
+-- visualization/
|   +-- plot_activity.py
|   +-- plot_hypotheses.py
|
+-- tests/
    +-- test_interfaces.py
    +-- test_persistence.py
    +-- test_reactivation.py
    +-- test_context_separation.py
```

This is a target structure, not an instruction to reorganize existing code prematurely. Adapters should be preferred where legacy implementations already exist.

## First Three Implementation Commits

### Commit 1

```text
Add cognitive substrate interfaces and hypothesis event schema
```

Contains:

- dataclasses;
- interfaces;
- event serialization;
- mode definitions;
- interface tests.

### Commit 2

```text
Add rule-based hypothesis tracking baseline and evaluator
```

Contains:

- YAML episode loader;
- rule baseline;
- ranking metric;
- persistence metric;
- initial AFGAN episodes;
- baseline tests.

### Commit 3

```text
Add minimal LiSNN hypothesis population prototype
```

Contains:

- four hypothesis populations;
- evidence injection;
- recurrent persistence;
- lateral inhibition;
- activity logging;
- comparison against the rule baseline.

## Definition of Done

LiSNN Hypothesis Workspace v0.1 succeeds when it can process at least 20 structured research episodes and demonstrate:

1. persistent competing hypotheses;
2. evidence-sensitive updating;
3. context-dependent reactivation;
4. controllable divergence and convergence;
5. interpretable internal dynamics;
6. a measurable advantage over at least one simpler baseline on delayed or context-sensitive tasks.

## Out of Scope for v0.1

- natural-language generation inside the SNN;
- autonomous code execution;
- full conversation-history ingestion;
- end-to-end synthetic personality training;
- unrestricted self-modification;
- consciousness or identity claims;
- large-scale neuron-count optimization.

## Next Milestone After v0.1

Once the workspace is validated, add experiment selection.

Input:

```text
active hypotheses
expected information gain
compute cost
confounding risk
project constraints
publication status
personal-use status
```

Output:

```text
execute
defer
reject
```

This is the first stage where the system begins reproducing Alan's research decision boundary rather than only maintaining cognitive state.
