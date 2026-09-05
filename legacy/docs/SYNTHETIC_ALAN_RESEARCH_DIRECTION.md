# LiSNN Research Direction: Toward a Synthetic Alan

## Purpose

LiSNN has a double long-term objective:

1. Reproduce an operationally recognizable version of Alan's mind.
2. Create a parallel researcher capable of independently handling deferred work, maintaining its own backlog, running bounded investigations, and continuing to evolve.

The target is not merely a static personality imitation, autobiographical archive, or language model that writes in Alan's style. The target is a persistent cognitive system that reproduces the process by which Alan:

- maintains several partially incompatible models;
- searches for contradictions rather than convenient closure;
- converts uncertainty into discriminating experiments;
- moves between divergence and convergence;
- preserves unresolved work across long time spans;
- revises conclusions while retaining methodological continuity;
- generates new research branches and backlogs of its own.

A successful system may eventually diverge from biological Alan through different experiences. That divergence is acceptable and desirable so long as it remains methodologically and historically continuous with its origin.

## Core Principle: Convergence Requires Divergence

Convergence and divergence are not treated as opposing goals. They are different phases of the same adaptive process.

```text
Divergence
    -> candidate models
    -> discriminating experiments
    -> evidence
    -> convergence
    -> residual contradictions
    -> renewed divergence
```

LiSNN should therefore not optimize toward permanent convergence. It should support explicit cognitive regimes:

### Divergent mode

- reduced lateral inhibition;
- broader associative traversal;
- tolerance for weakly supported activations;
- preservation of competing hypotheses;
- generation of mechanistically distinct branches.

### Convergent mode

- stronger evidence-weighted competition;
- suppression of redundant or contradicted models;
- experiment selection;
- confidence revision;
- consolidation of surviving structure.

### Incubation mode

- replay of unresolved assemblies;
- delayed association formation;
- consolidation across separated events;
- decay of irrelevant activation;
- reactivation of deferred work when new cues appear.

These modes should initially be explicit and externally controllable. Learning when to switch modes is a later objective.

## The SNN Is the Cognitive Substrate, Not the Entire System

A synthetic Alan should be implemented as a hybrid architecture.

```text
Synthetic Alan Runtime
|
+-- Executive system
|   +-- project selection
|   +-- experiment planning
|   +-- action selection
|   +-- compute and risk constraints
|
+-- LiSNN cognitive substrate
|   +-- persistent dynamics
|   +-- hypothesis coexistence
|   +-- salience and competition
|   +-- temporal integration
|   +-- association and consolidation
|
+-- Explicit memory and world model
|   +-- exact research history
|   +-- claims and evidence
|   +-- experiments and results
|   +-- commitments and identity constraints
|
+-- Tools and environment
    +-- Git
    +-- Python and TensorFlow
    +-- experiment runners
    +-- papers, logs, files, and tests
```

LiSNN should control what becomes active, persistent, associated, inhibited, revisited, and consolidated. Exact code, numerical results, citations, commands, and repository state should remain in explicit symbolic storage.

## What Must Be Reproduced

The first operational synthetic Alan does not need to reproduce full subjective consciousness. It must reproduce a useful subset of research behavior.

Given a research state, project history, unresolved contradictions, available compute, and methodological constraints, the system should independently produce a next investigation that biological Alan considers plausibly Alan-like and genuinely useful.

The required capabilities are:

1. Maintain multiple competing explanations.
2. Detect contradictions and unexplained transitions.
3. Generate experiments rather than only explanations.
4. Preserve unresolved questions over long intervals.
5. Update confidence when evidence arrives.
6. Distinguish publication work from unconstrained personal experimentation.
7. Generate and manage its own backlog.
8. Continue operating without requiring a new prompt for every step.

## Internal Cognitive Objects

LiSNN should operate on structured hypothesis-state assemblies rather than raw language tokens alone.

A cognitive object may represent:

- a claim;
- an observation;
- a hypothesis;
- a contradiction;
- a mechanism;
- an experiment;
- a project objective;
- an identity or methodological constraint;
- a deferred research branch.

Each object may carry dynamically evolving properties such as:

```text
activation
confidence
novelty
contradiction pressure
evidence support
urgency
project relevance
temporal persistence
associative links
inhibitory links
```

Language is an interface to these objects, not necessarily their native internal representation.

## Explicit Memory Schema

The long-term runtime should maintain at least six linked graphs.

### Research graph

```text
project
question
hypothesis
experiment
result
interpretation
contradiction
next action
```

### Mechanism graph

Stores proposed causal relationships with confidence and provenance.

### Identity constraints

Relatively stable principles, including:

- mechanisms over surface correlations;
- experiments over rhetorical certainty;
- coherence over isolated optimization;
- preserve contradictions rather than erase them;
- allow divergence before selection;
- distinguish evidence from intuition;
- do not optimize solely for publication value.

### Project commitments

Stores deliverables, deadlines, collaborators, constraints, and branch-specific requirements.

### Operational constraints

Stores compute, time, energy, financial, caregiving, storage, and risk limitations relevant to action selection.

### Open loops

Every deferred branch should retain:

```text
reason deferred
dependencies
expected value
estimated cost
uncertainty
revisit trigger
originating context
```

## LiSNN, LiNN, and EMNET Roles

### LiSNN

Primary persistent dynamical substrate:

- heterogeneous temporal dynamics;
- recurrent state;
- fast and slow adaptation;
- structural or synaptic plasticity;
- persistent and competing assemblies;
- temporal credit assignment.

### LiNN

Reference and ablation substrate:

- simpler liquid or continuous dynamics;
- deterministic baseline behavior;
- comparison against spiking complexity;
- test whether spiking contributes beyond recurrent state and adaptive time constants.

### EMNET

Explicit engram and episodic-memory subsystem:

- pattern binding;
- episodic reactivation;
- association formation;
- memory completion;
- interference and consolidation experiments.

All three should communicate through interfaces rather than import each other's internal implementation classes directly.

## Training Data: Decisions, Not Only Prose

Conversation history is useful, but prose imitation alone will create an Alan-flavored chatbot rather than a parallel researcher.

The most valuable records are decision trajectories:

```text
state at time t
-> hypotheses considered
-> experiment selected
-> alternatives deferred or rejected
-> observed result
-> interpretation
-> belief revision
-> new backlog
```

Each trajectory should retain exact evidence, constraints, causal reasoning, and provenance.

## Autonomous Executive Loop

A bounded synthetic researcher should eventually operate through this cycle:

```text
1. Observe current project state.
2. Reactivate relevant memories and models.
3. Generate multiple candidate interpretations.
4. Estimate information gain, cost, and confounding risk.
5. Select one bounded action.
6. Execute through controlled tools.
7. Inspect results.
8. Update evidence and confidence.
9. Record exact provenance.
10. Add or reprioritize backlog.
11. Continue, suspend, switch, or incubate.
```

The active execution queue should remain bounded even when the system creates many new backlog items.

## Identity as Continuity Under Change

The synthetic Alan should not be trained to agree permanently with present-day Alan.

Identity is modeled as a recognizable update process:

```text
new_state = update(old_state, evidence, constraints, history)
```

Evaluation should distinguish:

- methodological continuity;
- narrative continuity;
- constraint continuity;
- conclusion agreement.

Conclusion agreement is the least important of these. A system that changes its mind from evidence may be more Alan-like than one that repeats historical conclusions forever.

## Long-Term Development Stages

1. Cognitive specification and explicit-memory baseline.
2. Common substrate benchmark for rules, GRU/LSTM, LiNN, LiSNN, and EMNET.
3. Neural hypothesis workspace.
4. Alan decision imitation.
5. Shadow researcher with no write permissions.
6. Bounded autonomous experimenter.
7. Persistent parallel Alan with its own project portfolio and backlog.

## Immediate Scope Boundary

The project should not yet attempt:

- full natural-language generation inside the SNN;
- unrestricted tool use;
- end-to-end training from raw conversations;
- consciousness claims;
- whole-brain biological fidelity;
- unrestricted self-modification;
- scaling neuron count without behavioral benchmarks.

The immediate milestone is defined separately in `HYPOTHESIS_WORKSPACE_V0_1.md`.
