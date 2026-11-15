# P1.4 adaptation protocol and evidence policy

[P1.4 / #66](https://github.com/dreamsproxy/LiSNN/issues/66) evaluates adaptation,
recall, retention and transfer using task semantics outside the neural core.
Engineering readiness and experimental completion are separate gates:

| Gate | Required result |
| --- | --- |
| Engineering readiness | Validated protocol, complete run plan, replayable controls, isolated evaluation, provenance and failure accounting |
| Experimental completion | Recorded runs supporting explicit claims and limitations for the declared adaptation, recall, retention and transfer tasks |

A plan, passing tests or the two-neuron controlled-cycle example cannot close
the evidence gate. Learning improvement is an outcome to measure. Null,
negative, unstable and degenerate outcomes remain part of the evidence.
P1.4 remains open until a reproducible evidence package has been reviewed.

## Prepare without running experiments

From the repository root:

```bash
python -m examples.adaptation_protocol --output logs/p1.4-plan
```

This creates a frozen `protocol.json` and a run `manifest.json`; it does not
execute neural adaptation or mark any run complete. The default
`engineering_only` fixture plans three alternative learners, six causal-control
conditions and three seeds: 54 rows, each declaring the four interval arms.
It supplies no empirical task adapter or experiment runner. Complete the
task-specific declarations and implement the external task before collecting
scientific evidence. A changed protocol is a new study revision with its own
digest, not an in-place edit of the interpretation of existing results.

Validate and enumerate without writing any output:

```bash
python -m examples.adaptation_protocol --check
python -m examples.adaptation_protocol --protocol task-protocol.json --check
python -m examples.adaptation_protocol --protocol task-protocol.json --output logs/p1.4-study
```

`--protocol` accepts a declaration object, as returned by
`engineering_protocol()`. The output `protocol.json` wraps that declaration
with its run plan and IDs; it is not itself the declaration input. Existing
output directories are never overwritten. For a scientific declaration,
`scope="p1.4"` requires at least three declared seeds. Validation checks
structure and local learner parameter contracts; it does not verify an
external adapter's task semantics, scientific design or delivered controls.
The declaration has a `configuration` section for the task adapter, network,
environment, feedback and replay settings. Scientific scope requires a
`module:callable` adapter identity, but validation does not import or execute
it. Record every construction and mapping setting there; placeholders in the
engineering fixture must be replaced and reviewed before real execution.

The template is scaffolding, not a chosen task, calibrated stimulus regime,
or evidence that a learner adapts. Auditory, visual and symbolic tasks,
next-state prediction, context dependence and continual adaptation are
experiment choices. Arithmetic and Shakespeare experiments remain optional
external tasks; no language-model objective is introduced into the core.

## Freeze the study before interpreting results

Record the following for each declared task:

- Task definition, environment version and decoding/response rule. Define
  no-response, multiple responses, operator/context confusion and timeouts.
- Source-data provenance and content digests; explicit training, evaluation
  and transfer partitions with item or group IDs. Prevent source leakage,
  including shared sequence windows, subjects or contexts where relevant.
- Network construction and learner parameters, timestep in ms, input in pA,
  observation units, initial-state policy and timestamp-to-tick mapping.
- Seeds and separate random streams for construction, stimulus selection,
  feedback, replay and evaluation. Save actual schedules or generator state
  as well as seeds; a seed alone does not describe later random draws.
- Exposure and stopping budgets: presentations, delivered currents, ticks,
  feedback delay and event count, replay events/gain/duration, quiet periods,
  maximum runtime and numerical/memory failure rules.
- Primary and secondary metrics, their units and denominators, aggregation
  across seeds, confidence-interval method and planned comparisons.

Select a fixed training budget or a predeclared stopping rule. Preserve the
first protocol and results when changing parameters; label exploration and
confirmatory runs separately. Do not tune on held-out or transfer outcomes
and then report them as untouched evaluation.

## Causal feedback controls

Each learner uses one rule at a time: Pair, Triplet or Voltage STDP. They are
alternatives, not additive defaults. Keep construction, data splits, scheduled
sensory experience and stopping rules matched for the intended comparison.

| Condition | Delivered feedback | Learning |
| --- | --- | --- |
| contingent | Predictable success stimulation; seeded unpredictable error/no-response perturbation | Selected learner |
| no feedback | No outcome feedback | Selected learner |
| random | Performance-independent success/error selection | Selected learner |
| yoked | Recorded feedback from another identified donor run | Selected learner |
| inverted | Success/error contingency reversed | Selected learner |
| plasticity off | Contingent feedback with weight updates disabled | Frozen weights |

`FeedbackPolicy` supplies these mechanisms at the experiment boundary. Task
answers, expected outputs, error scalars and supervised weight deltas never
enter the local learner. Outcomes affect only subsequent scheduled experience,
with a delay of at least one tick from an already executed observation.

A random condition breaks outcome contingency but does not automatically
match the contingent run's realized number, distribution or amplitude of
perturbations. Report delivered event counts, timing, stimulated channels,
absolute-current sums and integrated current/exposure. If exposure differs,
report that confound. Declare what varies unpredictably on errors; a fixed
amplitude delivered to one fixed site at a fixed delay has no randomized
spatial or temporal contrast. A yoked control must identify its donor and replay
its actual delivered schedule/currents with the declared mapping. The plan
uses the next distinct declared seed's contingent run as donor, cyclically,
within the same learner. Engineering plans therefore need at least two
distinct seeds. Using the recipient's own paired-seed contingent trajectory
could reproduce the same outcomes and leave contingency intact. Verify event,
timing and current budgets; do not infer a matched exposure or broken
contingency from the label `yoked`. Keep donor and recipient outcomes separate.

Snapshot the separate feedback policy as well as the runtime when branching;
a runtime snapshot alone does not include `FeedbackPolicy`'s RNG/cursor.
Failed scheduling must leave random state reusable, and exhausted donor
schedules are explicit failures rather than silent random fallback.

## State cycles and evaluation isolation

Branch every interval arm from the same post-experience checkpoint:

| Interval arm | Stimulation | Weight updates |
| --- | --- | --- |
| quiet | None | Declared learner policy |
| ordered replay | Recorded experience in order | Enabled |
| shuffled replay | Same samples, timing slots and exposure budget, shuffled order | Enabled |
| ordered replay, learning off | Same ordered replay | Disabled |

For the `plasticity_off` causal condition, all four arms keep weight updates
disabled. The plan records this per-arm policy explicitly. The table above
describes the learning-enabled conditions.

Record the full starting state: neuron pool, synapses, learner traces, clock,
queues, replay buffer, regime transitions, feedback and background RNGs,
environment state and any decoder state. Checkpoint equality refers to all
of those states, not merely initial weights.

Use identical evaluation inputs and timing, with weight updates frozen. A
frozen learner still advances its traces; probes also change membrane,
refractory, clock and queue state. Run probes on a disposable full-state
branch or restore the full checkpoint afterward. Merely setting learning off
does not make evaluation noninvasive. Restore in a `finally` block if probing
the original object so failed probes cannot contaminate a later arm.

Declare whether a baseline probe can affect the later interval. To estimate
persistent changes, use a declared common washout and distinguish residual
neural activity from altered weights. Ordered and shuffled replay must have
matched stimulus budgets; shuffled order can remain an ineffective contrast
when the payloads are identical. Literal stimulation replay is not endogenous
reconstruction, imagination, or proof of cue-driven recall.

## Metrics and run accounting

Keep training recall, held-out generalization and cross-context/cross-modal
transfer separate. Report at least response/no-response rates, task quality,
activity/firing, weights and inspectable plasticity diagnostics. Quantify
adaptation speed, forgetting/interference and negative transfer for the tasks
that claim them. Report quality both with its declared response denominator
and alongside the no-response rate; high accuracy among rare responses can
hide failure to respond.

The run manifest enumerates every planned learner/condition/seed combination.
Keep unfinished, interrupted and failed runs visible. Record failure reason,
last successful step, partial artifacts and whether the failure meets a
predeclared exclusion rule. Exclusion never deletes the original record;
retries get their own identity and link to the original. Report spread across
all planned seeds with the declared uncertainty calculation, not the best run.

An inspectable run package contains:

- Frozen protocol and digest, run identity, condition, learner and seed.
- Source commit, actual source-content digest or dirty diff, backend,
  Python/NumPy versions and relevant environment settings.
- Data/split digests, initial/final state checkpoints, random-state records
  or independent delivered schedules, and replay/donor provenance.
- Raw response and neural observations, feedback deliveries, stimulus
  accounting, metrics, status and failure information.
- Claim-to-run mapping, null/negative results, confounds, limitations and
  next questions.

Use explicit versioned data formats and validate schema, units, shapes and
content digests before replay or aggregation.

## Evidence-package API

`freeze_protocol` returns an owned declaration and complete run plan. The
`EvidencePackage` ledger stores that plan and append-only terminal receipts;
it does not execute tasks. This example prepares a package and defines hooks
for an external adapter's actual results. The hooks do not run an experiment:

```python
import json
from pathlib import Path

from lisnn.experiments.evidence import EvidencePackage
from lisnn.experiments.protocol import freeze_protocol
from lisnn.experiments.provenance import capture_provenance

repository = Path.cwd()  # Run from the repository root.
declaration = json.loads(Path("task-protocol.json").read_text())
frozen = freeze_protocol(declaration)
package = EvidencePackage.create(
    "logs/p1.4-study", frozen, capture_provenance(repository),
)
(package.path / "raw").mkdir()

def capture_execution(run_id):
    # Call at actual run start; retain this artifact even if the run fails.
    provenance_path = f"raw/{run_id}-execution.json"
    with (package.path / provenance_path).open("x") as stream:
        json.dump(capture_provenance(repository), stream, allow_nan=False)
    return {"execution_provenance": provenance_path}

def record_completed_run(run_id, metrics, artifact_paths):
    # Include execution provenance, schedules, checkpoints and raw observations.
    return package.record(
        run_id, status="completed", metrics=metrics,
        artifacts=artifact_paths,
    )

def record_failed_run(run_id, error, partial_artifact_paths):
    return package.record(
        run_id, status="failed", error=str(error),
        artifacts=partial_artifact_paths,
    )

summary = EvidencePackage.open("logs/p1.4-study").summary()
assert summary["execution_complete"] is False  # All rows are still pending.
```

Capture execution provenance at run start and retain it if execution fails.
A package's initial provenance alone identifies preparation,
not a later run on different code or hardware. Artifact paths are relative
POSIX paths under the package; record schedules, checkpoints and observations
there before recording the terminal receipt. `record` hashes them and refuses
to replace an existing receipt, including a failed one. `open` and `summary`
verify protocol, receipts and referenced artifact hashes.

`execution_complete` means only that no planned row is pending. Failed rows
are terminal outcomes, so a plan containing failures can be execution-complete.
It does not mean every run succeeded, all required scientific artifacts or
metrics were supplied, or adaptation was demonstrated. Numeric metrics and
artifact hashes are validated by the ledger; the external adapter and review
must establish task-specific completeness, state-arm coverage and claims.
Keep a pending row pending after interruption until a failure receipt records
the interruption and partial evidence; never turn it into a fabricated success.

## CI and pipeline policy

Default CI validates protocol construction, manifest accounting, deterministic
engineering invariants, frozen-probe isolation and numerical/memory contracts.
It uses both optimized and reference topology validation. It does not launch
the planned adaptation study, require a particular improvement direction,
select favorable seeds, or substitute generated fixtures for measured results.

Tests of a deliberately chosen tiny STDP pattern can assert that pattern's
known behavior. Such a test validates the implementation; it cannot establish
adaptation, durable recall or transfer on a scientific task. CI diagnostics
remain available on failure; matrix failures do not cancel other validation
conditions. New changes cancel obsolete CI for the same branch.

Scientific runs are explicit, separately budgeted work. Before claiming P1.4
complete, review the frozen plan against the entire manifest, verify artifact
integrity and causal/state-cycle controls, and trace each conclusion to its
recorded runs. Keep #66 open when the evidence is missing or the declared
scope remains incomplete.
