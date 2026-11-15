"""Validate and freeze experiment-side P1.4 plans without running a network.

This is a structural/provenance gate, not verification that an external task
adapter implements its declared scientific controls or produces learning.
"""

from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
import math
import re

from lisnn.validation import scalar32


LEARNERS = ("pair", "triplet", "voltage")
FEEDBACK_CONTROLS = ("contingent", "none", "random", "yoked", "inverted", "plasticity_off")
STATE_ARMS = ("quiet", "ordered", "shuffled", "ordered_learning_off")
METRICS = ("response_rate", "no_response_rate", "response_quality", "activity",
           "weight_change", "adaptation_speed", "forgetting", "transfer")
RNG_STREAMS = ("network", "stimulus", "feedback", "replay", "evaluation")


def canonical_json(value):
    """Portable finite JSON; byte-identical for equivalent object key orders."""
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise ValueError("protocol metadata must contain only JSON values")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _number(value, name, *, integer=False, positive=True, signed=False):
    valid_type = type(value) is int if integer else type(value) in (int, float)
    finite = valid_type and (type(value) is int or math.isfinite(value))
    if not finite or (not signed and (value < 0 or (positive and value == 0))):
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'} "
                         f"{'greater than zero' if positive else 'at least zero'}")


def _choices(value, expected, name):
    if (not isinstance(value, list) or any(not isinstance(x, str) for x in value)
            or len(value) != len(set(value)) or set(value) != set(expected)):
        raise ValueError(f"{name} must include each of {expected} exactly once")


def validate_protocol(protocol):
    """Return an owned validated declaration, rejecting incomplete controls.

    The caller must still audit the external implementation. In particular,
    declaring a yoking rule does not prove matched delivered stimulation.
    """
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    # Normalize now; never accept NaN, NumPy objects, tuples or duplicate keys
    # hidden in a subsequent file conversion as scientific metadata.
    canonical_json(protocol)
    p = deepcopy(protocol)
    required = {"schema_version", "protocol_id", "scope", "task", "configuration", "seeds", "learners",
                "feedback_controls", "state_arms", "learner_parameters", "timing",
                "budgets", "splits", "metrics", "rules"}
    if set(p) != required:
        raise ValueError(f"protocol fields mismatch: missing={required - set(p)}, "
                         f"unknown={set(p) - required}")
    if type(p["schema_version"]) is not int or p["schema_version"] != 1:
        raise ValueError("unsupported protocol schema_version")
    if not isinstance(p["protocol_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", p["protocol_id"]):
        raise ValueError("protocol_id must be a short filesystem-safe identifier")
    if p["scope"] not in ("engineering_only", "p1.4"):
        raise ValueError("scope must be engineering_only or p1.4")
    _text(p["task"], "task")
    seeds = p["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("seeds must be a nonempty list of uint32 integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds cannot repeat")
    if len(seeds) < 2:
        raise ValueError("yoked control requires at least two independent donor/recipient seeds")
    if p["scope"] == "p1.4" and len(seeds) < 3:
        raise ValueError("p1.4 evidence plans require at least three declared seeds")
    configuration = p["configuration"]
    if not isinstance(configuration, dict) or set(configuration) != {"task_adapter", "network", "environment", "feedback", "replay"}:
        raise ValueError("configuration requires task_adapter, network, environment, feedback and replay")
    _text(configuration["task_adapter"], "task_adapter")
    if p["scope"] == "p1.4" and not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", configuration["task_adapter"]):
        raise ValueError("p1.4 task_adapter must identify the external module:callable implementation")
    for name in ("network", "environment", "feedback", "replay"):
        if not isinstance(configuration[name], dict) or not configuration[name]:
            raise ValueError(f"configuration.{name} must contain explicit settings")
    _choices(p["learners"], LEARNERS, "learners")
    _choices(p["feedback_controls"], FEEDBACK_CONTROLS, "feedback_controls")
    _choices(p["state_arms"], STATE_ARMS, "state_arms")
    if not isinstance(p["learner_parameters"], dict) or set(p["learner_parameters"]) != set(LEARNERS):
        raise ValueError("declare separate parameters for every alternative learner")
    for learner in LEARNERS:
        params = p["learner_parameters"][learner]
        if not isinstance(params, dict) or not params:
            raise ValueError(f"{learner} requires explicit learner parameters")
        for key, value in params.items():
            _text(key, "learner parameter name")
            _number(value, key, positive=False, signed=True)
    # Construct learners only to validate their actual parameter contract. No
    # neurons are constructed or stepped. Voltage's initial observation is
    # supplied by the external runtime, not a tunable learner parameter.
    from lisnn.plasticity import PairSTDP, TripletSTDP, VoltageSTDP
    from lisnn.synapses import create_synapses
    edges = create_synapses(2, [0], [1], weights=0)
    for name, cls in (("pair", PairSTDP), ("triplet", TripletSTDP), ("voltage", VoltageSTDP)):
        try:
            extra = {"initial_voltage_mV": -65} if name == "voltage" else {}
            cls(edges, **p["learner_parameters"][name], **extra)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid {name} learner parameters: {exc}") from exc
    timing = p["timing"]
    timing_fields = {"dt_ms", "stimulus_ticks", "response_ticks", "washout_ticks", "interval_ticks", "max_ticks"}
    if not isinstance(timing, dict) or set(timing) != timing_fields:
        raise ValueError(f"timing must declare {sorted(timing_fields)}")
    for key, value in timing.items():
        _number(value, key, integer=key != "dt_ms")
    scalar32("dt_ms", timing["dt_ms"], positive=True)
    if timing["max_ticks"] < sum(timing[k] for k in ("stimulus_ticks", "response_ticks", "washout_ticks", "interval_ticks")):
        raise ValueError("max_ticks cannot fit even one declared cycle")
    budgets = p["budgets"]
    budget_fields = {"train_trials", "evaluation_trials", "max_stimulus_events",
                     "max_feedback_events", "replay_events", "max_abs_current_pA"}
    if not isinstance(budgets, dict) or set(budgets) != budget_fields:
        raise ValueError(f"budgets must declare {sorted(budget_fields)}")
    for key, value in budgets.items():
        _number(value, key, integer=key != "max_abs_current_pA")
    scalar32("max_abs_current_pA", budgets["max_abs_current_pA"], positive=True)
    if budgets["replay_events"] < 2:
        raise ValueError("shuffled replay requires at least two declared events")
    splits = p["splits"]
    if not isinstance(splits, dict) or set(splits) != {"train", "held_out", "transfer"}:
        raise ValueError("declare train, held_out and transfer splits")
    seen_ids = set()
    for name, split in splits.items():
        if not isinstance(split, dict) or set(split) != {"source", "sha256", "ids"}:
            raise ValueError(f"{name} needs source, sha256 and ids")
        _text(split["source"], f"{name}.source")
        if not isinstance(split["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", split["sha256"]):
            raise ValueError(f"{name} needs a lowercase source SHA256")
        ids = split["ids"]
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{name} requires explicit sample/context IDs")
        for sample in ids:
            _text(sample, f"{name}.ids")
            if sample in seen_ids:
                raise ValueError("split IDs must be unique and disjoint across splits")
            seen_ids.add(sample)
    if not isinstance(p["metrics"], dict) or not set(METRICS) <= set(p["metrics"]):
        raise ValueError(f"metrics must define at least {METRICS}")
    for name, definition in p["metrics"].items():
        _text(name, "metric name")
        _text(definition, f"metric {name} (include units and denominator)")
    rule_names = {"stopping", "failure", "evaluation", "state_comparison", "yoking", "statistics"}
    if not isinstance(p["rules"], dict) or set(p["rules"]) != rule_names:
        raise ValueError(f"rules must declare {sorted(rule_names)}")
    for name, definition in p["rules"].items():
        _text(definition, f"rule {name}")
    return p


def build_run_plan(protocol):
    """One plan row per learner/control/seed; state arms branch within each.

    Network/stimulus/replay seeds are paired across controls; feedback has its
    own stream. No mutable generator is shared across condition executions.
    Yoked donors reference a different declared seed's contingent run; executing
    an adapter must load its recorded schedule rather than regenerate it.
    """
    p = validate_protocol(protocol)
    rows = []
    for learner, condition, seed in product(p["learners"], p["feedback_controls"], p["seeds"]):
        donor_seed = p["seeds"][(p["seeds"].index(seed) + 1) % len(p["seeds"])]
        streams = {name: int.from_bytes(sha256(f"lisnn-p1.4:{seed}:{name}".encode()).digest()[:4], "big")
                   for name in RNG_STREAMS}
        rows.append({"run_id": f"{learner}--{condition}--seed-{seed}",
                     "learner": learner, "condition": condition, "seed": seed,
                     "rng_seeds": streams,
                     "feedback_mode": "contingent" if condition == "plasticity_off" else condition,
                     "plasticity_enabled": condition != "plasticity_off",
                     "state_arms": list(p["state_arms"]),
                     "state_arm_plasticity": {arm: condition != "plasticity_off" and arm != "ordered_learning_off"
                                               for arm in p["state_arms"]},
                     "yoked_donor": f"{learner}--contingent--seed-{donor_seed}" if condition == "yoked" else None})
    return rows


def freeze_protocol(protocol):
    """Attach the fully enumerated run ledger; does not execute any run."""
    p = validate_protocol(protocol)
    plan = build_run_plan(p)
    return {"declaration": p, "run_plan": plan,
            "run_ids": [row["run_id"] for row in plan]}


def engineering_protocol():
    """Editable, fully explicit *planning fixture*, never empirical evidence.

    These numerical values are not tuned or recommended task parameters. The
    fixture exercises the protocol gate; supply a reviewed task declaration
    and implementation before executing scientific experiments.
    """
    source = "synthetic planning fixture v1; explicit disjoint sample IDs; no dataset generated"
    return {
        "schema_version": 1, "protocol_id": "p1.4-engineering-plan",
        "scope": "engineering_only",
        "task": "Planning fixture only: declare an external task adapter before scientific execution. No network is run by this plan.",
        "configuration": {
            "task_adapter": "planning_only",
            "network": {"population_size": 64, "neuron_type": "lif", "impulse_scale": 100,
                        "construction": "Planning fixture; task adapter must declare topology, efficacy and ports."},
            "environment": {"kind": "synthetic planning fixture", "response_rule": "No executable environment supplied."},
            "feedback": {"success_pA": 100, "error_pA": 1000, "delay_ticks": 1,
                         "random_probability": 0.5, "port": "declared_internal_feedback"},
            "replay": {"gain": 1, "repeats": 1, "max_duration_ticks": 1000}},
        "seeds": [0, 1, 2], "learners": list(LEARNERS),
        "feedback_controls": list(FEEDBACK_CONTROLS), "state_arms": list(STATE_ARMS),
        "learner_parameters": {
            "pair": {"tau_plus_ms": 20, "tau_minus_ms": 20, "a_plus": 0.001, "a_minus": 0.001, "w_max": 1},
            "triplet": {"tau_plus_ms": 20, "tau_minus_ms": 20, "tau_x_ms": 100, "tau_y_ms": 100,
                        "a2_plus": 0.001, "a2_minus": 0.001, "a3_plus": 0.001, "a3_minus": 0.001, "w_max": 1},
            "voltage": {"tau_x_ms": 20, "tau_minus_ms": 10, "tau_plus_ms": 10,
                        "theta_minus_mV": -70, "theta_plus_mV": -45,
                        "a_ltd": 0.0001, "a_ltp": 0.0001, "w_max": 1}},
        "timing": {"dt_ms": 0.1, "stimulus_ticks": 10, "response_ticks": 100,
                   "washout_ticks": 1000, "interval_ticks": 1000, "max_ticks": 100000},
        "budgets": {"train_trials": 10, "evaluation_trials": 10, "max_stimulus_events": 1000,
                    "max_feedback_events": 100, "replay_events": 10, "max_abs_current_pA": 1000},
        "splits": {name: {"source": source, "sha256": sha256(source.encode()).hexdigest(),
                          "ids": [f"{name}-{i}" for i in range(10)]}
                   for name in ("train", "held_out", "transfer")},
        "metrics": {
            "response_rate": "Trials with >=1 readout spike in the response window / all evaluation trials; fraction.",
            "no_response_rate": "Trials with zero readout spikes in the response window / all evaluation trials; fraction.",
            "response_quality": "Correct decoded responses / all evaluation trials, counting no response as incorrect; fraction. Adapter must define decoder and correctness before running.",
            "activity": "Total spikes / (neuron count * elapsed seconds); Hz per neuron.",
            "weight_change": "Mean absolute final-minus-initial efficacy over all declared edges; efficacy units.",
            "adaptation_speed": "Area under trial-index response-quality curve / declared training trials; fraction, not best-trial performance.",
            "forgetting": "100 * (pre-minus-post interval response quality) on identical frozen probes; percentage points.",
            "transfer": "100 * (transfer response quality minus naive matched-exposure control quality); percentage points, signed negative transfer retained."},
        "rules": {
            "stopping": "Stop at declared trial and tick budgets; no success-based early stopping or post-hoc seed replacement.",
            "failure": "Retain numerical nonfinite state, capacity errors, interrupted runs and all failed seeds with diagnostics; no NaN-as-zero metrics.",
            "evaluation": "Freeze plasticity; run identical probes on full-state copies and discard them. Declare response window, decoder and no-response denominator before executing.",
            "state_comparison": "Branch quiet/ordered/shuffled/learning-off from one post-experience snapshot, with matched washout and replay budgets; preserve initial and final snapshots plus traces.",
            "yoking": "Use another run's recorded delivered currents and relative delivery ticks, independent of recipient outcomes. Record donor ID and digest; audit actual count/timing/current distribution.",
            "statistics": "Report every declared seed, mean, spread and seed-level confidence intervals, all failures and null/negative outcomes; never select the best run."}}
