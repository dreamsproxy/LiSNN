"""Small controlled ACTIVE -> SLEEP -> ACTIVE replay/quiet comparison."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np

from lisnn.io import (InterfacePort, RegimeController, ReplayBuffer,
                      ReplayController, StreamFrame, StreamRuntime, Transducer)
from lisnn.network import create_nn
from lisnn.neurons import kernels as k
from lisnn.synapses import create_synapses


def _probe(regime):
    """Probe from a copy, then restore the complete regime/runtime state."""
    checkpoint = regime.snapshot()
    source = regime.step([32000, 0], plasticity_override=False).routed.neural
    target = regime.step(plasticity_override=False).routed.neural
    weights = regime.runtime.network.runtime.weights.copy()
    result = {
        "source_spikes": source.current_spikes.tolist(),
        "target_spikes": target.current_spikes.tolist(),
        "readout_synaptic_current_pA": float(target.propagation.synaptic_current_pA[1]),
        "target_voltage_mV": float(target.voltage_mV[1]),
        "target_plasticity_voltage_mV": float(target.plasticity_voltage_mV[1]),
        "weights": weights.tolist(),
        "probe_ticks": [source.tick, target.tick],
        "readout_spike": bool(target.current_spikes[1]),
    }
    regime.restore(checkpoint)
    return result


def run_controlled_cycle(*, progress=None, washout_ticks=100, sleep_ticks=6,
                         replay_gain=1, shuffle_seed=3):
    """Compare four arms from the same post-experience snapshot.

    A 100 ms washout precedes the baseline probe and follows SLEEP in every
    arm. Probe calls restore their pre-probe snapshot, so the baseline probe
    cannot alter the SLEEP starting state. Per-probe weight changes are frozen.
    Improvement is not assumed; report raw values and null/negative controls.
    """
    if isinstance(washout_ticks, bool) or not isinstance(washout_ticks, int) or washout_ticks < 20:
        raise ValueError("washout_ticks must be at least 20")
    if isinstance(sleep_ticks, bool) or not isinstance(sleep_ticks, int) or sleep_ticks < 4:
        raise ValueError("sleep_ticks must allow a quiet tick and both replay samples")
    network = create_nn(2, neuron_type="lif")
    network.pool[:, k.T_REF] = 0
    network.configure_runtime(create_synapses(2, [0], [1], [0.5]),
                              dt=1, impulse_scale=100, plasticity="pair",
                              plasticity_params=dict(tau_plus_ms=10, tau_minus_ms=10,
                                                     a_plus=0.2, a_minus=0.1, w_max=1))
    ports = {name: Transducer(InterfacePort(name, 2, indices, source=source))
             for name, source, indices in (
                 ("auditory_external", "EXTERNAL", [0]),
                 ("visual_external", "EXTERNAL", [1]),
                 ("auditory_internal", "INTERNAL", [0]),
                 ("visual_internal", "INTERNAL", [1]))}
    regime = RegimeController(StreamRuntime(network, transducers=ports, max_queued_frames=8),
                              seed=7)
    experience = ReplayBuffer(max_frames=2, max_duration_ms=2)
    for tick, (modality, port) in enumerate((
        ("auditory", "auditory_external"), ("visual", "visual_external"))):
        regime.runtime.schedule_frame(StreamFrame("EXTERNAL", "CONTINUOUS", modality,
                                                  [32000], ("channel",), tick,
                                                  f"declared_training:{port}"), port)
        experience.record(regime.step().routed)
    for _ in range(washout_ticks):
        regime.step(plasticity_override=False)
    baseline = regime.snapshot()
    buffer = experience.snapshot()
    before = _probe(regime)
    controls = {}
    configurations = (("quiet", None, False),
                      ("ordered", "ordered", False),
                      ("shuffled", "shuffled", False),
                      ("ordered_learning_off", "ordered", True))
    for number, (name, order, freeze) in enumerate(configurations, 1):
        if progress is not None:
            progress(number, len(configurations), name)
        branch = deepcopy(baseline)
        replay = ReplayController(buffer.snapshot(),
                                  internal_ports={"auditory": "auditory_internal",
                                                  "visual": "visual_internal"},
                                  max_gain=1, max_repeats=1, max_events=2,
                                  max_duration_ticks=2, seed=shuffle_seed)
        sleep_start = branch.runtime.tick
        branch.transition(sleep_start, "SLEEP")
        branch.transition(sleep_start + sleep_ticks, "ACTIVE")
        branch.step(plasticity_override=not freeze)  # Quiet before any replay.
        if order is not None:
            replay.schedule(branch, observed_tick=sleep_start,
                            start_tick=sleep_start + 1, gain=replay_gain, order=order)
        deliveries, observations = [], []
        for _ in range(sleep_ticks - 1):
            step = branch.step(plasticity_override=not freeze)
            update = step.routed.neural.plasticity_update
            components = {} if update is None else {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in asdict(update).items()}
            trace = step.routed.neural.trace_state
            observations.append({"tick": step.routed.neural.tick,
                                 "spikes": step.routed.neural.current_spikes.tolist(),
                                 "voltage_mV": step.routed.neural.voltage_mV.tolist(),
                                 "plasticity_voltage_mV": step.routed.neural.plasticity_voltage_mV.tolist(),
                                 "external_current_pA": step.routed.neural.external_current_pA.tolist(),
                                 "feedback_current_pA": step.routed.neural.feedback_current_pA.tolist(),
                                 "weights": branch.runtime.network.runtime.weights.tolist(),
                                 "trace_state": None if trace is None else
                                     [x.tolist() for x in trace],
                                 "update_components": components})
            deliveries.extend({"source": d.source, "port": d.port, "modality": d.modality,
                               "tick": d.delivery_tick, "current_pA": d.channel_current_pA.tolist(),
                               "provenance": d.mode} for d in step.routed.deliveries)
        sleep_weights = branch.runtime.network.runtime.weights.copy()
        for _ in range(washout_ticks):
            branch.step(plasticity_override=False)
        after = _probe(branch)
        controls[name] = {
            "probe": after,
            "change_from_baseline_pA": after["readout_synaptic_current_pA"] - before["readout_synaptic_current_pA"],
            "sleep_weights": sleep_weights.tolist(),
            "deliveries": deliveries,
            "sleep_observations": observations,
            "final_regime": branch.current.name,
            "learning_enabled_during_sleep": not freeze,
        }
    return {
        "configuration": {"seed": 7, "shuffle_seed": shuffle_seed,
                          "dt_ms": 1, "initial_weight": 0.5,
                          "washout_ticks": washout_ticks, "sleep_ticks": sleep_ticks,
                          "replay_gain": replay_gain,
                          "pre_probe_affects_sleep_start": False,
                          "probe_plasticity_enabled": False,
                          "metric": "target synaptic pA at one tick after direct source probe"},
        "experience": [{"source": "EXTERNAL", "modality": e.modality,
                        "port": e.external_port, "tick": e.original_tick,
                        "current_pA": e.channel_current_pA.tolist()}
                       for e in experience.entries],
        "baseline_weights": baseline.runtime.network.runtime.weights.tolist(),
        "baseline_probe": before,
        "controls": controls,
    }
