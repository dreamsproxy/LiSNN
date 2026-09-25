"""Structural and causal guarantees of the generic #61 stream contract."""

import numpy as np
import pytest

from lisnn.io import (CapacityError, ContinuousStream, EventStream,
                      InterfacePort, ModulatoryStream, Probe, StreamFrame,
                      StreamScheduler, Transducer)
from lisnn.network import create_nn
from lisnn.synapses import create_synapses


def frame(kind="CONTINUOUS", values=(1, 2), *, source="EXTERNAL", modality="auditory",
          axes=("channel",), stamp=0, sample_period=None, destination=None, effect=None):
    return StreamFrame(source, kind, modality, np.array(values), axes, stamp,
                       "experiment:two channels to two ordinary neurons",
                       sample_period, destination, effect)


def test_structure_timestamps_source_and_metadata_are_explicit():
    waveform = frame(values=[[1, 2], [3, 4]], axes=("time", "channel"), sample_period=2)
    assert waveform.channel_count == 2 and waveform.duration_ms == 4
    slice1 = waveform.sample(1)
    np.testing.assert_array_equal(slice1.values, [3, 4])
    assert slice1.timestamp_ms == 2 and slice1.modality == "auditory"
    assert slice1.mapping_provenance == waveform.mapping_provenance
    with pytest.raises(IndexError):
        waveform.sample(-1)
    with pytest.raises(ValueError):
        frame(values=[[1, 2]], axes=("time", "channel"))
    with pytest.raises(ValueError):
        frame(values=[[1, 2]], axes=("height", "width"), sample_period=1)
    with pytest.raises(ValueError):
        frame(kind="EVENT", values=[0.5, 0])
    with pytest.raises(ValueError):
        frame(kind="MODULATORY", values=[1, 0])
    with pytest.raises(ValueError):
        frame(source="unspecified")


def test_stream_kinds_source_separation_and_frame_order():
    e = frame(kind="EVENT", values=[1, 0])
    assert EventStream([e]).source == "EXTERNAL"
    assert ContinuousStream([frame(stamp=0), frame(stamp=2)]).kind == "CONTINUOUS"
    m = frame(kind="MODULATORY", destination="learning_gate", effect="scale", values=[1, 1])
    assert ModulatoryStream([m]).source == "EXTERNAL"
    with pytest.raises(ValueError):
        EventStream([e, frame(kind="EVENT", source="INTERNAL")])
    with pytest.raises(ValueError):
        ContinuousStream([frame(stamp=2), frame(stamp=1)])


def test_capacity_mapping_overlap_and_no_implicit_modulatory_injection():
    input_port = InterfacePort("auditory_in", 4, [1, 3], source="EXTERNAL", role="input")
    readout = InterfacePort("auditory_probe", 4, [3], source="EXTERNAL", role="readout")
    assert 3 in input_port.neuron_indices and 3 in readout.neuron_indices
    current = Transducer(input_port).to_current(frame(values=[10, 20]))
    np.testing.assert_array_equal(current, [0, 10, 0, 20])
    with pytest.raises(CapacityError, match="requires 3 channels; available 2"):
        Transducer(input_port).to_current(frame(values=[1, 2, 3]))
    with pytest.raises(ValueError, match="sources must match"):
        Transducer(input_port).to_current(frame(source="INTERNAL"))
    with pytest.raises(ValueError):
        Transducer(input_port).to_current(frame(kind="EVENT", values=[1, 0]))
    np.testing.assert_array_equal(Transducer(input_port, event_amplitude_pA=5).to_current(
        frame(kind="EVENT", values=[1, 0])), [0, 5, 0, 0])
    with pytest.raises(ValueError, match="does not permit current injection"):
        Transducer(input_port).to_current(frame(kind="MODULATORY", destination="learning_gate",
                                                effect="scale"))
    mod = frame(kind="MODULATORY", destination="feedback_current", effect="add_pA")
    np.testing.assert_array_equal(Transducer(input_port).to_current(mod), [0, 1, 0, 2])
    with pytest.raises(ValueError, match="explicit time sample"):
        Transducer(input_port).to_current(frame(values=[[1, 2]], axes=("time", "channel"), sample_period=1))


def test_scheduler_future_order_and_bounded_queue():
    scheduler = StreamScheduler(2)
    a, b = frame(source="INTERNAL"), frame(source="INTERNAL", stamp=1)
    with pytest.raises(ValueError):
        scheduler.schedule(a, 0, observed_tick=0)
    with pytest.raises(ValueError):
        scheduler.schedule(a, 1)
    scheduler.schedule(a, 1, observed_tick=0)
    scheduler.schedule(b, 1, observed_tick=0)
    with pytest.raises(CapacityError):
        scheduler.schedule(a, 2, observed_tick=0)
    assert scheduler.consume(0) == ()
    assert scheduler.consume(1) == (a, b)
    assert scheduler.consume(1) == ()


def test_probe_readback_is_owned_and_does_not_mutate_network():
    model = create_nn(2)
    model.configure_runtime(create_synapses(2, [], []), dt=0.1, impulse_scale=100)
    output = model.step([100, 0])
    probe = Probe(InterfacePort("readout", 2, [0], source="EXTERNAL", role="readout"))
    captured = probe.capture(output)
    assert captured["plasticity_voltage_mV"].shape == (1,)
    captured["voltage_mV"][:] = 999
    assert output.voltage_mV[0] != 999 and model.runtime.pool[0, 0] != 999
    assert "weights_after" in captured and captured["weights_after"] is None
