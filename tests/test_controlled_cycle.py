"""Engineering exit and negative-control checks for the declared state cycle."""

import numpy as np
import pytest

from lisnn.experiments import run_controlled_cycle


def test_all_arms_from_identical_snapshot_have_declared_controls():
    result = run_controlled_cycle()
    arms = result["controls"]
    assert set(arms) == {"quiet", "ordered", "shuffled", "ordered_learning_off"}
    baseline = result["baseline_weights"][0]
    assert len(result["experience"]) == 2
    assert result["configuration"]["pre_probe_affects_sleep_start"] is False
    assert result["configuration"]["washout_ticks"] == 100
    assert all(v["final_regime"] == "ACTIVE" for v in arms.values())
    assert len(arms["quiet"]["deliveries"]) == 0
    assert len(arms["ordered"]["deliveries"]) == len(arms["shuffled"]["deliveries"]) == 2
    assert [d["tick"] for d in arms["ordered"]["deliveries"]] == [d["tick"] for d in arms["shuffled"]["deliveries"]]
    assert arms["ordered"]["sleep_weights"][0] > baseline
    assert arms["shuffled"]["sleep_weights"][0] < baseline
    np.testing.assert_allclose(arms["quiet"]["sleep_weights"], [baseline])
    np.testing.assert_allclose(arms["ordered_learning_off"]["sleep_weights"], [baseline])
    assert arms["quiet"]["change_from_baseline_pA"] == 0
    assert arms["ordered_learning_off"]["change_from_baseline_pA"] == 0
    assert arms["shuffled"]["change_from_baseline_pA"] < 0
    assert arms["ordered"]["change_from_baseline_pA"] > 0


def test_determinism_raw_diagnostics_and_finite_state():
    a, b = run_controlled_cycle(), run_controlled_cycle()
    assert a == b
    for arm in a["controls"].values():
        assert len(arm["sleep_observations"]) == a["configuration"]["sleep_ticks"] - 1
        assert np.isfinite(arm["probe"]["readout_synaptic_current_pA"])
        for tick in arm["sleep_observations"]:
            for key in ("voltage_mV", "plasticity_voltage_mV", "external_current_pA",
                        "feedback_current_pA", "weights"):
                assert np.all(np.isfinite(tick[key]))
            assert "potentiation" in tick["update_components"]
            assert tick["trace_state"] is not None
    assert all(d["source"] == "INTERNAL" and d["provenance"].startswith("literal_replay:")
               for name, arm in a["controls"].items() if name != "quiet" for d in arm["deliveries"])


def test_progress_and_bounded_configuration():
    seen = []
    run_controlled_cycle(progress=lambda i, n, label: seen.append((i, n, label)),
                         washout_ticks=20)
    assert [x[0] for x in seen] == [1, 2, 3, 4]
    assert all(x[1] == 4 for x in seen)
    with pytest.raises(ValueError):
        run_controlled_cycle(washout_ticks=0)
    with pytest.raises(ValueError):
        run_controlled_cycle(sleep_ticks=2)
