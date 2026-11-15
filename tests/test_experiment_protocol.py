"""Protocol preparation checks: no scientific experiments are executed."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from lisnn.experiments import build_run_plan, freeze_protocol, validate_protocol
from lisnn.experiments.protocol import engineering_protocol
from lisnn.experiments.provenance import capture_provenance


def test_full_plan_controls_are_paired_and_yoked_donors_are_identified():
    p = engineering_protocol()
    rows = build_run_plan(p)
    assert len(rows) == 54
    by_id = {row["run_id"]: row for row in rows}
    assert len(by_id) == 54
    for row in rows:
        assert set(row["state_arms"]) == set(p["state_arms"])
        assert len(set(row["rng_seeds"].values())) == 5
        assert row["state_arm_plasticity"] == {
            arm: row["condition"] != "plasticity_off" and arm != "ordered_learning_off"
            for arm in row["state_arms"]}
        reference = by_id[f"{row['learner']}--contingent--seed-{row['seed']}"]
        assert row["rng_seeds"] == reference["rng_seeds"]
        if row["condition"] == "yoked":
            donor = by_id[row["yoked_donor"]]
            assert donor["condition"] == "contingent"
            assert donor["learner"] == row["learner"]
            assert donor["seed"] != row["seed"]
        else:
            assert row["yoked_donor"] is None
        if row["condition"] == "plasticity_off":
            assert row["feedback_mode"] == "contingent"
            assert row["plasticity_enabled"] is False
        else:
            assert row["plasticity_enabled"] is True
    assert freeze_protocol(p) == freeze_protocol(deepcopy(p))
    frozen = freeze_protocol(p)
    p["seeds"].append(9)
    assert frozen["declaration"]["seeds"] == [0, 1, 2]


@pytest.mark.parametrize("field,value", [
    ("seeds", []), ("seeds", [0, True]), ("seeds", [0, 0]),
    ("seeds", [-1]), ("seeds", [2**32]),
    ("learners", ["pair"]), ("learners", ["pair", "pair", "voltage"]),
    ("feedback_controls", ["contingent"]), ("state_arms", ["ordered"]),
    ("scope", "complete"), ("protocol_id", "../escape"),
    ("schema_version", True), ("task", "  "),
])
def test_invalid_declarations_cannot_be_frozen(field, value):
    p = engineering_protocol()
    p[field] = value
    with pytest.raises(ValueError):
        freeze_protocol(p)


@pytest.mark.parametrize("section,field,value", [
    ("timing", "dt_ms", float("nan")), ("timing", "dt_ms", float("inf")),
    ("timing", "dt_ms", 1e-300), ("timing", "dt_ms", 1e300),
    ("timing", "washout_ticks", 0), ("timing", "max_ticks", 1),
    ("timing", "response_ticks", True), ("budgets", "train_trials", 0),
    ("budgets", "replay_events", 1), ("budgets", "max_abs_current_pA", -1),
    ("budgets", "max_abs_current_pA", 1e-300), ("budgets", "max_abs_current_pA", 1e300),
    ("metrics", "response_quality", ""), ("rules", "failure", None),
])
def test_invalid_units_budgets_and_reporting_rules(section, field, value):
    p = engineering_protocol()
    p[section][field] = value
    with pytest.raises(ValueError):
        validate_protocol(p)


def test_split_leakage_missing_controls_and_bad_learner_parameters():
    p = engineering_protocol()
    p["splits"]["held_out"]["ids"] = p["splits"]["train"]["ids"][:1]
    with pytest.raises(ValueError, match="disjoint"):
        validate_protocol(p)
    p = engineering_protocol()
    p["learner_parameters"]["voltage"].pop("theta_plus_mV")
    with pytest.raises(ValueError, match="voltage learner parameters"):
        validate_protocol(p)
    p = engineering_protocol()
    p["learner_parameters"]["pair"]["tau_plus_ms"] = 0
    with pytest.raises(ValueError, match="pair learner parameters"):
        validate_protocol(p)
    p = engineering_protocol()
    p["unknown_policy"] = "no silent extras"
    with pytest.raises(ValueError, match="unknown"):
        validate_protocol(p)


def test_scientific_scope_requires_multiple_seeds_but_never_claims_evidence():
    p = engineering_protocol()
    p["scope"], p["seeds"] = "p1.4", [0, 1]
    with pytest.raises(ValueError, match="three"):
        validate_protocol(p)


def test_scientific_declaration_needs_adapter_and_explicit_configuration():
    p = engineering_protocol()
    p["scope"] = "p1.4"
    with pytest.raises(ValueError, match="module:callable"):
        validate_protocol(p)
    p["configuration"]["task_adapter"] = "study.task:run"
    assert validate_protocol(p)["scope"] == "p1.4"
    p["configuration"]["network"] = {}
    with pytest.raises(ValueError, match="network"):
        validate_protocol(p)


@pytest.mark.parametrize("bad", [(1, 2), {1: "non-string key"}])
def test_direct_api_json_contract_matches_package_creation(bad):
    p = engineering_protocol()
    p["configuration"]["environment"]["bad"] = bad
    with pytest.raises(ValueError):
        validate_protocol(p)


def test_prepare_cli_leaves_every_run_pending_and_refuses_overwrite(tmp_path, capsys):
    from examples.adaptation_protocol import main

    target = tmp_path / "package"
    result = main(["--output", str(target)])
    assert result["planned_runs"] == 54
    assert len(result["pending"]) == 54
    assert result["completed"] == result["failed"] == []
    assert result["execution_complete"] is False
    assert list((target / "runs").iterdir()) == []
    assert json.loads((target / "protocol.json").read_text())["declaration"]["scope"] == "engineering_only"
    assert "executed runs: 0" in capsys.readouterr().out
    before = (target / "manifest.json").read_bytes()
    with pytest.raises(SystemExit):
        main(["--output", str(target)])
    assert (target / "manifest.json").read_bytes() == before


def test_check_only_does_not_create_output_or_step_a_network(tmp_path, monkeypatch):
    from examples.adaptation_protocol import main
    from lisnn.network.core import SNN

    def forbidden(*args, **kwargs):
        pytest.fail("protocol preparation cannot step a neural experiment")

    monkeypatch.setattr(SNN, "step", forbidden)
    target = tmp_path / "never-created"
    result = main(["--check", "--output", str(target)])
    assert len(result["run_ids"]) == 54
    assert not target.exists()


def test_cli_rejects_duplicate_fields_and_loads_custom_protocol(tmp_path):
    from examples.adaptation_protocol import main

    declaration = tmp_path / "input.json"
    declaration.write_text('{"seeds": [0], "seeds": [1]}')
    with pytest.raises(SystemExit):
        main(["--protocol", str(declaration), "--check"])
    p = engineering_protocol()
    p["seeds"] = [8, 9]
    declaration.write_text(json.dumps(p))
    assert len(main(["--protocol", str(declaration), "--check"])["run_ids"]) == 36


def test_source_digest_tracks_actual_edits_and_nonignored_files(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args])

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    (tmp_path / ".gitignore").write_text("logs/\n")
    source = tmp_path / "task.py"
    source.write_text("fixture = 1\n")
    git("add", ".")
    git("commit", "-qm", "fixture")
    clean = capture_provenance(tmp_path)
    assert clean["git_dirty"] is False
    source.write_text("fixture = 2\n")
    changed = capture_provenance(tmp_path)
    assert changed["git_revision"] == clean["git_revision"]
    assert changed["git_dirty"] is True
    assert changed["source_sha256"] != clean["source_sha256"]
    (tmp_path / "new_adapter.py").write_text("fixture = 3\n")
    added = capture_provenance(tmp_path)
    assert added["source_sha256"] != changed["source_sha256"]
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "ignored.json").write_text("{}")
    assert capture_provenance(tmp_path)["source_sha256"] == added["source_sha256"]
    source.unlink()
    assert any(f["kind"] == "deleted" for f in capture_provenance(tmp_path)["source_files"])


def test_failed_probe_cannot_mutate_its_source_branch(monkeypatch):
    import numpy as np
    from lisnn.experiments.cycle import _probe
    from lisnn.io import InterfacePort, RegimeController, StreamRuntime, Transducer
    from lisnn import create_nn
    from lisnn.synapses import create_synapses

    network = create_nn(2, neuron_type="lif", seed=1)
    network.configure_runtime(create_synapses(2, [0], [1], weights=0.5), dt=1,
                              impulse_scale=100, plasticity="off")
    regime = RegimeController(StreamRuntime(network, transducers={
        "input": Transducer(InterfacePort("input", 2, [0], source="EXTERNAL"))}))
    before = regime.snapshot()
    original = RegimeController.step

    def fail_on_second(self, *args, **kwargs):
        if self.runtime.tick == 1:
            raise RuntimeError("probe failed after its first tick")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RegimeController, "step", fail_on_second)
    with pytest.raises(RuntimeError, match="probe failed"):
        _probe(regime)
    assert regime.runtime.tick == before.runtime.tick == 0
    np.testing.assert_array_equal(regime.runtime.network.runtime.pool,
                                  before.runtime.network.runtime.pool)
    assert regime.runtime.deliveries == []


def test_execution_provenance_captures_active_topology_context(monkeypatch):
    from lisnn.synapses import get_topology_validation, topology_validation

    repository = Path(__file__).resolve().parents[1]
    original = get_topology_validation()
    # Changing the environment after package import does not change the actual
    # backend. A provenance record must report the runtime context instead.
    monkeypatch.setenv("LISNN_TOPOLOGY_VALIDATION", "not-the-active-backend")
    assert capture_provenance(repository)["topology_validation"] == original
    with topology_validation("reference"):
        assert get_topology_validation() == "reference"
        assert capture_provenance(repository)["topology_validation"] == "reference"
        with topology_validation("optimized"):
            assert get_topology_validation() == "optimized"
            assert capture_provenance(repository)["topology_validation"] == "optimized"
        assert get_topology_validation() == "reference"
    assert get_topology_validation() == original
