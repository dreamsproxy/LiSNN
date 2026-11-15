"""Synthetic evidence accounting only; these tests run no adaptation trials."""

import json

import pytest

from lisnn.experiments.evidence import EvidencePackage


def package(tmp_path, run_ids=None):
    return EvidencePackage.create(
        tmp_path / "evidence", {"run_ids": run_ids or ["pair-0", "pair-1"]},
        {"source_commit": "example-test-revision", "purpose": "unit-test"})


def test_plan_starts_pending_and_retains_failed_outcomes(tmp_path):
    store = package(tmp_path)
    before = store.summary()
    assert before["pending"] == ["pair-0", "pair-1"]
    assert before["execution_complete"] is False
    store.record("pair-0", status="completed", metrics={"response_rate": 0.0})
    store.record("pair-1", status="failed", error="declared numerical stop")
    after = EvidencePackage.open(store.path).summary()
    assert after["pending"] == []
    assert after["completed"] == ["pair-0"]
    assert after["failed"] == ["pair-1"]
    assert after["execution_complete"] is True
    assert "evidence_complete" not in after
    assert "learning_success" not in after


def test_all_failed_is_still_a_completed_plan(tmp_path):
    store = package(tmp_path)
    for run_id in ("pair-0", "pair-1"):
        store.record(run_id, status="failed", error="failed seed retained")
    assert store.summary()["execution_complete"] is True
    assert len(store.summary()["failed"]) == 2


def test_exclusive_package_and_run_publication(tmp_path):
    store = package(tmp_path)
    with pytest.raises(FileExistsError):
        package(tmp_path)
    with pytest.raises(ValueError, match="frozen protocol"):
        store.record("unplanned", status="completed", metrics={"quality": 0.5})
    store.record("pair-0", status="failed", error="original failure")
    with pytest.raises(FileExistsError):
        store.record("pair-0", status="completed", metrics={"quality": 1.0})
    assert json.loads((store.path / "runs/pair-0.json").read_text())["error"] == "original failure"


@pytest.mark.parametrize("ids", [[], ["duplicate", "duplicate"], ["../outside"],
                                      ["/absolute"], [""], ["hidden/slash"], [1]])
def test_invalid_planned_identifiers_rejected_before_writing(tmp_path, ids):
    path = tmp_path / "new"
    with pytest.raises(ValueError):
        EvidencePackage.create(path, {"run_ids": ids}, {"test": True})
    assert not path.exists()


@pytest.mark.parametrize("metrics", [None, {}, {"x": float("nan")}, {"x": float("inf")},
                                          {"x": True}, {"x": "0.4"}, {"": 1}, [1]])
def test_invalid_completion_metrics_leave_run_pending(tmp_path, metrics):
    store = package(tmp_path)
    with pytest.raises(ValueError):
        store.record("pair-0", status="completed", metrics=metrics)
    assert "pair-0" in store.summary()["pending"]


@pytest.mark.parametrize("kwargs", [dict(status="running"), dict(status="failed"),
                                        dict(status="failed", error="  "),
                                        dict(status="completed", metrics={"x": 0}, error="bad")])
def test_invalid_terminal_outcomes_are_not_recorded(tmp_path, kwargs):
    store = package(tmp_path)
    with pytest.raises(ValueError):
        store.record("pair-0", **kwargs)
    assert len(store.summary()["pending"]) == 2


def test_artifacts_hashed_and_modification_detected(tmp_path):
    store = package(tmp_path)
    raw = store.path / "raw"
    raw.mkdir()
    target = raw / "observations.jsonl"
    target.write_text('{"synthetic":true}\n')
    receipt = store.record("pair-0", status="completed", metrics={"quality": -0.5},
                           artifacts={"observations": "raw/observations.jsonl"})
    metadata = receipt["artifacts"]["observations"]
    assert metadata["size_bytes"] == target.stat().st_size
    assert len(metadata["sha256"]) == 64
    target.write_text("corrupted\n")
    with pytest.raises(ValueError, match="artifact integrity"):
        EvidencePackage.open(store.path)


@pytest.mark.parametrize("relative", ["../outside", "/tmp/outside", "raw/../raw/sample",
                                           "raw//sample", "raw\\sample", "protocol.json",
                                           "runs/pair-0.json", "missing", "."])
def test_artifact_paths_must_be_raw_files_inside_package(tmp_path, relative):
    store = package(tmp_path)
    with pytest.raises(ValueError):
        store.record("pair-0", status="failed", error="failure",
                     artifacts={"raw": relative})
    assert "pair-0" in store.summary()["pending"]


def test_symlink_artifact_and_run_directory_rejected(tmp_path):
    store = package(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sample").write_text("sample")
    (store.path / "raw").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        store.record("pair-0", status="failed", error="failure", artifacts={"raw": "raw/sample"})
    (store.path / "runs").rmdir()
    (store.path / "runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="ordinary directory"):
        EvidencePackage.open(store.path)


def test_protocol_manifest_and_receipt_corruption_detected(tmp_path):
    store = package(tmp_path)
    target = store.path / "protocol.json"
    original = target.read_text()
    target.write_text(json.dumps({"run_ids": ["pair-0", "different"]}))
    with pytest.raises(ValueError, match="manifest"):
        store.summary()
    target.write_text(original)
    store.record("pair-0", status="completed", metrics={"quality": 0})
    receipt = store.path / "runs/pair-0.json"
    contents = json.loads(receipt.read_text())
    contents["metrics"]["quality"] = 10
    receipt.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="receipt integrity"):
        store.summary()


def test_unplanned_receipt_cannot_hide_in_package(tmp_path):
    store = package(tmp_path)
    (store.path / "runs/unplanned.json").write_text("{}")
    with pytest.raises(ValueError, match="unplanned"):
        store.summary()


def test_manifest_provenance_modification_detected(tmp_path):
    store = package(tmp_path)
    target = store.path / "manifest.json"
    manifest = json.loads(target.read_text())
    assert len(manifest["manifest_sha256"]) == 64
    manifest["provenance"]["source_commit"] = "edited-revision"
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest integrity"):
        EvidencePackage.open(store.path)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {1: "not-a-JSON-key"}])
def test_non_json_or_nonfinite_protocol_rejected_before_writing(tmp_path, bad):
    with pytest.raises((ValueError, TypeError)):
        EvidencePackage.create(tmp_path / "new", {"run_ids": ["one"], "bad": bad}, {"test": True})
    assert not (tmp_path / "new").exists()


def test_duplicate_json_keys_and_infinite_literals_rejected(tmp_path):
    store = package(tmp_path)
    (store.path / "protocol.json").write_text('{"run_ids":["one"],"run_ids":["two"]}')
    with pytest.raises(ValueError, match="duplicate"):
        store.summary()
    (store.path / "protocol.json").write_text('{"run_ids":["one"],"overflow":1e10000}')
    with pytest.raises(ValueError):
        store.summary()
