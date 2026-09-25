"""Runner coverage includes failure continuation, reports and CLI exit status."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lisnn.debugging import runner


def test_all_smoke_suites(tmp_path):
    result = runner.run_all_smoke_tests(tmp_path, progress=False)
    assert result["passed"] is True, result["failed"]
    assert result["total"] == 8
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["passed"] is True
    assert set(saved["results"]) == set(runner.SMOKE_TESTS)
    assert (tmp_path / "population.log").exists()


def test_runner_continues_after_failures_and_returns_nonzero(monkeypatch, tmp_path):
    visited = []

    def crash(log_dir, verbose):
        raise RuntimeError("injected failure")

    def success(log_dir, verbose):
        visited.append("success")
        return {"passed": np.bool_(True), "values": np.array([1, 2])}

    monkeypatch.setattr(runner, "SMOKE_TESTS", {
        "crash": crash,
        "failure": lambda *_: {"passed": False, "checks": {"injected": False}},
        "malformed": lambda *_: {},
        "success": success,
    })
    assert runner.main(["--log-dir", str(tmp_path)]) == 1
    assert visited == ["success"]
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["failed"] == ["crash", "failure", "malformed"]
    assert "injected failure" in saved["results"]["crash"]["traceback"]
    assert saved["results"]["success"]["values"] == [1, 2]


def test_selection_does_not_run_other_suites(monkeypatch, tmp_path):
    def unexpected(*_):
        raise AssertionError("unselected suite ran")

    monkeypatch.setattr(runner, "SMOKE_TESTS", {
        "skip": unexpected, "selected": lambda *_: {"passed": True},
    })
    result = runner.run_all_smoke_tests(tmp_path, only=["selected"], progress=False)
    assert result["passed"] and list(result["results"]) == ["selected"]
    with pytest.raises(ValueError):
        runner.run_all_smoke_tests(tmp_path, only=[])


@pytest.mark.parametrize("entry", [["debug.py"], ["-m", "lisnn.debugging"]])
def test_cli_entrypoints(entry, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, *entry, "--only", "units", "indices", "--log-dir", str(tmp_path)],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2/2 suites passed" in result.stdout


def test_listing_has_no_run_side_effects(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "debug.py"), "--list"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.splitlines() == list(runner.SMOKE_TESTS)
    assert not list(tmp_path.iterdir())
