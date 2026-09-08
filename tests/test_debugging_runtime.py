"""Runtime smoke checks for canonical LiSNN debugging APIs."""

from __future__ import annotations

from pathlib import Path

from lisnn.debugging import population_smoke_test, synapse_smoke_test


def test_population_smoke_runtime(tmp_path: Path) -> None:
    result = population_smoke_test(
        neuron_count=2,
        n_steps=1,
        watch_spikes=False,
        ouput_path=str(tmp_path / "population_smoke.log"),
        verbose=False,
    )

    assert result["passed"] is True
    assert Path(result["log_path"]).exists()


def test_synapse_smoke_runtime() -> None:
    result = synapse_smoke_test(
        population=8,
        synapses_per_neuron=3,
        seed=1,
        verbose=False,
    )

    assert result["passed"] is True
    assert result["edge_count"] == 24
