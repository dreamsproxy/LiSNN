"""Regression test for the independently callable M2.1 smoke debugger."""

from lisnn.debugging.synapse_smoke import synapse_smoke_test


def test_synapse_smoke_test_passes() -> None:
    result = synapse_smoke_test(
        population=8,
        synapses_per_neuron=3,
        seed=1,
        verbose=False,
    )

    assert result["passed"] is True
    assert result["edge_count"] == 24
    assert all(result["checks"].values())
