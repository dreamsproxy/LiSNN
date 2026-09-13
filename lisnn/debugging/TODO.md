# Debugging TODO

- Completed: all-suite CLI/progress/reports and foundation/network/spatial smoke coverage.
- Register every future smoke suite in `runner.SMOKE_TESTS` so `python debug.py` runs it.

- Add smoke test for #18 static synaptic propagation/current accumulation.
- Add pre-reset plasticity-voltage/history smoke test for #19.
- Add Pair/Triplet/Voltage-STDP rule-specific smoke tests as M2 progresses.
- Keep each debugger independently callable and keep pytest regression wrappers small.
