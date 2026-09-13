"""Hand-checkable causal pathway: python -m examples.fixed_weight_propagation."""

from dataclasses import asdict

import numpy as np

from lisnn.debugging.propagation_smoke import demonstration


def main():
    np.set_printoptions(precision=6, suppress=True)
    print('Edges: 0->2 (0.5), 0->3 (0.8), 1->2 (0.25), 2->3 (0.4)')
    print('dt=0.1 ms; J0=100 pA*ms; initial N2 voltage=-50.1 mV')
    for result in demonstration():
        print(f'\nTICK {result.tick}: {result.time_ms:.1f} to {result.time_ms + result.dt_ms:.1f} ms')
        for name, value in asdict(result).items():
            if name not in ('tick', 'time_ms', 'dt_ms'):
                print(f'{name}: {value}')


if __name__ == '__main__':
    main()
