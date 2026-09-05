# Time-code -> frame baseline

This directory preserves the working associative video-recall mechanism recovered from `BioEmulator/OLD/SNN v0.2` before the DLiNN overhaul.

The baseline learns a discrete association:

```text
one-hot frame/time code -> recurrent spiking activity -> 32 x 32 grayscale frame
```

## Files

- `main.py`: video loading, temporal encoding, network training, recall, and visualization.
- `LIF.py`: leaky integrate-and-fire neuron step.
- `WeightMatrix.py`: recurrent propagation plus the original combined STDP/Hebbian update.

## Preservation policy

The learning rule and network dynamics are intentionally left structurally unchanged. Only two mechanical repairs were made while moving the code:

1. `self.n_inputs` is assigned in `Network.__init__`.
2. The example constructor call now supplies `n_inputs` and `encodings` explicitly.

These repairs make the archived intent executable without attempting to explain, simplify, or redesign why the mechanism produces recall. That analysis and refactor belong to the next stage.

## Usage

Place a video at `lichen.mp4` in this directory, then run:

```bash
python main.py
```

Current dependencies:

```text
numpy
numba
opencv-python
matplotlib
tqdm
```
