"""Shared float32 boundary validation without neuron-model dependencies."""

import numpy as np


def scalar32(name, value, *, positive=False, nonnegative=False):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in 'iuf':
        raise TypeError(f'{name} must be a real numeric scalar')
    if not np.isfinite(raw) or (positive and raw <= 0) or (nonnegative and raw < 0):
        raise ValueError(f'{name} must be finite with the required sign')
    with np.errstate(over='ignore', under='ignore'):
        result = np.float32(raw)
    if not np.isfinite(result) or (positive and result <= 0) or (raw != 0 and result == 0):
        raise ValueError(f'{name} is not representable as a finite nonzero float32')
    return result


def vector32(name, value, size, *, scalar=True):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf':
        raise TypeError(f'{name} must contain real numeric values')
    if raw.shape != (size,) and not (scalar and raw.ndim == 0):
        raise ValueError(f'{name} must have shape ({size},)' + (' or be scalar' if scalar else ''))
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.array(np.broadcast_to(raw, (size,)), dtype=np.float32, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must contain finite float32 values')
    return result
