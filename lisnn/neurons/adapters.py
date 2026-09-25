"""Explicit Izhikevich calibration between native rates and shared pA/mV units.

C_M is an input/output reference capacitance for this phenomenological model.
Converted currents are equivalent model contributions, not measured currents.
"""

import numpy as np

from lisnn.neurons import kernels as k
from lisnn.validation import vector32


def _capacitance(capacitance_pF):
    raw = np.asarray(capacitance_pF)
    if raw.ndim != 1:
        raise ValueError('reference capacitance must be a vector')
    cap = vector32('reference capacitance', raw, raw.size, scalar=False)
    if np.any(cap <= 0):
        raise ValueError('reference capacitance must be positive')
    return cap


def current_to_izhikevich(current_pA, capacitance_pF):
    """Convert pA to the native additive voltage-rate input convention."""
    cap = _capacitance(capacitance_pF)
    current = vector32('current_pA', current_pA, cap.size)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        return current / cap


def izhikevich_to_current(native_rate, capacitance_pF):
    """Convert a native voltage-rate contribution to equivalent pA."""
    cap = _capacitance(capacitance_pF)
    native = vector32('native_rate', native_rate, cap.size)
    with np.errstate(over='raise', invalid='raise'):
        return native * cap


def observe_izhikevich(neurons):
    """Return copied shared-unit observations for the supplied instant.

    recovery_current_pA includes the minus sign in the voltage equation (-u).
    intrinsic_current_pA is the polynomial voltage contribution, excluding
    recovery and applied input. These are diagnostic equivalents only.
    Voltage is already mV; spike events require no amplitude conversion.
    """
    k._check_population(neurons)
    voltage = vector32('voltage_mV', neurons[:, k.V], len(neurons), scalar=False)
    cap = _capacitance(neurons[:, k.C_M])
    with np.errstate(over='raise', invalid='raise'):
        intrinsic = np.float32(0.04) * voltage**2 + np.float32(5) * voltage + np.float32(140)
    return {
        'voltage_mV': voltage,
        'reference_capacitance_pF': cap,
        'recovery_current_pA': izhikevich_to_current(-neurons[:, k.ADAPT], cap),
        'intrinsic_current_pA': izhikevich_to_current(intrinsic, cap),
    }
