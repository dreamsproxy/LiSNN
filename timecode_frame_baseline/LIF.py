import numpy as np
from numba import njit

@njit()
def step(params, spike):
    params = params.astype(np.float64)
    p, dt, tau, v_rest, v_reset, v_threshold = params

    dV = (-(p - v_rest) + spike) * (dt / tau)
    new_p = p + dV

    fire = new_p >= v_threshold
    wp = new_p
    if fire:
        new_p = v_reset

    if wp < np.float64(1e-16) or np.isinf(wp):
        wp = np.float64(1e-16)
    if new_p < np.float64(1e-16) or np.isinf(new_p):
        new_p = np.float64(1e-16)

    return np.float64(wp), np.float64(new_p)

def parse_params(params: dict):
    potential = params["potential"]
    dt = params["dt"]
    tau = params["tau"]
    v_rest = params["v_rest"]
    v_reset = params["v_reset"]
    v_threshold = params["v_threshold"]
    return np.array(
        [potential, dt, tau, v_rest, v_reset, v_threshold]
    )
