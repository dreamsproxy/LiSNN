"""
NeuronModels.py

Pure NumPy vectorized neuron models for LiSNN.

Every neuron model receives the same population matrix:

    neurons = np.ndarray(
        dtype=np.float32,
        shape=(n_neurons, NEURON_WIDTH)
    )

Every row has the same column order. Individual neuron models simply ignore
the parameter/state families they do not use.

All step functions:
    1. operate on every neuron simultaneously using NumPy vectorization,
    2. mutate neuron state in-place,
    3. return float32 spike events of shape (n_neurons,),
    4. use Forward Euler integration for continuous differential equations.

Models:
    LIF
    AdaptiveLIF
    Izhikevich
    AdEx
    GLIF3
    GLIF4
    GLIF5
    CAdEx
    CAdEx-GLIF5 experimental hybrid
"""

import numpy as np


DTYPE = np.float32


# =============================================================================
# COMMON NEURON MATRIX LAYOUT
# =============================================================================
#
# Mutable dynamic state
#
# ADAPT changes meaning by neuron family:
#
#   AdaptiveLIF -> adaptive threshold state
#   Izhikevich  -> recovery variable u
#   AdEx        -> adaptation current w
#   CAdEx       -> adaptation conductance g_A
#
# GLIF-specific state columns remain allocated for every neuron type.
# Models which do not require them simply ignore them.
#

V = 0
ADAPT = 1
THETA_S = 2
ASC_1 = 3
ASC_2 = 4
THETA_V = 5
REFRACTORY = 6

STATE = slice(0, 7)


# -----------------------------------------------------------------------------
# Common electrical parameters
# -----------------------------------------------------------------------------

C_M = 7
R_M = 8
G_L = 9
E_L = 10

V_RESET = 11
THETA_INF = 12
T_REF = 13
V_DETECT = 14

CORE = slice(7, 15)


# -----------------------------------------------------------------------------
# Adaptive LIF
# -----------------------------------------------------------------------------

TAU_ADAPT = 15
DELTA_ADAPT = 16

ALIF = slice(15, 17)


# -----------------------------------------------------------------------------
# Izhikevich
# -----------------------------------------------------------------------------

IZH_A = 17
IZH_B = 18
IZH_D = 19

IZH = slice(17, 20)


# -----------------------------------------------------------------------------
# Exponential spike initiation
#
# Shared by AdEx and CAdEx.
# -----------------------------------------------------------------------------

V_T = 20
DELTA_T = 21

EXP = slice(20, 22)


# -----------------------------------------------------------------------------
# AdEx adaptation-current parameters
# -----------------------------------------------------------------------------

TAU_W = 22
A_W = 23
B_W = 24

ADEX = slice(22, 25)


# -----------------------------------------------------------------------------
# GLIF parameters
# -----------------------------------------------------------------------------
#
# Voltage reset:
#
#   V(t+) =
#       E_L
#       + F_V * (V(t-) - E_L)
#       - DELTA_V
#
# Spike-dependent threshold:
#
#   dTHETA_S/dt =
#       -B_S * THETA_S
#
# After-spike currents:
#
#   dASC_1/dt = -K_1 * ASC_1
#   dASC_2/dt = -K_2 * ASC_2
#
# After spike:
#
#   ASC_j(t+) =
#       F_j * ASC_j(t-)
#       + DELTA_I_j
#
# Voltage-dependent threshold:
#
#   dTHETA_V/dt =
#       A_V * (V - E_L)
#       - B_V * THETA_V
#

F_V = 25
DELTA_V = 26

B_S = 27
DELTA_THETA_S = 28

K_1 = 29
DELTA_I_1 = 30
F_1 = 31

K_2 = 32
DELTA_I_2 = 33
F_2 = 34

A_V = 35
B_V = 36

GLIF = slice(25, 37)


# -----------------------------------------------------------------------------
# CAdEx conductance adaptation
# -----------------------------------------------------------------------------

E_A = 37

TAU_G_A = 38
G_A_BAR = 39

V_A = 40
DELTA_A = 41

DELTA_G_A = 42

CADEX = slice(37, 43)


# -----------------------------------------------------------------------------
# Entire row width
# -----------------------------------------------------------------------------

NEURON_WIDTH = 43


COLUMN_NAMES = (
    "V",
    "ADAPT",
    "THETA_S",
    "ASC_1",
    "ASC_2",
    "THETA_V",
    "REFRACTORY",

    "C_M",
    "R_M",
    "G_L",
    "E_L",
    "V_RESET",
    "THETA_INF",
    "T_REF",
    "V_DETECT",

    "TAU_ADAPT",
    "DELTA_ADAPT",

    "IZH_A",
    "IZH_B",
    "IZH_D",

    "V_T",
    "DELTA_T",

    "TAU_W",
    "A_W",
    "B_W",

    "F_V",
    "DELTA_V",
    "B_S",
    "DELTA_THETA_S",

    "K_1",
    "DELTA_I_1",
    "F_1",

    "K_2",
    "DELTA_I_2",
    "F_2",

    "A_V",
    "B_V",

    "E_A",
    "TAU_G_A",
    "G_A_BAR",
    "V_A",
    "DELTA_A",
    "DELTA_G_A",
)


SLICES = {
    "STATE": STATE,
    "CORE": CORE,
    "ALIF": ALIF,
    "IZH": IZH,
    "EXP": EXP,
    "ADEX": ADEX,
    "GLIF": GLIF,
    "CADEX": CADEX,
}

# =============================================================================
# DEFAULT PARAMETER RANGES
# =============================================================================
#
# Format:
#     parameter_index: (default, random_min, random_max)
#
# Dynamic state values are initialized separately.
#
# Units follow each underlying model's native/common conventions:
#     voltage      -> mV
#     time         -> ms
#     capacitance  -> model-compatible membrane units
#     conductance  -> model-compatible conductance units
#     current      -> model-compatible current units
#

PARAMETER_DEFAULTS = {
    # -------------------------------------------------------------------------
    # Common electrical parameters
    # -------------------------------------------------------------------------
    C_M: (
        200.0,
        100.0,
        300.0,
    ),

    R_M: (
        100.0,
        50.0,
        200.0,
    ),

    G_L: (
        10.0,
        5.0,
        20.0,
    ),

    E_L: (
        -65.0,
        -72.0,
        -58.0,
    ),

    V_RESET: (
        -65.0,
        -72.0,
        -58.0,
    ),

    THETA_INF: (
        -50.0,
        -55.0,
        -45.0,
    ),

    T_REF: (
        2.0,
        1.0,
        5.0,
    ),

    V_DETECT: (
        -30.0,
        -40.0,
        -20.0,
    ),

    # -------------------------------------------------------------------------
    # Adaptive LIF
    # -------------------------------------------------------------------------
    TAU_ADAPT: (
        100.0,
        50.0,
        300.0,
    ),

    DELTA_ADAPT: (
        2.0,
        0.5,
        5.0,
    ),

    # -------------------------------------------------------------------------
    # Izhikevich
    #
    # Default corresponds roughly to regular-spiking behavior:
    #
    #     a = 0.02
    #     b = 0.2
    #     c = -65
    #     d = 8
    #
    # V_RESET already provides c.
    # -------------------------------------------------------------------------
    IZH_A: (
        0.02,
        0.01,
        0.1,
    ),

    IZH_B: (
        0.20,
        0.15,
        0.30,
    ),

    IZH_D: (
        8.0,
        2.0,
        10.0,
    ),

    # -------------------------------------------------------------------------
    # Exponential spike-initiation parameters
    # -------------------------------------------------------------------------
    V_T: (
        -50.0,
        -55.0,
        -45.0,
    ),

    DELTA_T: (
        2.0,
        0.5,
        5.0,
    ),

    # -------------------------------------------------------------------------
    # AdEx
    # -------------------------------------------------------------------------
    TAU_W: (
        150.0,
        50.0,
        300.0,
    ),

    A_W: (
        2.0,
        0.0,
        6.0,
    ),

    B_W: (
        40.0,
        5.0,
        100.0,
    ),

    # -------------------------------------------------------------------------
    # GLIF reset
    # -------------------------------------------------------------------------
    F_V: (
        0.5,
        0.0,
        1.0,
    ),

    DELTA_V: (
        5.0,
        0.0,
        10.0,
    ),

    # -------------------------------------------------------------------------
    # GLIF spike-dependent threshold
    # -------------------------------------------------------------------------
    B_S: (
        0.02,
        0.005,
        0.05,
    ),

    DELTA_THETA_S: (
        5.0,
        1.0,
        10.0,
    ),

    # -------------------------------------------------------------------------
    # GLIF after-spike current 1
    #
    # Faster component.
    # -------------------------------------------------------------------------
    K_1: (
        0.10,
        0.05,
        0.20,
    ),

    DELTA_I_1: (
        20.0,
        5.0,
        50.0,
    ),

    F_1: (
        0.5,
        0.0,
        1.0,
    ),

    # -------------------------------------------------------------------------
    # GLIF after-spike current 2
    #
    # Slower component.
    # -------------------------------------------------------------------------
    K_2: (
        0.01,
        0.002,
        0.05,
    ),

    DELTA_I_2: (
        5.0,
        1.0,
        20.0,
    ),

    F_2: (
        0.8,
        0.2,
        1.0,
    ),

    # -------------------------------------------------------------------------
    # GLIF voltage-dependent threshold
    # -------------------------------------------------------------------------
    A_V: (
        0.01,
        0.001,
        0.05,
    ),

    B_V: (
        0.02,
        0.005,
        0.05,
    ),

    # -------------------------------------------------------------------------
    # CAdEx conductance adaptation
    # -------------------------------------------------------------------------
    E_A: (
        -80.0,
        -90.0,
        -70.0,
    ),

    TAU_G_A: (
        150.0,
        50.0,
        400.0,
    ),

    G_A_BAR: (
        10.0,
        1.0,
        30.0,
    ),

    V_A: (
        -50.0,
        -60.0,
        -40.0,
    ),

    DELTA_A: (
        5.0,
        1.0,
        10.0,
    ),

    DELTA_G_A: (
        2.0,
        0.1,
        10.0,
    ),
}

# =============================================================================
# POPULATION HELPERS
# =============================================================================

def initialize_population_parameters(
    neurons,
    randomize_params=False,
    rng=None,
):
    """
    Initialize all neuron parameter families.

    Every population receives valid parameters for:

        LIF
        AdaptiveLIF
        Izhikevich
        AdEx
        GLIF3
        GLIF4
        GLIF5
        CAdEx
        CAdEx-GLIF

    Unused parameters are simply ignored by the selected neuron model.


    PSEUDOCODE:

        if RNG not provided:
            create NumPy random generator

        for each parameter column:

            if randomization enabled:
                sample every neuron independently
                between realistic min/max

            else:
                fill every neuron with default value

        initialize dynamic state:

            V <- E_L

            ADAPT <- model-neutral zero

            THETA_S <- 0

            ASC_1 <- 0

            ASC_2 <- 0

            THETA_V <- 0

            REFRACTORY <- 0

        return neurons
    """

    if rng is None:
        rng = np.random.default_rng()

    n_neurons = neurons.shape[0]

    for index, (
        default,
        minimum,
        maximum,
    ) in PARAMETER_DEFAULTS.items():

        if randomize_params:

            neurons[:, index] = rng.uniform(
                minimum,
                maximum,
                size=n_neurons,
            ).astype(DTYPE)

        else:

            neurons[:, index] = DTYPE(
                default
            )

    # -------------------------------------------------------------------------
    # Initial dynamic state
    # -------------------------------------------------------------------------

    neurons[:, V] = (
        neurons[:, E_L]
    )

    neurons[:, ADAPT] = DTYPE(0.0)

    neurons[:, THETA_S] = DTYPE(0.0)

    neurons[:, ASC_1] = DTYPE(0.0)

    neurons[:, ASC_2] = DTYPE(0.0)

    neurons[:, THETA_V] = DTYPE(0.0)

    neurons[:, REFRACTORY] = DTYPE(0.0)

    return neurons

def new_population(
    n_neurons,
    fill=0.0,
    randomize_params=False,
    seed=None,
):
    """
    Create a common-layout neuron population and initialize its parameters.

    PSEUDOCODE:

        allocate:
            [N, NEURON_WIDTH]

        fill matrix

        create seeded RNG if requested

        initialize all neuron-model parameter families

        initialize dynamic state

        return population


    EXAMPLES:

        pool = nm.new_population(
            8,
            fill=0.0,
            randomize_params=False,
        )


        pool = nm.new_population(
            1000,
            randomize_params=True,
        )


        pool = nm.new_population(
            1000,
            randomize_params=True,
            seed=42,
        )
    """

    neurons = np.full(
        (
            int(n_neurons),
            NEURON_WIDTH,
        ),
        DTYPE(fill),
        dtype=DTYPE,
    )

    rng = np.random.default_rng(
        seed
    )

    return initialize_population_parameters(
        neurons,
        randomize_params=randomize_params,
        rng=rng,
    )


def _check_population(neurons):
    """
    Validate population representation.

    PSEUDOCODE:
        assert NumPy array
        assert rank == 2
        assert columns >= NEURON_WIDTH
        assert dtype == float32
    """

    if not isinstance(neurons, np.ndarray):
        raise TypeError(
            "neurons must be a NumPy ndarray"
        )

    if neurons.ndim != 2:
        raise ValueError(
            "neurons must have rank 2"
        )

    if neurons.shape[1] < NEURON_WIDTH:
        raise ValueError(
            f"neurons require at least {NEURON_WIDTH} columns, "
            f"got {neurons.shape[1]}"
        )

    if neurons.dtype != np.float32:
        raise TypeError(
            f"neurons must use np.float32, got {neurons.dtype}"
        )


def _input_vector(input_current, n_neurons):
    """
    Convert scalar or N-element input into float32 vector.

    PSEUDOCODE:
        if scalar:
            broadcast scalar across N neurons
        otherwise:
            verify shape == [N]
        return float32 input
    """

    current = np.asarray(
        input_current,
        dtype=DTYPE,
    )

    if current.ndim == 0:
        return np.full(
            n_neurons,
            current,
            dtype=DTYPE,
        )

    if current.shape != (n_neurons,):
        raise ValueError(
            f"input_current must be scalar or shape "
            f"({n_neurons},), got {current.shape}"
        )

    return current


def _refractory_active(neurons, dt):
    """
    Advance refractory timers.

    PSEUDOCODE:
        active <- refractory <= 0
        refractory <- max(
            refractory - dt,
            0
        )
        return active

    Neurons refractory at the beginning of a timestep remain refractory
    throughout that timestep.
    """

    refractory = neurons[:, REFRACTORY]

    active = refractory <= DTYPE(0.0)

    refractory[...] = np.maximum(
        DTYPE(0.0),
        refractory - dt,
    )

    return active


def _spike_vector(mask):
    """
    Boolean spike mask -> float32 event vector.
    """

    return mask.astype(
        DTYPE,
        copy=False,
    )


# =============================================================================
# LIF
# =============================================================================

def lif_step(
    neurons,
    input_current,
    dt,
):
    """
    Vectorized Leaky Integrate-and-Fire.

    EQUATIONS:

        C_m dV/dt =
            I
            - (V - E_L) / R_m

    SPIKE:

        if V >= THETA_INF:

            V <- V_RESET
            refractory <- T_REF


    PSEUDOCODE:

        current <- broadcast input

        active <- refractory <= 0

        update refractory timer

        dV <- (
            current
            - (V - E_L) / R_m
        ) / C_m

        V_candidate <- V + dt*dV

        if refractory:
            V <- V_RESET
        else:
            V <- V_candidate

        spike <- (
            active
            AND
            V >= THETA_INF
        )

        if spike:
            V <- V_RESET
            refractory <- T_REF

        return spike
    """

    _check_population(neurons)

    dt = DTYPE(dt)

    current = _input_vector(
        input_current,
        neurons.shape[0],
    )

    active = _refractory_active(
        neurons,
        dt,
    )

    v = neurons[:, V]

    dv = (
        current
        - (
            v
            - neurons[:, E_L]
        )
        / neurons[:, R_M]
    ) / neurons[:, C_M]

    v_candidate = (
        v
        + dt * dv
    )

    v_next = np.where(
        active,
        v_candidate,
        neurons[:, V_RESET],
    )

    spikes = (
        active
        & (
            v_next
            >= neurons[:, THETA_INF]
        )
    )

    v_next[spikes] = (
        neurons[spikes, V_RESET]
    )

    neurons[
        spikes,
        REFRACTORY,
    ] = neurons[
        spikes,
        T_REF,
    ]

    neurons[:, V] = v_next

    return _spike_vector(spikes)


# =============================================================================
# ADAPTIVE LIF
# =============================================================================

def adaptive_lif_step(
    neurons,
    input_current,
    dt,
):
    """
    Vectorized Adaptive LIF.

    ADAPT stores the spike-triggered adaptive threshold component.


    EQUATIONS:

        C_m dV/dt =
            I
            - (V - E_L) / R_m

        dA/dt =
            -A / TAU_ADAPT

        theta(t) =
            THETA_INF
            + A(t)


    SPIKE:

        if V >= theta:

            V <- V_RESET

            A <-
                A
                + DELTA_ADAPT

            refractory <- T_REF


    PSEUDOCODE:

        current <- broadcast input

        active <- refractory <= 0

        update refractory timer

        A <- A + dt*(-A/tau_A)

        theta <- THETA_INF + A

        dV <- (
            current
            - (V-E_L)/R_m
        ) / C_m

        V_candidate <- V + dt*dV

        if refractory:
            V <- V_RESET
        else:
            V <- V_candidate

        spike <- (
            active
            AND
            V >= theta
        )

        if spike:
            V <- V_RESET
            A <- A + DELTA_ADAPT
            refractory <- T_REF

        return spike
    """

    _check_population(neurons)

    dt = DTYPE(dt)

    current = _input_vector(
        input_current,
        neurons.shape[0],
    )

    active = _refractory_active(
        neurons,
        dt,
    )

    v = neurons[:, V]
    adapt = neurons[:, ADAPT]

    adapt_next = (
        adapt
        + dt
        * (
            -adapt
            / neurons[:, TAU_ADAPT]
        )
    )

    threshold = (
        neurons[:, THETA_INF]
        + adapt_next
    )

    dv = (
        current
        - (
            v
            - neurons[:, E_L]
        )
        / neurons[:, R_M]
    ) / neurons[:, C_M]

    v_candidate = (
        v
        + dt * dv
    )

    v_next = np.where(
        active,
        v_candidate,
        neurons[:, V_RESET],
    )

    spikes = (
        active
        & (
            v_next
            >= threshold
        )
    )

    v_next[spikes] = (
        neurons[spikes, V_RESET]
    )

    adapt_next[spikes] += (
        neurons[
            spikes,
            DELTA_ADAPT,
        ]
    )

    neurons[
        spikes,
        REFRACTORY,
    ] = neurons[
        spikes,
        T_REF,
    ]

    neurons[:, V] = v_next
    neurons[:, ADAPT] = adapt_next

    return _spike_vector(spikes)


# =============================================================================
# IZHIKEVICH
# =============================================================================

def izhikevich_step(
    neurons,
    input_current,
    dt,
):
    """
    Vectorized canonical Izhikevich neuron.

    ADAPT stores recovery variable u.


    EQUATIONS:

        dV/dt =
            0.04*V^2
            + 5*V
            + 140
            - u
            + I

        du/dt =
            a*(b*V - u)


    PARAMETER MAPPING:

        a -> IZH_A

        b -> IZH_B

        c -> V_RESET

        d -> IZH_D

        spike detection voltage
            -> V_DETECT


    SPIKE:

        if V >= V_DETECT:

            V <- V_RESET

            u <- u + IZH_D


    PSEUDOCODE:

        current <- broadcast input

        read V

        read u

        dV <- (
            0.04*V^2
            + 5*V
            + 140
            - u
            + current
        )

        du <- a*(b*V-u)

        V <- V + dt*dV

        u <- u + dt*du

        spike <- V >= V_DETECT

        if spike:
            V <- V_RESET
            u <- u + d

        return spike


    NOTE:

        The original differential equations are retained.

        This implementation uses ordinary Forward Euler integration rather
        than the two half-step numerical trick used in Izhikevich's example
        MATLAB implementation.
    """

    _check_population(neurons)

    dt = DTYPE(dt)

    current = _input_vector(
        input_current,
        neurons.shape[0],
    )

    v = neurons[:, V]
    u = neurons[:, ADAPT]

    dv = (
        DTYPE(0.04)
        * v
        * v

        + DTYPE(5.0)
        * v

        + DTYPE(140.0)

        - u

        + current
    )

    du = (
        neurons[:, IZH_A]
        * (
            neurons[:, IZH_B]
            * v
            - u
        )
    )

    v_next = (
        v
        + dt * dv
    )

    u_next = (
        u
        + dt * du
    )

    spikes = (
        v_next
        >= neurons[:, V_DETECT]
    )

    v_next[spikes] = (
        neurons[
            spikes,
            V_RESET,
        ]
    )

    u_next[spikes] += (
        neurons[
            spikes,
            IZH_D,
        ]
    )

    neurons[:, V] = v_next
    neurons[:, ADAPT] = u_next

    return _spike_vector(spikes)


# =============================================================================
# ADEX
# =============================================================================

def adex_step(
    neurons,
    input_current,
    dt,
):
    """
    Vectorized Adaptive Exponential Integrate-and-Fire.


    ADAPT stores adaptation current w.


    EQUATIONS:

        C_m dV/dt =

            g_L*(E_L - V)

            + g_L*DELTA_T
              * exp(
                    (V - V_T)
                    / DELTA_T
                )

            - w

            + I


        TAU_W dw/dt =

            A_W*(V - E_L)

            - w


    SPIKE:

        if V >= V_DETECT:

            V <- V_RESET

            w <- w + B_W

            refractory <- T_REF


    PSEUDOCODE:

        current <- broadcast input

        active <- refractory <= 0

        update refractory timer

        exponential <- (
            g_L
            * DELTA_T
            * exp(
                (V-V_T)
                / DELTA_T
            )
        )

        dV <- (
            leak
            + exponential
            - w
            + current
        ) / C_m

        dw <- (
            A_W*(V-E_L)
            - w
        ) / TAU_W

        V_candidate <- V + dt*dV

        w <- w + dt*dw

        if refractory:
            V <- V_RESET

        spike <- (
            active
            AND
            V_candidate >= V_DETECT
        )

        if spike:
            V <- V_RESET
            w <- w + B_W
            refractory <- T_REF

        return spike
    """

    _check_population(neurons)

    dt = DTYPE(dt)

    current = _input_vector(
        input_current,
        neurons.shape[0],
    )

    active = _refractory_active(
        neurons,
        dt,
    )

    v = neurons[:, V]
    w = neurons[:, ADAPT]

    g_l = neurons[:, G_L]
    e_l = neurons[:, E_L]

    delta_t = neurons[:, DELTA_T]

    exponential = (
        g_l
        * delta_t
        * np.exp(
            (
                v
                - neurons[:, V_T]
            )
            / delta_t
        )
    )

    dv = (
        g_l
        * (
            e_l
            - v
        )

        + exponential

        - w

        + current
    ) / neurons[:, C_M]

    dw = (
        neurons[:, A_W]
        * (
            v
            - e_l
        )

        - w
    ) / neurons[:, TAU_W]

    v_candidate = (
        v
        + dt * dv
    )

    w_next = (
        w
        + dt * dw
    )

    v_next = np.where(
        active,
        v_candidate,
        neurons[:, V_RESET],
    )

    spikes = (
        active
        & (
            v_candidate
            >= neurons[:, V_DETECT]
        )
    )

    v_next[spikes] = (
        neurons[
            spikes,
            V_RESET,
        ]
    )

    w_next[spikes] += (
        neurons[
            spikes,
            B_W,
        ]
    )

    neurons[
        spikes,
        REFRACTORY,
    ] = neurons[
        spikes,
        T_REF,
    ]

    neurons[:, V] = v_next
    neurons[:, ADAPT] = w_next

    return _spike_vector(spikes)


# =============================================================================
# GLIF INTERNAL FUNCTIONS
# =============================================================================

def _glif_voltage_derivative(
    neurons,
    current,
):
    """
    GLIF membrane equation.

    PSEUDOCODE:

        dV <- (
            external_current
            + ASC_1
            + ASC_2
            - (V-E_L)/R
        ) / C

        return dV
    """

    return (
        current

        + neurons[:, ASC_1]

        + neurons[:, ASC_2]

        - (
            neurons[:, V]
            - neurons[:, E_L]
        )
        / neurons[:, R_M]

    ) / neurons[:, C_M]


def _glif_asc_euler(
    neurons,
    dt,
):
    """
    Integrate both GLIF after-spike currents.

    EQUATIONS:

        dASC_1/dt =
            -K_1 * ASC_1

        dASC_2/dt =
            -K_2 * ASC_2


    PSEUDOCODE:

        ASC_1 <- (
            ASC_1
            + dt*(-K_1*ASC_1)
        )

        ASC_2 <- (
            ASC_2
            + dt*(-K_2*ASC_2)
        )

        return ASC_1, ASC_2
    """

    asc1_next = (
        neurons[:, ASC_1]

        + dt
        * (
            -neurons[:, K_1]
            * neurons[:, ASC_1]
        )
    )

    asc2_next = (
        neurons[:, ASC_2]

        + dt
        * (
            -neurons[:, K_2]
            * neurons[:, ASC_2]
        )
    )

    return (
        asc1_next,
        asc2_next,
    )


def _glif_apply_asc_spike_reset(
    neurons,
    spikes,
    asc1_next,
    asc2_next,
):
    """
    GLIF after-spike current reset/increment.

    EQUATIONS:

        ASC_1(t+) =
            F_1*ASC_1(t-)
            + DELTA_I_1

        ASC_2(t+) =
            F_2*ASC_2(t-)
            + DELTA_I_2


    PSEUDOCODE:

        for all spiking neurons simultaneously:

            ASC_1 <- (
                F_1*ASC_1
                + DELTA_I_1
            )

            ASC_2 <- (
                F_2*ASC_2
                + DELTA_I_2
            )
    """

    asc1_next[spikes] = (
        neurons[
            spikes,
            F_1,
        ]
        * asc1_next[spikes]

        + neurons[
            spikes,
            DELTA_I_1,
        ]
    )

    asc2_next[spikes] = (
        neurons[
            spikes,
            F_2,
        ]
        * asc2_next[spikes]

        + neurons[
            spikes,
            DELTA_I_2,
        ]
    )


# =============================================================================
# GLIF3
# =============================================================================

def glif3_step(
    neurons,
    input_current,
    dt,
):
    """
    Vectorized Allen GLIF3 / LIF_ASC.


    STATE:

        V
        ASC_1
        ASC_2


    EQUATIONS:

        dV/dt =

            (
                I_e
                + ASC_1
                + ASC_2
                - (V-E_L)/R
            ) / C


        dASC_1/dt =
            -K_1*ASC_1


        dASC_2/dt =
            -K_2*ASC_2


    THRESHOLD:

        theta =
            THETA_INF


    SPIKE RESET:

        V <- E_L


        ASC_1 <-
            F_1*ASC_1
            + DELTA_I_1


        ASC_2 <-
            F_2*ASC_2
            + DELTA_I_2


        refractory <- T_REF


    PSEUDOCODE:

        current <- broadcast input

        active <- refractory <= 0

        update refractory timer

        integrate ASC_1

        integrate ASC_2

        integrate membrane voltage

        spike <- (
            active
            AND
            V_candidate >= THETA_INF
        )

        if spike:
            V <- E_L
            update ASC_1
            update ASC_2
            refractory <- T_REF

        write state

        return spike
    """

    _check_population(neurons)

    dt = DTYPE(dt)

    current = _input_vector(
        input_current,
        neurons.shape[0],
    )

    active = _refractory_active(
        neurons,
        dt,
    )

    dv = _glif_voltage_derivative(
        neurons,
        current,
    )

    (
        asc1_next,
        asc2_next,
    ) = _glif_asc_euler(
        neurons,
        dt,
    )

    v_candidate = (
        neurons[:, V]
        + dt * dv
    )

    v_next = np.where(
        active,
        v_candidate,
        neurons[:, E_L],
    )

    spikes = (
        active
        & (
            v_candidate
            >= neurons[:, THETA_INF]
        )
    )

    v_next[spikes] = (
        neurons[
            spikes,
            E_L,
        ]
    )

    _glif_apply_asc_spike_reset(
        neurons,
        spikes,
        asc1_next,
        asc2_next,
    )

    neurons[
        spikes,
        REFRACTORY,
    ] = neurons[
        spikes,
        T_REF,
    ]

    neurons[:, V] = v_next
    neurons[:, ASC_1] = asc1_next
    neurons[:, ASC_2] = asc2_next

    return _spike_vector(spikes)


# =============================================================================
# GLIF4
# =============================================================================

def glif4_step(
    neurons,
    input_current,
    dt,
):
    """
    Vectorized Allen GLIF4 / LIF_R_ASC.


    STATE:

        V

        THETA_S

        ASC_1

        ASC_2


    EQUATIONS:

        dV/dt =

            (
                I_e
                + ASC_1
                + ASC_2
                - (V-E_L)/R
            ) / C


        dTHETA_S/dt =
            -B_S*THETA_S


        dASC_1/dt =
            -K_1*ASC_1


        dASC_2/dt =
            -K_2*ASC_2


    THRESHOLD:

        theta =
            THETA_INF
            + THETA_S


    SPIKE RESET:

        V(t+) =

            E_L

            + F_V
              * (
                    V(t-)
                    - E_L
                )

            - DELTA_V


        THETA_S(t+) =

            THETA_S(t-)

            + DELTA_THETA_S


        ASC_1(t+) =

            F_1*ASC_1(t-)

            + DELTA_I_1


        ASC_2(t+) =

            F_2*ASC_2(t-)

            + DELTA_I_2


        refractory <- T_REF


    PSEUDOCODE:

        current <- broadcast input

        active <- refractory <= 0

        update refractory timer

        preserve pre-spike V

        decay THETA_S

        decay ASC_1

        decay ASC_2

        integrate membrane

        theta <- (
            THETA_INF
            + THETA_S
        )

        spike <- (
            active
            AND
            V_candidate >= theta
        )

        if spike:
            apply GLIF voltage reset
            increment THETA_S
            update ASC_1
            update ASC_2
            refractory <- T_REF

        write all state

        return spike
    """

    _check_population(neurons)

    dt = DTYPE(dt)

    current = _input_vector(
        input_current,
        neurons.shape[0],
    )

    active = _refractory_active(
        neurons,
        dt,
    )

    v_pre = neurons[:, V].copy()

    dv = _glif_voltage_derivative(
        neurons,
        current,
    )

    theta_s_next = (
        neurons[:, THETA_S]

        + dt
        * (
            -neurons[:, B_S]
            * neurons[:, THETA_S]
        )
    )

    (
        asc1_next,
        asc2_next,
    ) = _glif_asc_euler(
        neurons,
        dt,
    )

    v_candidate = (
        neurons[:, V]
        + dt * dv
    )

    threshold = (
        neurons[:, THETA_INF]
        + theta_s_next
    )

    v_next = np.where(
        active,
        v_candidate,
        neurons[:, V],
    )

    spikes = (
        active
        & (
            v_candidate
            >= threshold
        )
    )

    v_next[spikes] = (
        neurons[
            spikes,
            E_L,
        ]

        + neurons[
            spikes,
            F_V,
        ]
        * (
            v_pre[spikes]
            - neurons[
                spikes,
                E_L,
            ]
        )

        - neurons[
            spikes,
            DELTA_V,
        ]
    )

    theta_s_next[spikes] += (
        neurons[
            spikes,
            DELTA_THETA_S,
        ]
    )

    _glif_apply_asc_spike_reset(
        neurons,
        spikes,
        asc1_next,
        asc2_next,
    )

    neurons[
        spikes,
        REFRACTORY,
    ] = neurons[
        spikes,
        T_REF,
    ]

    neurons[:, V] = v_next
    neurons[:, THETA_S] = theta_s_next
    neurons[:, ASC_1] = asc1_next
    neurons[:, ASC_2] = asc2_next

    return _spike_vector(spikes)


# =============================================================================
# GLIF5
# =============================================================================

def glif5_step(
    neurons,
    input_current,
    dt,
):
    """
    Vectorized Allen GLIF5 / LIF_R_ASC_A.


    STATE:

        V

        THETA_S

        ASC_1

        ASC_2

        THETA_V


    EQUATIONS:

        dV/dt =

            (
                I_e
                + ASC_1
                + ASC_2
                - (V-E_L)/R
            ) / C


        dTHETA_S/dt =

            -B_S*THETA_S


        dASC_1/dt =

            -K_1*ASC_1


        dASC_2/dt =

            -K_2*ASC_2


        dTHETA_V/dt =

            A_V*(V-E_L)

            - B_V*THETA_V


    THRESHOLD:

        theta =

            THETA_INF

            + THETA_S

            + THETA_V


    SPIKE RESET:

        V(t+) =

            E_L

            + F_V
              * (
                    V(t-)
                    - E_L
                )

            - DELTA_V


        THETA_S(t+) =

            THETA_S(t-)

            + DELTA_THETA_S


        ASC_1(t+) =

            F_1*ASC_1(t-)

            + DELTA_I_1


        ASC_2(t+) =

            F_2*ASC_2(t-)

            + DELTA_I_2


        THETA_V(t+) =

            THETA_V(t-)


        refractory <- T_REF


    PSEUDOCODE:

        current <- broadcast input

        active <- refractory <= 0

        update refractory timer

        preserve pre-spike voltage

        integrate THETA_S

        integrate ASC_1

        integrate ASC_2

        integrate THETA_V

        integrate membrane voltage

        theta <- (
            THETA_INF
            + THETA_S
            + THETA_V
        )

        spike <- (
            active
            AND
            V_candidate >= theta
        )

        if spike:
            apply biological voltage reset
            increment THETA_S
            update both after-spike currents
            leave THETA_V continuous
            refractory <- T_REF

        write all state

        return spike
    """

    _check_population(neurons)

    dt = DTYPE(dt)

    current = _input_vector(
        input_current,
        neurons.shape[0],
    )

    active = _refractory_active(
        neurons,
        dt,
    )

    v_pre = neurons[:, V].copy()

    dv = _glif_voltage_derivative(
        neurons,
        current,
    )

    theta_s_next = (
        neurons[:, THETA_S]

        + dt
        * (
            -neurons[:, B_S]
            * neurons[:, THETA_S]
        )
    )

    (
        asc1_next,
        asc2_next,
    ) = _glif_asc_euler(
        neurons,
        dt,
    )

    theta_v_next = (
        neurons[:, THETA_V]

        + dt
        * (
            neurons[:, A_V]
            * (
                neurons[:, V]
                - neurons[:, E_L]
            )

            - neurons[:, B_V]
            * neurons[:, THETA_V]
        )
    )

    v_candidate = (
        neurons[:, V]
        + dt * dv
    )

    threshold = (
        neurons[:, THETA_INF]
        + theta_s_next
        + theta_v_next
    )

    v_next = np.where(
        active,
        v_candidate,
        neurons[:, V],
    )

    spikes = (
        active
        & (
            v_candidate
            >= threshold
        )
    )

    v_next[spikes] = (
        neurons[
            spikes,
            E_L,
        ]

        + neurons[
            spikes,
            F_V,
        ]
        * (
            v_pre[spikes]
            - neurons[
                spikes,
                E_L,
            ]
        )

        - neurons[
            spikes,
            DELTA_V,
        ]
    )

    theta_s_next[spikes] += (
        neurons[
            spikes,
            DELTA_THETA_S,
        ]
    )

    _glif_apply_asc_spike_reset(
        neurons,
        spikes,
        asc1_next,
        asc2_next,
    )

    neurons[
        spikes,
        REFRACTORY,
    ] = neurons[
        spikes,
        T_REF,
    ]

    neurons[:, V] = v_next
    neurons[:, THETA_S] = theta_s_next
    neurons[:, ASC_1] = asc1_next
    neurons[:, ASC_2] = asc2_next
    neurons[:, THETA_V] = theta_v_next

    return _spike_vector(spikes)


# =============================================================================
# CADEX
# =============================================================================

def cadex_step(
    neurons,
    input_current,
    dt,
):
    """
    Vectorized canonical Conductance-Based Adaptive Exponential
    Integrate-and-Fire neuron.

    ADAPT stores adaptation conductance g_A.


    EQUATIONS:

        C_m dV/dt =

            g_L*(E_L-V)

            + g_L*DELTA_T
              * exp(
                    (V-V_T)
                    / DELTA_T
                )

            + g_A*(E_A-V)

            + I


        TAU_G_A dg_A/dt =

            G_A_BAR
            /
            (
                1
                + exp(
                    (V_A-V)
                    / DELTA_A
                )
            )

            - g_A


    SPIKE:

        if V >= V_DETECT:

            V <- V_RESET

            g_A <-
                g_A
                + DELTA_G_A

            refractory <- T_REF


    PSEUDOCODE:

        current <- broadcast input

        active <- refractory <= 0

        update refractory timer

        g_target <- (
            G_A_BAR
            /
            (
                1
                + exp(
                    (V_A-V)
                    / DELTA_A
                )
            )
        )

        dg_A <- (
            g_target
            - g_A
        ) / TAU_G_A

        exponential <- (
            g_L
            * DELTA_T
            * exp(
                (V-V_T)
                / DELTA_T
            )
        )

        dV <- (
            leak
            + exponential
            + adaptation_conductance_current
            + external_current
        ) / C_m

        g_A <- g_A + dt*dg_A

        V_candidate <- V + dt*dV

        spike <- (
            active
            AND
            V_candidate >= V_DETECT
        )

        if spike:
            V <- V_RESET
            g_A <- g_A + DELTA_G_A
            refractory <- T_REF

        return spike
    """

    _check_population(neurons)

    dt = DTYPE(dt)

    current = _input_vector(
        input_current,
        neurons.shape[0],
    )

    active = _refractory_active(
        neurons,
        dt,
    )

    v = neurons[:, V]
    g_a = neurons[:, ADAPT]

    g_l = neurons[:, G_L]
    e_l = neurons[:, E_L]

    delta_t = neurons[:, DELTA_T]

    g_target = (
        neurons[:, G_A_BAR]
        /
        (
            DTYPE(1.0)

            + np.exp(
                (
                    neurons[:, V_A]
                    - v
                )
                / neurons[:, DELTA_A]
            )
        )
    )

    dg_a = (
        g_target
        - g_a
    ) / neurons[:, TAU_G_A]

    exponential = (
        g_l
        * delta_t
        * np.exp(
            (
                v
                - neurons[:, V_T]
            )
            / delta_t
        )
    )

    dv = (
        g_l
        * (
            e_l
            - v
        )

        + exponential

        + g_a
        * (
            neurons[:, E_A]
            - v
        )

        + current

    ) / neurons[:, C_M]

    v_candidate = (
        v
        + dt * dv
    )

    g_a_next = (
        g_a
        + dt * dg_a
    )

    v_next = np.where(
        active,
        v_candidate,
        neurons[:, V_RESET],
    )

    spikes = (
        active
        & (
            v_candidate
            >= neurons[:, V_DETECT]
        )
    )

    v_next[spikes] = (
        neurons[
            spikes,
            V_RESET,
        ]
    )

    g_a_next[spikes] += (
        neurons[
            spikes,
            DELTA_G_A,
        ]
    )

    neurons[
        spikes,
        REFRACTORY,
    ] = neurons[
        spikes,
        T_REF,
    ]

    neurons[:, V] = v_next
    neurons[:, ADAPT] = g_a_next

    return _spike_vector(spikes)


# =============================================================================
# CADEX + GLIF5 HYBRID
# =============================================================================

def cadex_glif_step(
    neurons,
    input_current,
    dt,
):
    """
    Experimental full CAdEx + GLIF5 hybrid.


    IMPORTANT:

        "CAdEx-GLIF" is not a standardized published neuron model.

        This implementation deliberately combines:

            full CAdEx membrane dynamics

            full CAdEx conductance adaptation

            GLIF dual after-spike currents

            GLIF spike-dependent threshold

            GLIF voltage-dependent threshold

            GLIF biological reset terms


    ADAPT stores CAdEx adaptation conductance g_A.


    --------------------------------------------------------------------------
    CADEX MEMBRANE
    --------------------------------------------------------------------------

        C_m dV/dt =

            g_L*(E_L-V)

            + g_L*DELTA_T
              * exp(
                    (V-V_T)
                    / DELTA_T
                )

            + g_A*(E_A-V)

            + ASC_1

            + ASC_2

            + I


    --------------------------------------------------------------------------
    CADEX CONDUCTANCE ADAPTATION
    --------------------------------------------------------------------------

        TAU_G_A dg_A/dt =

            G_A_BAR
            /
            (
                1
                + exp(
                    (V_A-V)
                    / DELTA_A
                )
            )

            - g_A


    --------------------------------------------------------------------------
    GLIF5 AUXILIARY DYNAMICS
    --------------------------------------------------------------------------

        dTHETA_S/dt =
            -B_S*THETA_S


        dASC_1/dt =
            -K_1*ASC_1


        dASC_2/dt =
            -K_2*ASC_2


        dTHETA_V/dt =

            A_V*(V-E_L)

            - B_V*THETA_V


    --------------------------------------------------------------------------
    GLIF THRESHOLD
    --------------------------------------------------------------------------

        theta_GLIF =

            THETA_INF

            + THETA_S

            + THETA_V


    --------------------------------------------------------------------------
    CADEX DETECTION LIMIT
    --------------------------------------------------------------------------

        theta_CAdEx =
            V_DETECT


    --------------------------------------------------------------------------
    HYBRID SPIKE DETECTION
    --------------------------------------------------------------------------

        theta_detect =

            max(
                V_DETECT,
                theta_GLIF
            )


        spike if:

            V >= theta_detect


    Setting:

        V_DETECT == THETA_INF

    produces a unified baseline threshold while retaining the dynamic
    THETA_S and THETA_V terms.


    --------------------------------------------------------------------------
    HYBRID RESET
    --------------------------------------------------------------------------

    CAdEx uses a hard V_RESET.

    GLIF uses:

        E_L
        + F_V*(V_pre-E_L)
        - DELTA_V

    To prevent either reset mechanism from simply deleting the other,
    this hybrid uses:

        V_post =

            V_RESET

            + F_V
              * (
                    V_pre
                    - E_L
                )

            - DELTA_V


    Remaining spike updates:

        g_A <-
            g_A
            + DELTA_G_A


        THETA_S <-
            THETA_S
            + DELTA_THETA_S


        ASC_1 <-
            F_1*ASC_1
            + DELTA_I_1


        ASC_2 <-
            F_2*ASC_2
            + DELTA_I_2


        THETA_V <-
            THETA_V


        refractory <-
            T_REF


    --------------------------------------------------------------------------
    PSEUDOCODE
    --------------------------------------------------------------------------

        current <- broadcast input

        active <- refractory <= 0

        update refractory timer


        # CAdEx adaptation

        g_target <- sigmoid_adaptation(V)

        dg_A <- (
            g_target-g_A
        ) / tau_g_A

        g_A_next <- (
            g_A
            + dt*dg_A
        )


        # GLIF states

        THETA_S_next <- integrate THETA_S

        ASC_1_next <- integrate ASC_1

        ASC_2_next <- integrate ASC_2

        THETA_V_next <- integrate THETA_V


        # CAdEx exponential initiation

        exponential <- (
            g_L
            * DELTA_T
            * exp(
                (V-V_T)
                / DELTA_T
            )
        )


        # Hybrid membrane

        dV <- (
            leak
            + exponential
            + conductance adaptation
            + ASC_1
            + ASC_2
            + input
        ) / C_m

        V_candidate <- V + dt*dV


        # Threshold

        theta_GLIF <- (
            THETA_INF
            + THETA_S_next
            + THETA_V_next
        )

        detection <- max(
            V_DETECT,
            theta_GLIF
        )

        spike <- (
            active
            AND
            V_candidate >= detection
        )


        # Reset

        if spike:

            V <- combined CAdEx/GLIF reset

            g_A <- g_A + DELTA_G_A

            THETA_S <-
                THETA_S
                + DELTA_THETA_S

            update ASC_1

            update ASC_2

            keep THETA_V continuous

            refractory <- T_REF


        write all states

        return spike
    """

    _check_population(neurons)

    dt = DTYPE(dt)

    current = _input_vector(
        input_current,
        neurons.shape[0],
    )

    active = _refractory_active(
        neurons,
        dt,
    )

    v_pre = neurons[:, V].copy()

    v = neurons[:, V]
    g_a = neurons[:, ADAPT]

    g_l = neurons[:, G_L]
    e_l = neurons[:, E_L]

    delta_t = neurons[:, DELTA_T]


    # -------------------------------------------------------------------------
    # CAdEx conductance adaptation
    # -------------------------------------------------------------------------

    g_target = (
        neurons[:, G_A_BAR]
        /
        (
            DTYPE(1.0)

            + np.exp(
                (
                    neurons[:, V_A]
                    - v
                )
                / neurons[:, DELTA_A]
            )
        )
    )

    g_a_next = (
        g_a

        + dt
        * (
            (
                g_target
                - g_a
            )
            / neurons[:, TAU_G_A]
        )
    )


    # -------------------------------------------------------------------------
    # GLIF spike-dependent threshold
    # -------------------------------------------------------------------------

    theta_s_next = (
        neurons[:, THETA_S]

        + dt
        * (
            -neurons[:, B_S]
            * neurons[:, THETA_S]
        )
    )


    # -------------------------------------------------------------------------
    # GLIF after-spike currents
    # -------------------------------------------------------------------------

    (
        asc1_next,
        asc2_next,
    ) = _glif_asc_euler(
        neurons,
        dt,
    )


    # -------------------------------------------------------------------------
    # GLIF voltage-dependent threshold
    # -------------------------------------------------------------------------

    theta_v_next = (
        neurons[:, THETA_V]

        + dt
        * (
            neurons[:, A_V]
            * (
                v
                - e_l
            )

            - neurons[:, B_V]
            * neurons[:, THETA_V]
        )
    )


    # -------------------------------------------------------------------------
    # CAdEx exponential spike initiation
    # -------------------------------------------------------------------------

    exponential = (
        g_l
        * delta_t

        * np.exp(
            (
                v
                - neurons[:, V_T]
            )
            / delta_t
        )
    )


    # -------------------------------------------------------------------------
    # Hybrid membrane equation
    # -------------------------------------------------------------------------

    dv = (
        g_l
        * (
            e_l
            - v
        )

        + exponential

        + g_a
        * (
            neurons[:, E_A]
            - v
        )

        + neurons[:, ASC_1]

        + neurons[:, ASC_2]

        + current

    ) / neurons[:, C_M]


    v_candidate = (
        v
        + dt * dv
    )


    # -------------------------------------------------------------------------
    # Combined detection threshold
    # -------------------------------------------------------------------------

    theta_glif = (
        neurons[:, THETA_INF]
        + theta_s_next
        + theta_v_next
    )

    detection_threshold = np.maximum(
        neurons[:, V_DETECT],
        theta_glif,
    )


    # During refractory state retain the post-spike voltage rather than
    # overwriting it with a second reset rule.
    v_next = np.where(
        active,
        v_candidate,
        v,
    )


    spikes = (
        active
        & (
            v_candidate
            >= detection_threshold
        )
    )


    # -------------------------------------------------------------------------
    # Combined CAdEx + GLIF voltage reset
    # -------------------------------------------------------------------------

    v_next[spikes] = (
        neurons[
            spikes,
            V_RESET,
        ]

        + neurons[
            spikes,
            F_V,
        ]
        * (
            v_pre[spikes]
            - neurons[
                spikes,
                E_L,
            ]
        )

        - neurons[
            spikes,
            DELTA_V,
        ]
    )


    # -------------------------------------------------------------------------
    # CAdEx spike-triggered conductance increment
    # -------------------------------------------------------------------------

    g_a_next[spikes] += (
        neurons[
            spikes,
            DELTA_G_A,
        ]
    )


    # -------------------------------------------------------------------------
    # GLIF spike-triggered threshold increment
    # -------------------------------------------------------------------------

    theta_s_next[spikes] += (
        neurons[
            spikes,
            DELTA_THETA_S,
        ]
    )


    # -------------------------------------------------------------------------
    # GLIF ASC reset
    # -------------------------------------------------------------------------

    _glif_apply_asc_spike_reset(
        neurons,
        spikes,
        asc1_next,
        asc2_next,
    )


    neurons[
        spikes,
        REFRACTORY,
    ] = neurons[
        spikes,
        T_REF,
    ]


    # -------------------------------------------------------------------------
    # Write mutable states
    # -------------------------------------------------------------------------

    neurons[:, V] = v_next

    neurons[:, ADAPT] = g_a_next

    neurons[:, THETA_S] = theta_s_next

    neurons[:, ASC_1] = asc1_next

    neurons[:, ASC_2] = asc2_next

    neurons[:, THETA_V] = theta_v_next


    return _spike_vector(spikes)


# Alias matching the spelling used in the initial design discussion.
cadec_glif_step = cadex_glif_step


# =============================================================================
# OPTIONAL MODEL DISPATCHER
# =============================================================================

MODEL_STEPS = {
    "lif":
        lif_step,

    "adaptive_lif":
        adaptive_lif_step,

    "alif":
        adaptive_lif_step,

    "izhikevich":
        izhikevich_step,

    "adex":
        adex_step,

    "glif3":
        glif3_step,

    "glif4":
        glif4_step,

    "glif5":
        glif5_step,

    "cadex":
        cadex_step,

    "cadex_glif":
        cadex_glif_step,

    "cadec_glif":
        cadex_glif_step,
}


def step(
    model,
    neurons,
    input_current,
    dt,
):
    """
    Generic dispatcher.

    PSEUDOCODE:

        normalize model name

        lookup independent model function

        call:

            model_step(
                neurons,
                input_current,
                dt
            )

        return spike events


    EXAMPLE:

        spikes = step(
            "glif5",
            neurons,
            current,
            dt=0.1,
        )
    """

    key = str(model).lower()

    try:
        function = MODEL_STEPS[key]

    except KeyError as exc:

        raise ValueError(
            f"Unknown model: {model!r}. "
            f"Available models: "
            f"{tuple(MODEL_STEPS.keys())}"
        ) from exc

    return function(
        neurons,
        input_current,
        dt,
    )