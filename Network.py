# Network.py

import numpy as np

from collections import OrderedDict
from collections.abc import Mapping

import NeuronModels as nm


# =============================================================================
# NEURON TYPE NORMALIZATION
# =============================================================================

NEURON_TYPE_ALIASES = {
    "lif":
        "lif",

    "adaptivelif":
        "adaptive_lif",

    "adaptive_lif":
        "adaptive_lif",

    "alif":
        "adaptive_lif",

    "izhikevich":
        "izhikevich",

    "adex":
        "adex",

    "glif3":
        "glif3",

    "glif4":
        "glif4",

    "glif5":
        "glif5",

    "cadex":
        "cadex",

    "cadex_glif":
        "cadex_glif",

    "cadec_glif":
        "cadex_glif",
}


def _normalize_neuron_type(neuron_type: str) -> str:
    """
    Normalize a neuron type name into its canonical internal identifier.

    PSEUDOCODE:

        convert input to string

        lowercase

        remove spaces and hyphens

        resolve aliases

        return canonical neuron type
    """

    if not isinstance(neuron_type, str):
        raise TypeError(
            "neuron type must be a string"
        )

    key = (
        neuron_type
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "")
    )

    try:
        return NEURON_TYPE_ALIASES[key]

    except KeyError as exc:
        raise ValueError(
            f"Unsupported neuron type: {neuron_type!r}. "
            f"Supported types are: "
            f"{sorted(set(NEURON_TYPE_ALIASES.values()))}"
        ) from exc


# =============================================================================
# POPULATION SPECIFICATION
# =============================================================================

def _parse_population_spec(
    population: int,
    neuron_type,
):
    """
    Convert homogeneous or mixed neuron specification into contiguous counts.

    INPUT FORMS
    -----------

    Homogeneous:

        neuron_type="GLIF5"

    means:

        GLIF5 -> entire population


    Mixed:

        neuron_type={
            "default": "LIF",
            "GLIF5": 2,
            "AdEx": 3,
        }

    with population=8 means:

        GLIF5 -> 2
        AdEx  -> 3
        LIF   -> 3


    PSEUDOCODE:

        if neuron_type is string:

            normalize neuron type

            assign entire population to that type


        if neuron_type is mapping:

            require "default"

            normalize default type

            read explicit population counts

            verify:
                count >= 0
                total explicit <= population

            remaining =
                population - explicit total

            add remaining neurons to default type

            combine duplicate aliases/types

            return:
                default type
                ordered neuron type counts
    """

    if isinstance(neuron_type, str):

        canonical = _normalize_neuron_type(
            neuron_type
        )

        return (
            canonical,
            OrderedDict(
                [
                    (
                        canonical,
                        population,
                    )
                ]
            ),
        )


    if not isinstance(neuron_type, Mapping):
        raise TypeError(
            "neuron_type must be either a string "
            "or a mapping"
        )


    if "default" not in neuron_type:
        raise ValueError(
            "Mixed neuron populations must contain "
            "a 'default' neuron type"
        )


    default_type = _normalize_neuron_type(
        neuron_type["default"]
    )


    counts = OrderedDict()

    explicitly_assigned = 0


    for model_name, count in neuron_type.items():

        if model_name == "default":
            continue


        canonical = _normalize_neuron_type(
            model_name
        )


        if not isinstance(
            count,
            (int, np.integer),
        ):
            raise TypeError(
                f"Population count for "
                f"{model_name!r} must be an integer"
            )


        count = int(count)


        if count < 0:
            raise ValueError(
                f"Population count for "
                f"{model_name!r} cannot be negative"
            )


        explicitly_assigned += count


        counts[canonical] = (
            counts.get(
                canonical,
                0,
            )
            + count
        )


    if explicitly_assigned > population:
        raise ValueError(
            "Specified mixed neuron populations exceed "
            f"the total population size: "
            f"{explicitly_assigned} > {population}"
        )


    remaining = (
        population
        - explicitly_assigned
    )


    if remaining > 0:

        counts[default_type] = (
            counts.get(
                default_type,
                0,
            )
            + remaining
        )


    # Remove zero-size groups.
    counts = OrderedDict(
        (
            model_name,
            count,
        )
        for model_name, count
        in counts.items()
        if count > 0
    )


    return (
        default_type,
        counts,
    )


# =============================================================================
# SNN NETWORK
# =============================================================================

class SNN:
    """
    LiSNN network container.

    Current responsibility:

        population construction
        neuron-type organization
        parameter initialization
        model-specific initial state

    Not implemented yet:

        synaptic weights
        connectivity
        propagation
        plasticity
        learning
        network stepping
    """

    def __init__(
        self,
        population: int = 8,
        neuron_type="LIF",
        fill: np.float32 = np.float32(0.0),
        randomize_params: bool = False,
        seed=None,
    ):
        """
        Construct an SNN population.


        PSEUDOCODE:

            validate population

            parse neuron specification

            create shared:
                [population, NEURON_WIDTH]
                float32 matrix

            construct contiguous neuron-type slices

            construct per-neuron type labels

            initialize any model-specific dynamic states

            retain network metadata
        """

        # ---------------------------------------------------------------------
        # Validate population
        # ---------------------------------------------------------------------

        if not isinstance(
            population,
            (int, np.integer),
        ):
            raise TypeError(
                "population must be an integer"
            )


        population = int(population)


        if population <= 0:
            raise ValueError(
                "population must be greater than zero"
            )


        self.population_size = population

        self.randomize_params = bool(
            randomize_params
        )

        self.seed = seed


        # ---------------------------------------------------------------------
        # Parse neuron population specification
        # ---------------------------------------------------------------------

        (
            self.default_neuron_type,
            self.type_counts,
        ) = _parse_population_spec(
            population,
            neuron_type,
        )


        self.homogeneous = (
            len(self.type_counts) == 1
        )


        # ---------------------------------------------------------------------
        # Create common neuron matrix
        # ---------------------------------------------------------------------

        self.pool = nm.new_population(
            population,
            fill=fill, # pyright: ignore[reportArgumentType]
            randomize_params=randomize_params,
            seed=seed,
        )


        # Convenience alias.
        #
        # Both:
        #
        #     model.pool
        #
        # and:
        #
        #     model.neurons
        #
        # reference the same ndarray.

        self.neurons = self.pool


        # ---------------------------------------------------------------------
        # Build contiguous type slices
        # ---------------------------------------------------------------------
        #
        # Example:
        #
        #     {
        #         "glif5": slice(0, 2),
        #         "adex":  slice(2, 5),
        #         "lif":   slice(5, 8),
        #     }
        #
        # This becomes important later because:
        #
        #     nm.glif5_step(pool[slice], ...)
        #
        # remains vectorized.
        # ---------------------------------------------------------------------

        self.type_slices = OrderedDict()


        # Per-neuron canonical type label.
        self.neuron_types = np.empty(
            population,
            dtype=object,
        )


        cursor = 0


        for model_name, count in self.type_counts.items():

            start = cursor
            stop = cursor + count

            model_slice = slice(
                start,
                stop,
            )


            self.type_slices[
                model_name
            ] = model_slice


            self.neuron_types[
                model_slice
            ] = model_name


            cursor = stop


        if cursor != population:
            raise RuntimeError(
                "Internal population construction error: "
                f"assigned {cursor} neurons "
                f"for population {population}"
            )


        # ---------------------------------------------------------------------
        # Model-specific dynamic-state initialization
        # ---------------------------------------------------------------------

        self._initialize_model_states()


    # =========================================================================
    # MODEL-SPECIFIC INITIAL STATE
    # =========================================================================

    def _initialize_model_states(self):
        """
        Initialize dynamic states whose neutral value is model-specific.

        Most models correctly initialize with:

            ADAPT = 0
            THETA_S = 0
            ASC_1 = 0
            ASC_2 = 0
            THETA_V = 0
            REFRACTORY = 0

        Izhikevich is different.

        Its recovery variable should begin at:

            u = b * V

        and ADAPT stores u for Izhikevich neurons.


        PSEUDOCODE:

            if population contains Izhikevich neurons:

                locate contiguous Izhikevich slice

                ADAPT =
                    IZH_B * V
        """

        if "izhikevich" in self.type_slices:

            s = self.type_slices[
                "izhikevich"
            ]

            self.pool[
                s,
                nm.ADAPT,
            ] = (
                self.pool[
                    s,
                    nm.IZH_B,
                ]
                * self.pool[
                    s,
                    nm.V,
                ]
            ).astype(
                np.float32
            )


    # =========================================================================
    # REPRESENTATION
    # =========================================================================

    def __repr__(self):

        population_description = ", ".join(
            f"{model_name}={count}"
            for model_name, count
            in self.type_counts.items()
        )

        return (
            f"SNN("
            f"population={self.population_size}, "
            f"neurons=[{population_description}], "
            f"randomize_params={self.randomize_params}"
            f")"
        )


# =============================================================================
# NETWORK FACTORY
# =============================================================================

def create_nn(
    population: int = 8,
    neuron_type="LIF",
    fill: np.float32 = np.float32(0.0),
    randomize_params: bool = False,
    seed=None,
):
    """
    Create and return an initialized SNN instance.

    No synaptic weight matrix or connectivity is created yet.


    HOMOGENEOUS EXAMPLE:

        model = Network.create_nn(
            population=8,
            neuron_type="GLIF5",
            randomize_params=False,
        )


    MIXED EXAMPLE:

        model = Network.create_nn(
            population=8,

            neuron_type={
                "default": "LIF",
                "GLIF5": 2,
                "AdEx": 3,
            },

            randomize_params=True,

            seed=1,
        )


    PSEUDOCODE:

        construct SNN

        return SNN instance
    """

    return SNN(
        population=population,
        neuron_type=neuron_type,
        fill=fill,
        randomize_params=randomize_params,
        seed=seed,
    )


# Optional naming alias because both names have come up in the API design.
create_snn = create_nn