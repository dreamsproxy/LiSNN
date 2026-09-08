import numpy as np
from pathlib import Path
from datetime import datetime

import NeuronModels as nm


# =============================================================================
# DEBUG DEFAULTS
# =============================================================================

SMOKE_DT = np.float32(0.1)
SMOKE_INPUT_CURRENT = np.float32(100.0)

FORCED_SPIKE_MARGIN = np.float32(1.0)

DEFAULT_SEED = 1


MODEL_TESTS = {
    "LIF":
        nm.lif_step,

    "AdaptiveLIF":
        nm.adaptive_lif_step,

    "Izhikevich":
        nm.izhikevich_step,

    "AdEx":
        nm.adex_step,

    "GLIF3":
        nm.glif3_step,

    "GLIF4":
        nm.glif4_step,

    "GLIF5":
        nm.glif5_step,

    "CAdEx":
        nm.cadex_step,

    "CAdEx-GLIF":
        nm.cadex_glif_step,
}


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _resolve_log_path(path_template, funcname):
    """
    Resolve timestamp tokens in a debug log path.

    Supported tokens:

        {YYYY}
        {MM}
        {DD}
        {HH}
        {MMIN}
        {SS}
        {funcname}

    Compatibility behavior:

        Because the requested template uses {MM} for both month and minute:

            ./logs/debug_{YYYY}_{MM}_{DD}-{HH}_{MM}_{SS}-{funcname}.log

        the first {MM} is treated as month and the second as minute.
    """

    now = datetime.now()

    # Handle the exact requested format first.
    path = path_template

    path = path.replace(
        "{YYYY}",
        now.strftime("%Y"),
    )

    path = path.replace(
        "{DD}",
        now.strftime("%d"),
    )

    path = path.replace(
        "{HH}",
        now.strftime("%H"),
    )

    path = path.replace(
        "{SS}",
        now.strftime("%S"),
    )

    path = path.replace(
        "{MMIN}",
        now.strftime("%M"),
    )

    path = path.replace(
        "{funcname}",
        funcname,
    )

    # The requested path uses {MM} twice:
    #
    #     YYYY_MM_DD-HH_MM_SS
    #
    # First occurrence = month.
    # Remaining occurrence = minute.

    first_mm = path.find("{MM}")

    if first_mm >= 0:

        path = (
            path[:first_mm]
            + now.strftime("%m")
            + path[first_mm + 4:]
        )

    path = path.replace(
        "{MM}",
        now.strftime("%M"),
    )

    return Path(path)


def _prepare_population(
    neuron_count,
    model_name,
    seed,
):
    """
    Create a fresh deterministic randomized population.

    PSEUDOCODE:

        create population using fixed seed

        if Izhikevich:
            initialize recovery variable:
                u = b * V

        return population
    """

    pool = nm.new_population(
        neuron_count,
        randomize_params=True,
        seed=seed,
    )

    if model_name == "Izhikevich":

        pool[:, nm.ADAPT] = (
            pool[:, nm.IZH_B]
            * pool[:, nm.V]
        ).astype(np.float32)

    return pool


def _forced_threshold(
    pool,
    model_name,
):
    """
    Return the effective spike threshold for every neuron.

    PSEUDOCODE:

        LIF:
            THETA_INF

        AdaptiveLIF:
            THETA_INF + ADAPT

        Izhikevich:
            V_DETECT

        AdEx:
            V_DETECT

        GLIF3:
            THETA_INF

        GLIF4:
            THETA_INF + THETA_S

        GLIF5:
            THETA_INF
            + THETA_S
            + THETA_V

        CAdEx:
            V_DETECT

        CAdEx-GLIF:
            max(
                V_DETECT,
                GLIF5 threshold
            )
    """

    if model_name == "LIF":

        return pool[:, nm.THETA_INF].copy()

    if model_name == "AdaptiveLIF":

        return (
            pool[:, nm.THETA_INF]
            + pool[:, nm.ADAPT]
        )

    if model_name in (
        "Izhikevich",
        "AdEx",
        "CAdEx",
    ):

        return pool[:, nm.V_DETECT].copy()

    if model_name == "GLIF3":

        return pool[:, nm.THETA_INF].copy()

    if model_name == "GLIF4":

        return (
            pool[:, nm.THETA_INF]
            + pool[:, nm.THETA_S]
        )

    if model_name == "GLIF5":

        return (
            pool[:, nm.THETA_INF]
            + pool[:, nm.THETA_S]
            + pool[:, nm.THETA_V]
        )

    if model_name == "CAdEx-GLIF":

        glif_threshold = (
            pool[:, nm.THETA_INF]
            + pool[:, nm.THETA_S]
            + pool[:, nm.THETA_V]
        )

        return np.maximum(
            pool[:, nm.V_DETECT],
            glif_threshold,
        )

    raise ValueError(
        f"Unsupported model: {model_name}"
    )


def _state_snapshot(pool):
    """
    Extract mutable neuron state for readable debugging.
    """

    return {
        "V":
            pool[:, nm.V].copy(),

        "ADAPT":
            pool[:, nm.ADAPT].copy(),

        "THETA_S":
            pool[:, nm.THETA_S].copy(),

        "ASC_1":
            pool[:, nm.ASC_1].copy(),

        "ASC_2":
            pool[:, nm.ASC_2].copy(),

        "THETA_V":
            pool[:, nm.THETA_V].copy(),

        "REFRACTORY":
            pool[:, nm.REFRACTORY].copy(),
    }


# =============================================================================
# POPULATION SMOKE TEST
# =============================================================================

def population_smoke_test(
    neuron_count: int = 8,
    n_steps: int = 8,
    watch_spikes: bool = True,
    ouput_path: str = (
        "./logs/"
        "debug_{YYYY}_{MM}_{DD}-{HH}_{MM}_{SS}-{funcname}.log"
    ),
    verbose: bool = True,
):
    """
    Run population-level integration and spike/reset smoke tests.

    NOTE:
        `ouput_path` intentionally preserves the requested argument spelling.


    --------------------------------------------------------------------------
    PHASE A - ALTERNATING INPUT SMOKE TEST
    --------------------------------------------------------------------------

    PSEUDOCODE:

        construct:

            T0 =
                [signal, none, signal, none, ...]

            T1 =
                [none, signal, none, signal, ...]

        repeat alternating T0/T1 for n_steps

        for every neuron model:

            create fresh randomized population
            using identical seed

            initialize model-specific state

            for step in n_steps:

                select T0 or T1

                save voltage before step

                execute model step

                save voltage after step

                verify:

                    population shape

                    float32 dtype

                    spike shape

                    spike dtype

                    finite state

                    finite spikes

                record:

                    input
                    voltage before
                    voltage after
                    voltage delta
                    spikes

                if watch_spikes and natural spike occurs:
                    inspect mutable post-spike state


    --------------------------------------------------------------------------
    PHASE B - FORCED SPIKE / RESET VERIFICATION
    --------------------------------------------------------------------------

    PSEUDOCODE:

        if watch_spikes:

            for every neuron model:

                create ANOTHER fresh identical population

                determine effective spike threshold

                set:

                    V =
                        threshold
                        + small margin

                save all mutable state

                execute actual neuron step
                with zero external input

                inspect:

                    spike event

                    voltage reset

                    refractory state

                    adaptation state

                    GLIF thresholds

                    after-spike currents

                verify model-specific expected changes


    RETURNS:

        dict containing:

            log_path

            pass/fail summary

            model test data
    """

    if neuron_count <= 0:
        raise ValueError(
            "neuron_count must be > 0"
        )

    if n_steps <= 0:
        raise ValueError(
            "n_steps must be > 0"
        )

    funcname = "population_smoke_test"

    log_path = _resolve_log_path(
        ouput_path,
        funcname,
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_lines = []

    def log(*items):
        text = " ".join(
            str(item)
            for item in items
        )

        log_lines.append(text)

        if verbose:
            print(text)


    # =========================================================================
    # INPUT PATTERN
    # =========================================================================

    neuron_indices = np.arange(
        neuron_count,
    )

    even_mask = (
        neuron_indices % 2
        == 0
    )

    odd_mask = ~even_mask


    T0 = np.where(
        even_mask,
        SMOKE_INPUT_CURRENT,
        np.float32(0.0),
    ).astype(np.float32)


    T1 = np.where(
        odd_mask,
        SMOKE_INPUT_CURRENT,
        np.float32(0.0),
    ).astype(np.float32)


    log("=" * 90)
    log("LiSNN POPULATION SMOKE TEST")
    log("=" * 90)

    log(
        "Neuron count:",
        neuron_count,
    )

    log(
        "Steps:",
        n_steps,
    )

    log(
        "Seed:",
        DEFAULT_SEED,
    )

    log(
        "dt:",
        SMOKE_DT,
        "ms",
    )

    log(
        "Input current:",
        SMOKE_INPUT_CURRENT,
    )

    log("")

    log(
        "T0:",
        T0,
    )

    log(
        "T1:",
        T1,
    )

    log("")


    results = {
        "log_path":
            str(log_path),

        "models":
            {},

        "passed":
            True,
    }


    # =========================================================================
    # MODEL LOOP
    # =========================================================================

    for model_name, step_fn in MODEL_TESTS.items():

        model_passed = True
        failures = []

        log("=" * 90)
        log(model_name)
        log("=" * 90)


        # ---------------------------------------------------------------------
        # Fresh population
        # ---------------------------------------------------------------------

        pool = _prepare_population(
            neuron_count,
            model_name,
            DEFAULT_SEED,
        )

        expected_shape = (
            neuron_count,
            nm.NEURON_WIDTH,
        )


        model_results = {
            "steps":
                [],

            "forced_spike":
                None,

            "passed":
                True,

            "failures":
                failures,
        }


        # =====================================================================
        # PHASE A
        # =====================================================================

        log("PHASE A: alternating-input integration")
        log("-" * 90)


        for step_index in range(n_steps):

            input_vector = (
                T0
                if step_index % 2 == 0
                else T1
            )

            active_mask = (
                even_mask
                if step_index % 2 == 0
                else odd_mask
            )


            v_before = (
                pool[:, nm.V].copy()
            )


            spikes = step_fn(
                pool,
                input_vector,
                SMOKE_DT,
            )


            v_after = (
                pool[:, nm.V].copy()
            )


            delta_v = (
                v_after
                - v_before
            )


            # -----------------------------------------------------------------
            # Core verification
            # -----------------------------------------------------------------

            checks = {
                "population_shape":
                    pool.shape
                    == expected_shape,

                "population_dtype":
                    pool.dtype
                    == np.float32,

                "spike_shape":
                    spikes.shape
                    == (neuron_count,),

                "spike_dtype":
                    spikes.dtype
                    == np.float32,

                "finite_population":
                    bool(
                        np.all(
                            np.isfinite(pool)
                        )
                    ),

                "finite_spikes":
                    bool(
                        np.all(
                            np.isfinite(spikes)
                        )
                    ),

                "binary_spikes":
                    bool(
                        np.all(
                            (
                                spikes == 0.0
                            )
                            |
                            (
                                spikes == 1.0
                            )
                        )
                    ),
            }


            failed_checks = [
                name
                for name, passed
                in checks.items()
                if not passed
            ]


            if failed_checks:

                model_passed = False

                failures.append(
                    {
                        "phase":
                            "integration",

                        "step":
                            step_index,

                        "checks":
                            failed_checks,
                    }
                )


            # -----------------------------------------------------------------
            # Diagnostics
            # -----------------------------------------------------------------

            log(
                f"Step {step_index:03d}"
            )

            log(
                "Input:     ",
                np.round(
                    input_vector,
                    4,
                ),
            )

            log(
                "V before:  ",
                np.round(
                    v_before,
                    4,
                ),
            )

            log(
                "V after:   ",
                np.round(
                    v_after,
                    4,
                ),
            )

            log(
                "Delta V:   ",
                np.round(
                    delta_v,
                    4,
                ),
            )

            log(
                "Spikes:    ",
                spikes,
            )


            if np.any(active_mask):

                active_delta = np.mean(
                    delta_v[active_mask]
                )

                inactive_delta = np.mean(
                    delta_v[~active_mask]
                ) if np.any(~active_mask) else np.nan

                log(
                    "Mean dV stimulated:",
                    np.round(
                        active_delta,
                        6,
                    ),
                )

                log(
                    "Mean dV unstimulated:",
                    np.round(
                        inactive_delta,
                        6,
                    ),
                )


            # -----------------------------------------------------------------
            # Natural post-spike inspection
            # -----------------------------------------------------------------

            if (
                watch_spikes
                and np.any(spikes > 0.0)
            ):

                spike_indices = np.flatnonzero(
                    spikes > 0.0
                )

                snapshot = _state_snapshot(
                    pool
                )

                log(
                    "Natural spike indices:",
                    spike_indices,
                )

                for state_name, values in snapshot.items():

                    log(
                        f"Post-spike {state_name}:",
                        np.round(
                            values,
                            6,
                        ),
                    )


            log(
                "Checks:",
                (
                    "PASS"
                    if not failed_checks
                    else
                    f"FAIL {failed_checks}"
                ),
            )

            log("")


            model_results[
                "steps"
            ].append(
                {
                    "step":
                        step_index,

                    "input":
                        input_vector.copy(),

                    "V_before":
                        v_before,

                    "V_after":
                        v_after,

                    "delta_V":
                        delta_v,

                    "spikes":
                        spikes.copy(),

                    "checks":
                        checks,
                }
            )


        # =====================================================================
        # PHASE B
        # =====================================================================

        if watch_spikes:

            log("")
            log(
                "PHASE B: forced spike / "
                "post-spike state verification"
            )

            log("-" * 90)


            spike_pool = _prepare_population(
                neuron_count,
                model_name,
                DEFAULT_SEED,
            )


            threshold = _forced_threshold(
                spike_pool,
                model_name,
            )


            # Save threshold-related state before forcing spike.
            pre_threshold = threshold.copy()


            # Force all neurons just above their own effective detection
            # threshold.
            spike_pool[:, nm.V] = (
                threshold
                + FORCED_SPIKE_MARGIN
            ).astype(np.float32)


            pre_state = _state_snapshot(
                spike_pool
            )


            # No external current is required.
            zero_input = np.zeros(
                neuron_count,
                dtype=np.float32,
            )


            forced_spikes = step_fn(
                spike_pool,
                zero_input,
                SMOKE_DT,
            )


            post_state = _state_snapshot(
                spike_pool
            )


            # -----------------------------------------------------------------
            # Generic spike verification
            # -----------------------------------------------------------------

            spike_checks = {
                "all_spiked":
                    bool(
                        np.all(
                            forced_spikes
                            == np.float32(1.0)
                        )
                    ),

                "finite_state":
                    bool(
                        np.all(
                            np.isfinite(
                                spike_pool
                            )
                        )
                    ),

                "finite_spikes":
                    bool(
                        np.all(
                            np.isfinite(
                                forced_spikes
                            )
                        )
                    ),

                "binary_spikes":
                    bool(
                        np.all(
                            (
                                forced_spikes
                                == 0.0
                            )
                            |
                            (
                                forced_spikes
                                == 1.0
                            )
                        )
                    ),
            }


            # -----------------------------------------------------------------
            # Model-specific reset verification
            # -----------------------------------------------------------------

            if model_name in (
                "LIF",
                "AdaptiveLIF",
                "AdEx",
                "CAdEx",
            ):

                expected_v = (
                    spike_pool[
                        :,
                        nm.V_RESET,
                    ]
                )

                spike_checks[
                    "voltage_reset"
                ] = bool(
                    np.allclose(
                        post_state["V"],
                        expected_v,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                )


            elif model_name == "Izhikevich":

                expected_v = (
                    spike_pool[
                        :,
                        nm.V_RESET,
                    ]
                )

                spike_checks[
                    "voltage_reset"
                ] = bool(
                    np.allclose(
                        post_state["V"],
                        expected_v,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                )


            elif model_name == "GLIF3":

                expected_v = (
                    spike_pool[
                        :,
                        nm.E_L,
                    ]
                )

                spike_checks[
                    "voltage_reset"
                ] = bool(
                    np.allclose(
                        post_state["V"],
                        expected_v,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                )


            elif model_name in (
                "GLIF4",
                "GLIF5",
            ):

                expected_v = (
                    spike_pool[
                        :,
                        nm.E_L,
                    ]

                    + spike_pool[
                        :,
                        nm.F_V,
                    ]
                    * (
                        pre_state["V"]
                        - spike_pool[
                            :,
                            nm.E_L,
                        ]
                    )

                    - spike_pool[
                        :,
                        nm.DELTA_V,
                    ]
                )

                spike_checks[
                    "voltage_reset"
                ] = bool(
                    np.allclose(
                        post_state["V"],
                        expected_v,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                )


            elif model_name == "CAdEx-GLIF":

                expected_v = (
                    spike_pool[
                        :,
                        nm.V_RESET,
                    ]

                    + spike_pool[
                        :,
                        nm.F_V,
                    ]
                    * (
                        pre_state["V"]
                        - spike_pool[
                            :,
                            nm.E_L,
                        ]
                    )

                    - spike_pool[
                        :,
                        nm.DELTA_V,
                    ]
                )

                spike_checks[
                    "voltage_reset"
                ] = bool(
                    np.allclose(
                        post_state["V"],
                        expected_v,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                )


            # -----------------------------------------------------------------
            # Refractory verification
            # -----------------------------------------------------------------

            if model_name != "Izhikevich":

                spike_checks[
                    "refractory_set"
                ] = bool(
                    np.all(
                        post_state[
                            "REFRACTORY"
                        ]
                        > 0.0
                    )
                )


            # -----------------------------------------------------------------
            # Adaptation verification
            # -----------------------------------------------------------------

            if model_name == "AdaptiveLIF":

                spike_checks[
                    "adaptive_threshold_increment"
                ] = bool(
                    np.all(
                        post_state["ADAPT"]
                        > pre_state["ADAPT"]
                    )
                )


            elif model_name == "Izhikevich":

                spike_checks[
                    "recovery_updated"
                ] = bool(
                    np.any(
                        post_state["ADAPT"]
                        != pre_state["ADAPT"]
                    )
                )


            elif model_name == "AdEx":

                spike_checks[
                    "adaptation_current_increment"
                ] = bool(
                    np.all(
                        post_state["ADAPT"]
                        > pre_state["ADAPT"]
                    )
                )


            elif model_name in (
                "CAdEx",
                "CAdEx-GLIF",
            ):

                spike_checks[
                    "adaptation_conductance_increment"
                ] = bool(
                    np.all(
                        post_state["ADAPT"]
                        > pre_state["ADAPT"]
                    )
                )


            # -----------------------------------------------------------------
            # GLIF ASC verification
            # -----------------------------------------------------------------

            if model_name in (
                "GLIF3",
                "GLIF4",
                "GLIF5",
                "CAdEx-GLIF",
            ):

                spike_checks[
                    "ASC_1_updated"
                ] = bool(
                    np.any(
                        post_state["ASC_1"]
                        != pre_state["ASC_1"]
                    )
                )

                spike_checks[
                    "ASC_2_updated"
                ] = bool(
                    np.any(
                        post_state["ASC_2"]
                        != pre_state["ASC_2"]
                    )
                )


            # -----------------------------------------------------------------
            # GLIF spike-threshold adaptation
            # -----------------------------------------------------------------

            if model_name in (
                "GLIF4",
                "GLIF5",
                "CAdEx-GLIF",
            ):

                spike_checks[
                    "THETA_S_updated"
                ] = bool(
                    np.all(
                        post_state["THETA_S"]
                        > pre_state["THETA_S"]
                    )
                )


            # -----------------------------------------------------------------
            # GLIF5 voltage-dependent threshold
            # -----------------------------------------------------------------

            if model_name in (
                "GLIF5",
                "CAdEx-GLIF",
            ):

                spike_checks[
                    "THETA_V_evolved"
                ] = bool(
                    np.any(
                        post_state["THETA_V"]
                        != pre_state["THETA_V"]
                    )
                )


            failed_spike_checks = [
                name
                for name, passed
                in spike_checks.items()
                if not passed
            ]


            if failed_spike_checks:

                model_passed = False

                failures.append(
                    {
                        "phase":
                            "forced_spike",

                        "checks":
                            failed_spike_checks,
                    }
                )


            # -----------------------------------------------------------------
            # Log full post-spike state
            # -----------------------------------------------------------------

            log(
                "Forced threshold:",
                np.round(
                    pre_threshold,
                    6,
                ),
            )

            log(
                "Forced initial V:",
                np.round(
                    pre_state["V"],
                    6,
                ),
            )

            log(
                "Spikes:",
                forced_spikes,
            )

            log("")

            log("POST-SPIKE STATE")

            for state_name in (
                "V",
                "ADAPT",
                "THETA_S",
                "ASC_1",
                "ASC_2",
                "THETA_V",
                "REFRACTORY",
            ):

                log(
                    f"{state_name}:",
                    np.round(
                        post_state[
                            state_name
                        ],
                        6,
                    ),
                )


            log("")

            log("POST-SPIKE VERIFICATION")

            for check_name, passed in spike_checks.items():

                log(
                    f"  {check_name}:",
                    (
                        "PASS"
                        if passed
                        else "FAIL"
                    ),
                )


            log(
                "Forced spike phase:",
                (
                    "PASS"
                    if not failed_spike_checks
                    else "FAIL"
                ),
            )

            log("")


            model_results[
                "forced_spike"
            ] = {
                "threshold":
                    pre_threshold,

                "pre_state":
                    pre_state,

                "post_state":
                    post_state,

                "spikes":
                    forced_spikes.copy(),

                "checks":
                    spike_checks,
            }


        # =====================================================================
        # MODEL RESULT
        # =====================================================================

        model_results[
            "passed"
        ] = model_passed


        results[
            "models"
        ][
            model_name
        ] = model_results


        if not model_passed:

            results[
                "passed"
            ] = False


        log(
            f"{model_name} RESULT:",
            (
                "PASS"
                if model_passed
                else "FAIL"
            ),
        )

        log("")


    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    log("=" * 90)
    log("SUMMARY")
    log("=" * 90)


    for model_name, model_data in results[
        "models"
    ].items():

        log(
            f"{model_name:<16}",
            (
                "PASS"
                if model_data["passed"]
                else "FAIL"
            ),
        )


    log("")

    log(
        "OVERALL:",
        (
            "PASS"
            if results["passed"]
            else "FAIL"
        ),
    )


    # =========================================================================
    # WRITE LOG
    # =========================================================================

    log_path.write_text(
        "\n".join(log_lines)
        + "\n",
        encoding="utf-8",
    )


    if verbose:

        print()
        print(
            f"Debug log written to: "
            f"{log_path}"
        )


    return results