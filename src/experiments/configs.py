"""Experiment config variants. Each experiment is a list of (name, overrides) tuples applied to the base config."""


def get_experiment_configs(experiment_id: str) -> list[tuple[str, dict]]:
    """Return config variants for the given experiment.

    Args:
        experiment_id: Key in the experiment table below, e.g. "E1.1" or "E7.3".

    Returns:
        List of (config_name, overrides_dict) tuples.

    Raises:
        ValueError: If experiment_id is not recognized.
    """
    experiments = {
        "E1.1": _e1_1_encoding,
        "E2.1": _e2_1_constraints,
        "E3.1": _e3_1_rov_vs_npv,
        "E4.1": _e4_1_stickiness,
        "E5.2": _e5_2_sensitivity,
        "E7.1": _e7_1_update_weights,
        "E7.2": _e7_2_swarm_budget,
        "E7.3": _e7_3_algorithm_robustness,
        "E7.4": _e7_4_dsbpso_parameters,
        "E7.5": _e7_5_scenario_count,
        "E7.6": _e7_6_nsga2_benchmark,
        "E7.7": _e7_7_bpso_swarm_budget,
        "E7.8": _e7_8_bpso_scenario_count,
    }

    if experiment_id not in experiments:
        valid = ", ".join(sorted(experiments))
        raise ValueError(
            f"Unknown experiment '{experiment_id}'. Valid: {valid}"
        )

    return experiments[experiment_id]()


# ── E1.1: Gray code vs standard binary per variable type ────────


def _e1_1_encoding() -> list[tuple[str, dict]]:
    return [
        ("all_gray", {
            "pso.encoding.assignment": "gray",
            "pso.encoding.timing": "gray",
            "pso.encoding.wells": "gray",
        }),
        ("all_binary", {
            "pso.encoding.assignment": "binary",
            "pso.encoding.timing": "binary",
            "pso.encoding.wells": "binary",
        }),
        ("assignment_binary", {
            "pso.encoding.assignment": "binary",
            "pso.encoding.timing": "gray",
            "pso.encoding.wells": "gray",
        }),
        ("timing_binary", {
            "pso.encoding.assignment": "gray",
            "pso.encoding.timing": "binary",
            "pso.encoding.wells": "gray",
        }),
        ("wells_binary", {
            "pso.encoding.assignment": "gray",
            "pso.encoding.timing": "gray",
            "pso.encoding.wells": "binary",
        }),
    ]


# ── E2.1: Repair (capping) vs CDP ───────────────────────────────


def _e2_1_constraints() -> list[tuple[str, dict]]:
    return [
        ("capping", {
            "pso.constraint_handling": "capping",
        }),
        ("cdp", {
            "pso.constraint_handling": "cdp",
        }),
    ]


# ── E3.1: ROV (LSMC) vs static NPV ─────────────────────────────


def _e3_1_rov_vs_npv() -> list[tuple[str, dict]]:
    return [
        ("rov", {
            "pso.use_lsmc": True,
        }),
        ("npv", {
            "pso.use_lsmc": False,
        }),
    ]


# ── E4.1: DSBPSO vs Standard BPSO ───────────────────────────────


def _e4_1_stickiness() -> list[tuple[str, dict]]:
    return [
        ("dsbpso", {
            "pso.algorithm": "dsbpso",
        }),
        ("bpso_constant_w", {
            "pso.algorithm": "bpso",
            "pso.bpso.w_schedule": "constant",
            "pso.bpso.w_initial": 0.9,
            "pso.bpso.w_final": 0.9,
        }),
        ("bpso_increasing_w", {
            "pso.algorithm": "bpso",
            "pso.bpso.w_schedule": "increasing",
            "pso.bpso.w_initial": 0.4,
            "pso.bpso.w_final": 0.9,
        }),
    ]


# ── E5.2: Sensitivity analysis (OAT parameter variation) ─────

# Baseline defaults from config/default.yaml
_BASELINE_DISCOUNT_RATE = 0.08
_BASELINE_CVAR_ALPHA = 0.05
_BASELINE_OIL_SIGMA_E = 0.219507719
_BASELINE_OIL_SIGMA_X = 0.466884607
_BASELINE_GAS_SIGMA_E = 0.245022334
_BASELINE_GAS_SIGMA_X = 0.748697922
_BASELINE_HOST_OIL_CAP_MIN = 0.1
_BASELINE_HOST_OIL_CAP_MAX = 0.8
_BASELINE_HOST_GAS_CAP_MIN = 0.05
_BASELINE_HOST_GAS_CAP_MAX = 0.3
_BASELINE_HOST_WATER_CAP_MIN = 0.3
_BASELINE_HOST_WATER_CAP_MAX = 1.5
_BASELINE_OIL_IN_PLACE_MEAN = 10.0
_BASELINE_OIL_IN_PLACE_STD = 3.0
_BASELINE_OIL_E0 = 4.26
_BASELINE_WELL_COST_BASE = 400
_BASELINE_SUBSEA_BASE = 2000
_BASELINE_PIPELINE_COST_PER_KM2 = 3.0


def _e5_2_sensitivity() -> list[tuple[str, dict]]:
    configs: list[tuple[str, dict]] = []

    configs.append(("baseline", {}))

    # Discount rate (baseline 8%)
    for rate in [0.04, 0.06, 0.10, 0.12]:
        pct = int(rate * 100)
        configs.append((f"discount_{pct}pct", {
            "valuation.discount_rate": rate,
        }))

    # Price volatility scale
    for scale in [0.5, 0.75, 1.25, 1.5]:
        label = str(scale).replace(".", "p")
        configs.append((f"volatility_{label}x", {
            "price.oil.sigma_E": _BASELINE_OIL_SIGMA_E * scale,
            "price.oil.sigma_X": _BASELINE_OIL_SIGMA_X * scale,
            "price.gas.sigma_E": _BASELINE_GAS_SIGMA_E * scale,
            "price.gas.sigma_X": _BASELINE_GAS_SIGMA_X * scale,
        }))

    # Host capacity scale
    for scale in [0.5, 0.75, 1.5, 2.0]:
        label = str(scale).replace(".", "p")
        configs.append((f"capacity_{label}x", {
            "data_generation.host_oil_capacity_min": _BASELINE_HOST_OIL_CAP_MIN * scale,
            "data_generation.host_oil_capacity_max": _BASELINE_HOST_OIL_CAP_MAX * scale,
            "data_generation.host_gas_capacity_min": _BASELINE_HOST_GAS_CAP_MIN * scale,
            "data_generation.host_gas_capacity_max": _BASELINE_HOST_GAS_CAP_MAX * scale,
            "data_generation.host_water_capacity_min": _BASELINE_HOST_WATER_CAP_MIN * scale,
            "data_generation.host_water_capacity_max": _BASELINE_HOST_WATER_CAP_MAX * scale,
        }))

    # Field size scale; mean and std scale together to keep the coefficient of variation constant
    for scale in [0.5, 0.75, 1.25, 1.5]:
        label = str(scale).replace(".", "p")
        configs.append((f"fieldsize_{label}x", {
            "data_generation.oil_in_place_mean": _BASELINE_OIL_IN_PLACE_MEAN * scale,
            "data_generation.oil_in_place_std": _BASELINE_OIL_IN_PLACE_STD * scale,
        }))

    # Oil price level (baseline E0 = 4.26, about 71 USD/bbl); E0 is a log-price, so additive shifts give multiplicative price changes
    for shift in [-0.3, -0.15, 0.15, 0.44]:
        approx_usd = int(round(__import__("math").exp(_BASELINE_OIL_E0 + shift)))
        configs.append((f"oilprice_{approx_usd}usd", {
            "price.oil.E0": _BASELINE_OIL_E0 + shift,
        }))

    # All CAPEX components scaled together
    for scale in [0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 4.0]:
        label = str(scale).replace(".", "p")
        configs.append((f"capex_{label}x", {
            "data_generation.well_cost_base": _BASELINE_WELL_COST_BASE * scale,
            "data_generation.subsea_base": _BASELINE_SUBSEA_BASE * scale,
            "data_generation.pipeline_cost_per_km2": _BASELINE_PIPELINE_COST_PER_KM2 * scale,
        }))

    # Each CAPEX component scaled on its own, to see which one drives portfolio value
    for scale in [0.5, 1.5, 2.0, 4.0]:
        label = str(scale).replace(".", "p")

        configs.append((f"wellcost_{label}x", {
            "data_generation.well_cost_base": _BASELINE_WELL_COST_BASE * scale,
        }))

        configs.append((f"subsea_{label}x", {
            "data_generation.subsea_base": _BASELINE_SUBSEA_BASE * scale,
        }))

        configs.append((f"pipeline_{label}x", {
            "data_generation.pipeline_cost_per_km2": _BASELINE_PIPELINE_COST_PER_KM2 * scale,
        }))

    return configs


# ── E7.1: Update-weight sensitivity (DSBPSO rho, BPSO c1/c2 and w) ──

_E7_1_RATIOS = [0.5, 1.0, 2.0, 4.0, 8.0]
_E7_1_BPSO_C_SUM = 4.0
_E7_1_BPSO_W_BASELINE = 0.9
_E7_1_BPSO_W_VALUES = [0.4, 0.6, 0.8]


def _bpso_constant_w(w: float, c1: float, c2: float) -> dict:
    """Overrides for standard BPSO with constant inertia weight.

    Args:
        w: Inertia weight.
        c1: Cognitive coefficient (pull toward personal best).
        c2: Social coefficient (pull toward archive leader).

    Returns:
        Dot-notation override dict.
    """
    return {
        "pso.algorithm": "bpso",
        "pso.bpso.w_schedule": "constant",
        "pso.bpso.w_initial": w,
        "pso.bpso.w_final": w,
        "pso.bpso.c1": c1,
        "pso.bpso.c2": c2,
    }


def _e7_1_update_weights() -> list[tuple[str, dict]]:
    configs: list[tuple[str, dict]] = []

    # DSBPSO: importance ratio rho = i_p / i_g (baseline 2.0)
    for ratio in _E7_1_RATIOS:
        label = str(ratio).replace(".", "p")
        configs.append((f"dsbpso_rho_{label}", {
            "pso.algorithm": "dsbpso",
            "pso.alpha": ratio,
        }))

    # BPSO: ratio c1 / c2 with c1 + c2 fixed (baseline c1 = c2 = 2.0)
    for ratio in _E7_1_RATIOS:
        label = str(ratio).replace(".", "p")
        c2 = _E7_1_BPSO_C_SUM / (1.0 + ratio)
        c1 = ratio * c2
        configs.append((
            f"bpso_cratio_{label}",
            _bpso_constant_w(_E7_1_BPSO_W_BASELINE, c1, c2),
        ))

    # BPSO: constant inertia weight w (w = 0.9 covered by cratio_1p0)
    half_sum = _E7_1_BPSO_C_SUM / 2.0
    for w in _E7_1_BPSO_W_VALUES:
        label = str(w).replace(".", "p")
        configs.append((
            f"bpso_w_{label}",
            _bpso_constant_w(w, half_sum, half_sum),
        ))

    return configs


# ── E7.2: Swarm size N x generations G (DSBPSO only) ────────────

_E7_2_N_PARTICLES = [50, 100, 200, 400]
_E7_2_N_ITERATIONS = [500, 1000, 2000, 4000]


def _e7_2_swarm_budget() -> list[tuple[str, dict]]:
    configs: list[tuple[str, dict]] = []

    for n_particles in _E7_2_N_PARTICLES:
        for n_iterations in _E7_2_N_ITERATIONS:
            configs.append((f"n{n_particles}_g{n_iterations}", {
                "pso.algorithm": "dsbpso",
                "pso.n_particles": n_particles,
                "pso.max_iterations": n_iterations,
            }))

    return configs


# ── E7.3: DSBPSO vs BPSO across problem settings ────────────────

# (n_fields, n_hosts); baseline is (9, 3)
_E7_3_PROBLEM_SIZES = [(18, 6), (18, 3), (5, 2)]


def _e7_3_problem_settings() -> list[tuple[str, dict]]:
    """Problem settings for the algorithm-robustness comparison.

    Returns:
        List of (setting_name, overrides) tuples: every E5.2 setting followed by the problem-size settings.
    """
    settings = _e5_2_sensitivity()

    for n_fields, n_hosts in _E7_3_PROBLEM_SIZES:
        settings.append((f"size_{n_fields}f_{n_hosts}h", {
            "n_fields": n_fields,
            "n_hosts": n_hosts,
        }))

    return settings


def _e7_3_algorithm_robustness() -> list[tuple[str, dict]]:
    half_sum = _E7_1_BPSO_C_SUM / 2.0
    bpso_overrides = _bpso_constant_w(_E7_1_BPSO_W_BASELINE, half_sum, half_sum)

    configs: list[tuple[str, dict]] = []
    for setting_name, setting_overrides in _e7_3_problem_settings():
        configs.append((f"{setting_name}_dsbpso", {
            **setting_overrides,
            "pso.algorithm": "dsbpso",
        }))
        configs.append((f"{setting_name}_bpso", {
            **setting_overrides,
            **bpso_overrides,
        }))

    return configs


# ── E7.4: DSBPSO exploration and unsticking period ──────────────

_E7_4_IS_FACTORS = [5.0, 10.0, 20.0]
_E7_4_USTK_PCTS = [(0.5, 5.0), (1.0, 10.0), (2.0, 20.0)]
_E7_4_IS_FACTOR_BASELINE = 10.0
_E7_4_USTK_PCT_BASELINE = (1.0, 10.0)


def _e7_4_dsbpso_parameters() -> list[tuple[str, dict]]:
    configs: list[tuple[str, dict]] = []

    # Initial exploration i_s = factor / n_bits (baseline 10, holds default ustk)
    for factor in _E7_4_IS_FACTORS:
        label = str(factor).replace(".", "p")
        configs.append((f"is_{label}", {
            "pso.algorithm": "dsbpso",
            "pso.is_initial_factor": factor,
        }))

    # Unsticking period start/end % of iterations (baseline 1 -> 10 is in is_10p0)
    for start_pct, end_pct in _E7_4_USTK_PCTS:
        if (start_pct, end_pct) == _E7_4_USTK_PCT_BASELINE:
            continue
        start_label = str(start_pct).replace(".", "p")
        end_label = str(end_pct).replace(".", "p")
        configs.append((f"ustk_{start_label}_{end_label}", {
            "pso.algorithm": "dsbpso",
            "pso.is_initial_factor": _E7_4_IS_FACTOR_BASELINE,
            "pso.ustk_initial_pct": start_pct,
            "pso.ustk_final_pct": end_pct,
        }))

    return configs


# ── E7.5: Optimization scenario count (DSBPSO only) ─────────────

_E7_5_N_SCENARIOS = [50, 100, 200, 500]


def _e7_5_scenario_count() -> list[tuple[str, dict]]:
    configs: list[tuple[str, dict]] = []

    for n_scenarios in _E7_5_N_SCENARIOS:
        configs.append((f"s{n_scenarios}", {
            "pso.algorithm": "dsbpso",
            "price.n_scenarios": n_scenarios,
        }))

    return configs


# ── E7.6: NSGA-II benchmark across all E7.3 problem settings ────


def _e7_6_nsga2_benchmark() -> list[tuple[str, dict]]:
    configs: list[tuple[str, dict]] = []

    # Compared against the {setting}_dsbpso and {setting}_bpso runs of E7.3
    for setting_name, setting_overrides in _e7_3_problem_settings():
        configs.append((f"{setting_name}_nsga2", {
            **setting_overrides,
            "pso.algorithm": "nsga2",
        }))

    return configs


# ── E7.7: Swarm size N x generations G (both BPSO variants) ─────


def _e4_1_bpso_variants() -> list[tuple[str, dict]]:
    """Return the bpso_constant_w and bpso_increasing_w variants of E4.1 as (variant_name, overrides) tuples."""
    return [
        (name, overrides)
        for name, overrides in _e4_1_stickiness()
        if overrides["pso.algorithm"] == "bpso"
    ]


def _e7_7_bpso_swarm_budget() -> list[tuple[str, dict]]:
    configs: list[tuple[str, dict]] = []

    # Same grid as E7.2, compared against its n{N}_g{G} DSBPSO runs
    for variant_name, variant_overrides in _e4_1_bpso_variants():
        for n_particles in _E7_2_N_PARTICLES:
            for n_iterations in _E7_2_N_ITERATIONS:
                configs.append((f"{variant_name}_n{n_particles}_g{n_iterations}", {
                    **variant_overrides,
                    "pso.n_particles": n_particles,
                    "pso.max_iterations": n_iterations,
                }))

    return configs


# ── E7.8: Optimization scenario count (both BPSO variants) ──────


def _e7_8_bpso_scenario_count() -> list[tuple[str, dict]]:
    configs: list[tuple[str, dict]] = []

    # Same scenario counts as E7.5, compared against its s{S} DSBPSO runs
    for variant_name, variant_overrides in _e4_1_bpso_variants():
        for n_scenarios in _E7_5_N_SCENARIOS:
            configs.append((f"{variant_name}_s{n_scenarios}", {
                **variant_overrides,
                "price.n_scenarios": n_scenarios,
            }))

    return configs
