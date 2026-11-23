"""
Stochastic field production profiles.

  - Oil: linear ramp, plateau, exponential decline, scaled to ultimate recovery (UR)
  - Gas: oil × gas-oil ratio (GOR)
  - Water: oil × wc/(1-wc) with a logistic water-cut curve

Profiles start at year 0 (development start). The optimizer shifts them
by start_year in PortfolioAllocation.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ── Curve helpers ────────────────────────────────────────────────


def triangular_ramp(
    t: np.ndarray, start: float, ramp_years: float, peak: float
) -> np.ndarray:
    """Linear ramp from 0 at start to peak after ramp_years (t in years)."""
    x = (t - start) / max(ramp_years, 1e-9)
    return peak * np.clip(x, 0.0, 1.0)


def exp_decline(
    t: np.ndarray, start: float, q0: float, decline_rate: float
) -> np.ndarray:
    """Exponential decline q0 * exp(-decline_rate * (t - start)) for t >= start, else 0."""
    dt = np.maximum(t - start, 0.0)
    return q0 * np.exp(-decline_rate * dt) * (t >= start)


def logistic_watercut(
    t: np.ndarray, t0: float, wc0: float, wc_max: float, k: float
) -> np.ndarray:
    """
    Logistic water-cut curve rising from wc0 to wc_max.

    Args:
        t0: Midpoint year.
        wc0: Initial water cut fraction.
        wc_max: Maximum water cut fraction.
        k: Steepness.
    """
    return wc0 + (wc_max - wc0) / (1.0 + np.exp(-k * (t - t0)))


# ── Profile generation functions ─────────────────────────────────


def make_oil_profile(
    n_years: int,
    ultimate_recovery: float,
    max_rate: float,
    ramp_years: int,
    decline_rate: float,
    cutoff_fraction: float = 0.0,
) -> np.ndarray:
    """
    Generate one field's oil profile: ramp, plateau at max_rate, exponential
    decline, scaled to UR.

    The plateau lasts as long as needed to produce UR at max_rate. Years
    below cutoff_fraction × peak rate are set to zero (abandonment), which
    sets the year ABEX is charged. The lost tail volume is not rescaled.

    Args:
        n_years: Profile length.
        ultimate_recovery: Recoverable volume, MSm3.
        max_rate: Plateau rate, MSm3/y.
        ramp_years: Years of linear build-up to max_rate.
        decline_rate: Exponential decline rate after the plateau.
        cutoff_fraction: Abandonment threshold as a fraction of peak rate.
            0.0 means no cutoff.

    Returns:
        (n_years,) oil rate, MSm3/y.
    """
    t = np.arange(n_years, dtype=float)
    ramp_end = float(ramp_years)

    # Plateau volume = UR minus ramp volume (ramp_years × max_rate / 2)
    # minus decline volume (integral to infinity, max_rate / decline_rate)
    ramp_volume = ramp_years * max_rate / 2.0
    decline_volume = max_rate / decline_rate
    plateau_volume = max(ultimate_recovery - ramp_volume - decline_volume, 0.0)
    plateau_years = plateau_volume / max_rate if max_rate > 0 else 0.0

    plateau_end = ramp_end + plateau_years

    ramp_mask = t < ramp_end
    plateau_mask = (t >= ramp_end) & (t < plateau_end)
    decline_mask = t >= plateau_end

    q = np.zeros(n_years)
    q += triangular_ramp(t, 0.0, ramp_years, max_rate) * ramp_mask
    q += max_rate * plateau_mask
    q += exp_decline(t, plateau_end, max_rate, decline_rate) * decline_mask

    total = q.sum()
    if total > 0:
        q *= ultimate_recovery / total

    if cutoff_fraction > 0:
        peak = q.max()
        if peak > 0:
            q[q < peak * cutoff_fraction] = 0.0

    return q


def make_gas_profile(
    oil_profile: np.ndarray, gor: float
) -> np.ndarray:
    """
    Gas production from oil and gas-oil ratio.

    Args:
        oil_profile: Oil rate, MSm3/y.
        gor: Gas-oil ratio, Sm3 gas / Sm3 oil.

    Returns:
        Gas rate, BSm3/y (MSm3 oil × GOR = MSm3 gas, / 1000 = BSm3).
    """
    return oil_profile * gor / 1000.0


def make_water_profile(
    oil_profile: np.ndarray,
    n_years: int,
    wc0: float,
    wc_max: float,
    wc_midpoint: float,
    wc_steepness: float,
) -> np.ndarray:
    """
    Water production from oil and a logistic water-cut curve.

    With wc = water / (oil + water), water = oil * wc / (1 - wc).

    Args:
        oil_profile: Oil rate, MSm3/y.
        n_years: Profile length.
        wc0: Initial water cut fraction.
        wc_max: Maximum water cut fraction.
        wc_midpoint: Midpoint year of the logistic curve.
        wc_steepness: Logistic steepness.

    Returns:
        Water rate, MSm3/y.
    """
    t = np.arange(n_years, dtype=float)
    wc = logistic_watercut(t, wc_midpoint, wc0, wc_max, wc_steepness)

    wc = np.where(oil_profile > 0, wc, 0.0)

    eps = 1e-9
    water = oil_profile * (wc / np.maximum(1.0 - wc, eps))
    return water


# ── Main field profile generator ─────────────────────────────────


def generate_field_profiles(
    config: dict,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, list[str],
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Generate production profiles and reservoir attributes for all fields.

    Samples oil in place, recovery factor, complexity, and depth per field,
    derives n_wells and max_rate from them, then builds the profiles.

    Args:
        config: Full config dict.

    Returns:
        Tuple of (oil, gas, water, names, ur, complexity, depth, n_wells, gor,
        ramp_years, decline_rate, wc_initial, wc_final, wc_midpoint,
        wc_steepness):
        - oil, water: (n_fields, n_years) MSm3/y
        - gas: (n_fields, n_years) BSm3/y
        - names: field names
        - ur: (n_fields,) ultimate recovery, MSm3
        - complexity: (n_fields,) per-well productivity multiplier
        - depth: (n_fields,) reservoir depth, m
        - n_wells: (n_fields,) int, number of wells
        - gor: (n_fields,) gas-oil ratio, Sm3/Sm3
        - ramp_years: (n_fields,) int, build-up duration
        - decline_rate: (n_fields,) exponential decline rate
        - wc_initial, wc_final: (n_fields,) initial and maximum water cut
        - wc_midpoint, wc_steepness: (n_fields,) logistic midpoint year and steepness
    """
    n_fields = config["n_fields"]
    n_years = config["n_years"]
    dg = config["data_generation"]

    oil = np.zeros((n_fields, n_years))
    gas = np.zeros((n_fields, n_years))
    water = np.zeros((n_fields, n_years))
    names = [f"Field_{i+1}" for i in range(n_fields)]
    ur_array = np.zeros(n_fields)
    complexity_array = np.zeros(n_fields)
    depth_array = np.zeros(n_fields)
    n_wells_array = np.zeros(n_fields, dtype=int)
    gor_array = np.zeros(n_fields)
    ramp_years_array = np.zeros(n_fields, dtype=int)
    decline_rate_array = np.zeros(n_fields)
    wc_initial_array = np.zeros(n_fields)
    wc_final_array = np.zeros(n_fields)
    wc_midpoint_array = np.zeros(n_fields)
    wc_steepness_array = np.zeros(n_fields)

    for i in range(n_fields):
        oil_in_place = rng.normal(dg["oil_in_place_mean"], dg["oil_in_place_std"])
        oil_in_place = max(oil_in_place, dg["oil_in_place_min"])
        rf = rng.uniform(dg["recovery_factor_min"], dg["recovery_factor_max"])
        ur = oil_in_place * rf

        complexity = rng.uniform(dg["complexity_factor_min"], dg["complexity_factor_max"])
        depth = rng.uniform(dg["depth_min"], dg["depth_max"])
        n_wells = max(1, round(ur * dg["wells_per_msm3"] / complexity))

        max_rate = n_wells * dg["per_well_rate"] * complexity

        ramp_yrs = int(rng.integers(dg["ramp_years_min"], dg["ramp_years_max"] + 1))
        dec_rate = rng.uniform(dg["decline_rate_min"], dg["decline_rate_max"])

        gor = rng.lognormal(mean=dg["gor_log_mean"], sigma=dg["gor_log_std"])
        gor = np.clip(gor, dg["gor_min"], dg["gor_max"])

        wc0 = rng.uniform(0.0, dg["wc_initial_max"])
        wc_max = rng.uniform(dg["wc_final_min"], dg["wc_final_max"])
        wc_max = max(wc_max, wc0 + 0.05)
        wc_mid_offset = int(rng.integers(
            dg["wc_midpoint_offset_min"], dg["wc_midpoint_offset_max"] + 1
        ))
        wc_k = rng.uniform(dg["wc_steepness_min"], dg["wc_steepness_max"])

        cutoff = dg.get("production_cutoff_fraction", 0.0)
        oil[i] = make_oil_profile(n_years, ur, max_rate, ramp_yrs, dec_rate, cutoff)
        gas[i] = make_gas_profile(oil[i], gor)
        water[i] = make_water_profile(
            oil[i], n_years, wc0, wc_max, float(wc_mid_offset), wc_k,
        )

        ur_array[i] = ur
        complexity_array[i] = complexity
        depth_array[i] = depth
        n_wells_array[i] = n_wells
        gor_array[i] = gor
        ramp_years_array[i] = ramp_yrs
        decline_rate_array[i] = dec_rate
        wc_initial_array[i] = wc0
        wc_final_array[i] = wc_max
        wc_midpoint_array[i] = float(wc_mid_offset)
        wc_steepness_array[i] = wc_k

        logger.info(
            "Field_%d: UR=%.1f MSm3, wells=%d, complexity=%.2f, "
            "max_rate=%.2f MSm3/y, depth=%.0fm, GOR=%.0f",
            i + 1, ur, n_wells, complexity, max_rate, depth, gor,
        )

    return (
        oil, gas, water, names,
        ur_array, complexity_array, depth_array, n_wells_array, gor_array,
        ramp_years_array, decline_rate_array,
        wc_initial_array, wc_final_array, wc_midpoint_array, wc_steepness_array,
    )
