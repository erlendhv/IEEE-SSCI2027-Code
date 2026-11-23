"""
Capacity capping for shared host platforms.

Oil uses first-come-first-served (FCFS) priority capping by start year,
with capped volume deferred to later years. Gas and water are re-derived
from capped oil and then capped pro-rata against their capacities.

Capping is computed once per particle from the PSO start years. LSMC can
only delay exercise beyond these years, so the capping is conservative.
"""

import logging

import numpy as np

from src.models import HostData, PortfolioAllocation

logger = logging.getLogger(__name__)


def apply_prorata_capping(
    production: np.ndarray,
    capacity: np.ndarray,
) -> np.ndarray:
    """
    Scale production proportionally in years where total exceeds capacity.

    Args:
        production: (n_fields, n_years) production rates for one stream.
        capacity: (n_years,) host capacity for that stream.

    Returns:
        Capped production, same shape as input.
    """
    total = production.sum(axis=0)

    # The `where` argument avoids division by zero when total is 0
    ratio = np.ones_like(total)
    exceeds = total > capacity
    np.divide(capacity, total, out=ratio, where=exceeds)

    return production * ratio[np.newaxis, :]


def apply_fcfs_capping(
    production: np.ndarray,
    capacity: np.ndarray,
    start_years: np.ndarray,
    max_rates: np.ndarray,
    ramp_years: np.ndarray,
    remaining_volumes: np.ndarray,
) -> np.ndarray:
    """
    FCFS priority capping with volume deferral for oil production.

    Fields with earlier start years get capacity first. Capped volume
    stays in the ground and is produced in later years, so total recovery
    is preserved up to the horizon. Fields with the same start year share
    the remaining capacity pro-rata.

    Args:
        production: (n_fields, n_years) shifted, uncapped oil rates.
        capacity: (n_years,) host oil capacity.
        start_years: (n_fields,) start year per field (priority key).
        max_rates: (n_fields,) plateau rate per field.
        ramp_years: (n_fields,) ramp-up duration per field.
        remaining_volumes: (n_fields,) ultimate recovery per field.
            Modified in place.

    Returns:
        (n_fields, n_years) capped oil production.
    """
    n_fields, n_years = production.shape
    actual = np.zeros((n_fields, n_years))

    unique_starts = np.unique(start_years)

    for t in range(n_years):
        desired = np.zeros(n_fields)
        for f in range(n_fields):
            field_year = t - int(start_years[f])
            if field_year < 0 or remaining_volumes[f] <= 0:
                continue

            # During ramp-up the rate follows the ramp profile; after it,
            # deferred volume lets the field hold its plateau rate
            if field_year < int(ramp_years[f]):
                ceiling = production[f, t]
            else:
                ceiling = max_rates[f]

            desired[f] = min(ceiling, remaining_volumes[f])

        remaining_cap = capacity[t]
        for sy in unique_starts:
            group = np.where(start_years == sy)[0]
            group_desired = desired[group]
            group_total = group_desired.sum()

            if group_total <= 0:
                continue

            if group_total <= remaining_cap:
                actual[group, t] = group_desired
            else:
                if remaining_cap <= 0:
                    break
                ratio = remaining_cap / group_total
                actual[group, t] = group_desired * ratio

            remaining_cap -= actual[group, t].sum()

        remaining_volumes -= actual[:, t]

    return actual


def _shift_profile(
    profile: np.ndarray,
    start_year: int,
    n_years: int,
) -> np.ndarray:
    """
    Shift a production profile to start at start_year, truncated at n_years.

    Args:
        profile: (n_profile_years,) production starting at year 0.
        start_year: Year when production begins.
        n_years: Timeline length.

    Returns:
        (n_years,) shifted profile.
    """
    shifted = np.zeros(n_years)
    usable = min(len(profile), n_years - start_year)
    if usable > 0:
        shifted[start_year:start_year + usable] = profile[:usable]
    return shifted


def compute_capped_production(
    oil_profiles: np.ndarray,
    host_data: HostData,
    allocation: PortfolioAllocation,
    n_years: int,
    max_rates: np.ndarray,
    ramp_years: np.ndarray,
    gor: np.ndarray,
    wc_fractions: np.ndarray,
) -> dict[int, dict]:
    """
    Compute capped production for all hosts in a portfolio.

    Oil is FCFS-capped with deferral; gas and water are derived from
    capped oil and then capped pro-rata.

    Args:
        oil_profiles: (n_fields, n_profile_years) oil production from year 0.
        host_data: Host capacities.
        allocation: Field-to-host assignments and start years.
        n_years: Timeline length.
        max_rates: (n_fields,) plateau rate per field, MSm3/y.
        ramp_years: (n_fields,) ramp-up duration per field.
        gor: (n_fields,) gas-oil ratio, Sm3 gas / Sm3 oil.
        wc_fractions: (n_fields, n_years) water cut fractions at
            field-relative time (year 0 = development start).

    Returns:
        Dict keyed by host index, each value a dict with 'oil', 'gas',
        'water' as (n_fields_on_host, n_years) capped production and
        'field_indices' as the original field indices on the host.
    """
    result = {}

    for host_idx in range(host_data.n_hosts):
        field_indices = [
            i for i in range(allocation.n_fields)
            if allocation.assignment[i] == host_idx
        ]

        if not field_indices:
            continue

        n_local = len(field_indices)

        oil = np.zeros((n_local, n_years))
        for local_idx, field_idx in enumerate(field_indices):
            start = int(allocation.start_year[field_idx])
            oil[local_idx] = _shift_profile(
                oil_profiles[field_idx], start, n_years,
            )

        local_starts = np.array([
            int(allocation.start_year[fi]) for fi in field_indices
        ])
        local_max_rates = max_rates[field_indices]
        local_ramp_years = ramp_years[field_indices]

        # The sum of the uncapped profile equals ultimate recovery
        local_remaining = np.array([
            oil_profiles[fi].sum() for fi in field_indices
        ])

        capped_oil = apply_fcfs_capping(
            oil,
            host_data.oil_capacity[host_idx],
            local_starts,
            local_max_rates,
            local_ramp_years,
            local_remaining,
        )

        # GOR / 1000 converts MSm3 oil to BSm3 gas
        capped_gas = np.zeros((n_local, n_years))
        for local_idx, field_idx in enumerate(field_indices):
            capped_gas[local_idx] = capped_oil[local_idx] * gor[field_idx] / 1000.0

        # water = oil * wc / (1 - wc), with wc at field-relative time
        capped_water = np.zeros((n_local, n_years))
        for local_idx, field_idx in enumerate(field_indices):
            start = int(allocation.start_year[field_idx])
            for t in range(start, n_years):
                field_year = t - start
                if field_year >= wc_fractions.shape[1]:
                    break
                wc = wc_fractions[field_idx, field_year]
                oil_t = capped_oil[local_idx, t]
                if oil_t > 0 and wc < 1.0:
                    capped_water[local_idx, t] = oil_t * wc / (1.0 - wc)

        capped_gas = apply_prorata_capping(
            capped_gas, host_data.gas_capacity[host_idx],
        )
        capped_water = apply_prorata_capping(
            capped_water, host_data.water_capacity[host_idx],
        )

        result[host_idx] = {
            "oil": capped_oil,
            "gas": capped_gas,
            "water": capped_water,
            "field_indices": field_indices,
        }

    return result


def compute_uncapped_production(
    oil_profiles: np.ndarray,
    host_data: HostData,
    allocation: PortfolioAllocation,
    n_years: int,
    gor: np.ndarray,
    wc_fractions: np.ndarray,
) -> dict[int, dict]:
    """
    Compute production grouped by host without capacity capping.

    Used when the optimizer handles capacity constraints explicitly
    through constrained dominance (CDP).

    Args:
        oil_profiles: (n_fields, n_profile_years) oil production from year 0.
        host_data: Host data, used only for the number of hosts.
        allocation: Field-to-host assignments and start years.
        n_years: Timeline length.
        gor: (n_fields,) gas-oil ratio, Sm3/Sm3.
        wc_fractions: (n_fields, n_years) water cut fractions at
            field-relative time.

    Returns:
        Same format as compute_capped_production.
    """
    result = {}

    for host_idx in range(host_data.n_hosts):
        field_indices = [
            i for i in range(allocation.n_fields)
            if allocation.assignment[i] == host_idx
        ]
        if not field_indices:
            continue

        n_local = len(field_indices)

        oil = np.zeros((n_local, n_years))
        for local_idx, field_idx in enumerate(field_indices):
            start = int(allocation.start_year[field_idx])
            oil[local_idx] = _shift_profile(
                oil_profiles[field_idx], start, n_years,
            )

        gas = np.zeros((n_local, n_years))
        for local_idx, field_idx in enumerate(field_indices):
            gas[local_idx] = oil[local_idx] * gor[field_idx] / 1000.0

        water = np.zeros((n_local, n_years))
        for local_idx, field_idx in enumerate(field_indices):
            start = int(allocation.start_year[field_idx])
            for t in range(start, n_years):
                field_year = t - start
                if field_year >= wc_fractions.shape[1]:
                    break
                wc = wc_fractions[field_idx, field_year]
                oil_t = oil[local_idx, t]
                if oil_t > 0 and wc < 1.0:
                    water[local_idx, t] = oil_t * wc / (1.0 - wc)

        result[host_idx] = {
            "oil": oil,
            "gas": gas,
            "water": water,
            "field_indices": field_indices,
        }

    return result


def compute_capacity_violation(
    production: dict[int, dict],
    host_data: HostData,
) -> float:
    """
    Sum max(0, production - capacity) over hosts, years, and streams.

    CDP uses the total to rank infeasible solutions.

    Args:
        production: Output from compute_uncapped_production or
            compute_capped_production.
        host_data: Host capacities.

    Returns:
        Total violation, 0.0 if all production is within capacity.
    """
    violation = 0.0

    for host_idx, host_result in production.items():
        oil_total = host_result["oil"].sum(axis=0)
        gas_total = host_result["gas"].sum(axis=0)
        water_total = host_result["water"].sum(axis=0)

        violation += np.maximum(
            0.0, oil_total - host_data.oil_capacity[host_idx],
        ).sum()
        violation += np.maximum(
            0.0, gas_total - host_data.gas_capacity[host_idx],
        ).sum()
        violation += np.maximum(
            0.0, water_total - host_data.water_capacity[host_idx],
        ).sum()

    return float(violation)
