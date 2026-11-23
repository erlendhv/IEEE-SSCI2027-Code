"""Fitness evaluation: PortfolioAllocation to objectives (expected ROV, CVaR).

Pipeline: allocation -> profiles -> capacity capping -> NPV -> LSMC -> sum of field ROVs -> CVaR
"""

import numpy as np

from src.data_generation.profiles import logistic_watercut, make_oil_profile
from src.models import (
    CostData,
    FieldData,
    HostData,
    PortfolioAllocation,
    PriceScenarios,
    ValuationResult,
)
from src.optimizer.utils import calculate_cvar
from src.valuation.capacity import (
    compute_capped_production,
    compute_capacity_violation,
    compute_uncapped_production,
)
from src.valuation.lsmc import lsmc_field
from src.valuation.npv import compute_field_npvs


def evaluate_portfolio(
    allocation: PortfolioAllocation,
    field_data: FieldData,
    host_data: HostData,
    cost_data: CostData,
    prices: PriceScenarios,
    config: dict,
    use_lsmc: bool = True,
    use_capping: bool = True,
) -> ValuationResult:
    """Evaluate one portfolio allocation across all price scenarios.

    Args:
        allocation: Host assignment, start year and well count per field.
        prices: Oil and gas price scenarios shared by all evaluations.
        config: Uses config['valuation'] (discount_rate, cvar_alpha, barrel,
            btu, nok_per_usd) and config['data_generation']['per_well_rate'].
        use_lsmc: If True, value each field by LSMC (ROV with deferral
            option). If False, use static NPV at start_year with no
            deferral or go/no-go floor.
        use_capping: If True, apply FCFS and pro-rata capacity capping.
            If False, use uncapped profiles and compute the capacity
            violation for CDP.

    Returns:
        ValuationResult; violation is non-zero only when use_capping=False.
    """
    val_cfg = config["valuation"]
    dg_cfg = config["data_generation"]
    per_well_rate = dg_cfg["per_well_rate"]
    cutoff_fraction = dg_cfg.get("production_cutoff_fraction", 0.0)
    n_fields = allocation.n_fields
    n_scenarios = prices.n_scenarios
    n_years = config["n_years"]

    field_rovs = np.zeros((n_fields, n_scenarios))

    developed = allocation.assignment >= 0
    if not developed.any():
        return ValuationResult(
            expected_rov=0.0,
            cvar=0.0,
            field_values=np.zeros(n_fields),
            scenario_values=np.zeros(n_scenarios),
        )

    oil_profiles = np.zeros((n_fields, n_years))
    max_rates = np.zeros(n_fields)

    for f in range(n_fields):
        if allocation.assignment[f] < 0:
            continue
        n_wells_f = int(allocation.n_wells[f])
        max_rate = n_wells_f * per_well_rate * field_data.complexity_factor[f]
        max_rates[f] = max_rate

        oil_profiles[f] = make_oil_profile(
            n_years=n_years,
            ultimate_recovery=float(field_data.ultimate_recovery[f]),
            max_rate=max_rate,
            ramp_years=int(field_data.ramp_years[f]),
            decline_rate=float(field_data.decline_rate[f]),
            cutoff_fraction=cutoff_fraction,
        )

    # Water cut is a function of time since field start
    t = np.arange(n_years, dtype=float)
    wc_fractions = np.zeros((n_fields, n_years))
    for f in range(n_fields):
        if allocation.assignment[f] < 0:
            continue
        wc_fractions[f] = logistic_watercut(
            t, float(field_data.wc_midpoint[f]),
            float(field_data.wc_initial[f]),
            float(field_data.wc_final[f]),
            float(field_data.wc_steepness[f]),
        )

    if use_capping:
        production = compute_capped_production(
            oil_profiles, host_data, allocation, n_years,
            max_rates=max_rates,
            ramp_years=field_data.ramp_years,
            gor=field_data.gor,
            wc_fractions=wc_fractions,
        )
        violation = 0.0
    else:
        production = compute_uncapped_production(
            oil_profiles, host_data, allocation, n_years,
            gor=field_data.gor,
            wc_fractions=wc_fractions,
        )
        violation = compute_capacity_violation(production, host_data)

    for host_idx, host_result in production.items():
        field_indices = host_result["field_indices"]
        n_fields_on_host = len(field_indices)

        # OPEX is a shared platform cost, split evenly across fields on this host
        shared_opex = float(cost_data.opex_rate[host_idx]) / n_fields_on_host

        for local_idx, field_idx in enumerate(field_indices):
            start = int(allocation.start_year[field_idx])

            n_wells_f = int(allocation.n_wells[field_idx])
            capex = (
                float(cost_data.well_cost_per_well[field_idx]) * n_wells_f
                + cost_data.subsea_base
                + float(cost_data.pipeline_cost[field_idx, host_idx])
            )

            capped_oil = host_result["oil"][local_idx]
            capped_gas = host_result["gas"][local_idx]

            # NPV for each scenario and exercise year
            npvs = compute_field_npvs(
                capped_oil=capped_oil,
                capped_gas=capped_gas,
                oil_prices=prices.oil,
                gas_prices=prices.gas,
                capex=capex,
                opex_rate=shared_opex,
                variable_opex_rate=float(cost_data.variable_opex_rate[host_idx]),
                abex=float(cost_data.abex[host_idx]),
                discount_rate=val_cfg["discount_rate"],
                start_year=start,
                nok_per_usd=val_cfg["nok_per_usd"],
                barrel=val_cfg["barrel"],
                btu=val_cfg["btu"],
            )

            if use_lsmc:
                rov, _ = lsmc_field(
                    npvs=npvs,
                    oil_prices=prices.oil,
                    gas_prices=prices.gas,
                    discount_rate=val_cfg["discount_rate"],
                    start_year=start,
                )
            else:
                # Static NPV: develop unconditionally at start_year, no floor
                rov = npvs[:, start] * np.exp(
                    -val_cfg["discount_rate"] * start,
                )

            field_rovs[field_idx] = rov

    scenario_values = field_rovs.sum(axis=0)

    expected_rov = float(np.mean(scenario_values))
    cvar = calculate_cvar(scenario_values, alpha=val_cfg["cvar_alpha"])

    field_values = field_rovs.mean(axis=1)

    return ValuationResult(
        expected_rov=expected_rov,
        cvar=cvar,
        field_values=field_values,
        scenario_values=scenario_values,
        violation=violation,
    )
