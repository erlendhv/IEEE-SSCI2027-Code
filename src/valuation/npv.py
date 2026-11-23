"""
Field NPV per Monte Carlo scenario and exercise year, used as LSMC payoffs.

Prices are in USD and converted to NOK with the config conversion factors.
"""

import numpy as np


def compute_field_npvs(
    capped_oil: np.ndarray,
    capped_gas: np.ndarray,
    oil_prices: np.ndarray,
    gas_prices: np.ndarray,
    capex: float,
    opex_rate: float,
    variable_opex_rate: float,
    abex: float,
    discount_rate: float,
    start_year: int,
    nok_per_usd: float,
    barrel: float,
    btu: float,
) -> np.ndarray:
    """
    Compute NPV for one field on one host, per scenario and exercise year.

    For exercise at year t: revenue from capped production from year t,
    minus CAPEX at year t, fixed and variable OPEX, and ABEX at the last
    production year, all discounted to year t.

    Args:
        capped_oil: (n_years,) capped oil production, MSm3/y.
        capped_gas: (n_years,) capped gas production, BSm3/y.
        oil_prices: (n_scenarios, n_years) oil prices, USD/bbl.
        gas_prices: (n_scenarios, n_years) gas prices, USD/MMBtu.
        capex: Investment cost, MNOK.
        opex_rate: Fixed operating cost, MNOK/y.
        variable_opex_rate: Variable operating cost, USD/bbl.
        abex: Abandonment cost, MNOK.
        discount_rate: Annual discount rate.
        start_year: Earliest valid exercise year.
        nok_per_usd: Exchange rate, NOK/USD.
        barrel: Barrels per Sm3.
        btu: Million MMBtu per BSm3.

    Returns:
        (n_scenarios, n_years) NPV in MNOK if exercised at year t,
        discounted to year t. Years before start_year are -inf.
    """
    n_scenarios = oil_prices.shape[0]
    n_years = len(capped_oil)

    # Oil: MSm3 × bbl/Sm3 × USD/bbl × NOK/USD = MNOK
    # Gas: BSm3 × M-MMBtu/BSm3 × USD/MMBtu × NOK/USD = MNOK
    oil_revenue = oil_prices * (capped_oil[np.newaxis, :] * barrel * nok_per_usd)
    gas_revenue = gas_prices * (capped_gas[np.newaxis, :] * btu * nok_per_usd)
    revenue_flows = oil_revenue + gas_revenue

    has_production = (capped_oil + capped_gas) > 0

    npvs = np.full((n_scenarios, n_years), -np.inf)

    if not has_production.any():
        return npvs

    last_prod_year = int(np.max(np.where(has_production)))

    for t in range(start_year, n_years):
        remaining = n_years - t
        df = np.exp(-discount_rate * np.arange(remaining))

        rev = revenue_flows[:, t:] @ df

        # Fixed OPEX is charged only in years with production
        active_mask = has_production[t:].astype(float)
        opex_disc = opex_rate * float(active_mask @ df)

        var_opex_per_year = variable_opex_rate * capped_oil[t:] * barrel * nok_per_usd
        var_opex_disc = float(var_opex_per_year @ df)

        if t <= last_prod_year:
            abex_disc = abex * np.exp(-discount_rate * (last_prod_year - t))
        else:
            abex_disc = 0.0

        npvs[:, t] = rev - capex - opex_disc - var_opex_disc - abex_disc

    return npvs
