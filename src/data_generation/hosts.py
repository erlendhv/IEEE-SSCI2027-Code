"""Host platform capacity generation. Capacity does not depend on which fields are assigned."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def generate_host_capacities(
    config: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Generate capacity profiles for all host platforms.

    Each host gets a base capacity sampled from config ranges, with small
    annual growth and year-to-year noise.

    Args:
        config: Full experiment config dict.
        rng: NumPy random generator.

    Returns:
        Tuple of (oil_cap, gas_cap, water_cap, names) where:
        - oil_cap: (n_hosts, n_years) MSm3/y
        - gas_cap: (n_hosts, n_years) BSm3/y
        - water_cap: (n_hosts, n_years) MSm3/y
        - names: list of host name strings
    """
    n_hosts = config["n_hosts"]
    n_years = config["n_years"]
    dg = config["data_generation"]

    oil_cap = np.zeros((n_hosts, n_years))
    gas_cap = np.zeros((n_hosts, n_years))
    water_cap = np.zeros((n_hosts, n_years))
    names = [f"Host_{j+1}" for j in range(n_hosts)]

    for j in range(n_hosts):
        # Sample base capacities from config ranges
        oil_base = rng.uniform(dg["host_oil_capacity_min"], dg["host_oil_capacity_max"])
        gas_base = rng.uniform(dg["host_gas_capacity_min"], dg["host_gas_capacity_max"])
        water_base = rng.uniform(
            dg["host_water_capacity_min"], dg["host_water_capacity_max"]
        )

        # Small annual growth rate (0-3%, representing debottlenecking)
        growth = rng.uniform(0.0, 0.03)

        # Small noise level for year-to-year variation
        noise_std = rng.uniform(0.0, 0.03)

        # Build capacity profile year by year
        for y in range(n_years):
            base_factor = (1.0 + growth) ** y
            noise = 1.0 + rng.normal(0.0, noise_std)
            noise = max(noise, 0.9)

            oil_cap[j, y] = oil_base * base_factor * noise
            gas_cap[j, y] = gas_base * base_factor * noise
            water_cap[j, y] = water_base * base_factor * noise

        # Enforce non-decreasing capacity
        oil_cap[j] = np.maximum.accumulate(oil_cap[j])
        gas_cap[j] = np.maximum.accumulate(gas_cap[j])
        water_cap[j] = np.maximum.accumulate(water_cap[j])

        logger.info(
            "Host_%d: oil_cap=%.2f-%.2f MSm3/y, gas_cap=%.2f-%.2f BSm3/y",
            j + 1,
            oil_cap[j, 0],
            oil_cap[j, -1],
            gas_cap[j, 0],
            gas_cap[j, -1],
        )

    return oil_cap, gas_cap, water_cap, names
