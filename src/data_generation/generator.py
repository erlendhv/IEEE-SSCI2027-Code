"""Synthetic dataset generation: field profiles, host capacities, and costs."""

import logging

import numpy as np

from src.data_generation.costs import generate_costs
from src.data_generation.hosts import generate_host_capacities
from src.data_generation.profiles import generate_field_profiles
from src.models import CostData, FieldData, HostData

logger = logging.getLogger(__name__)


def generate_coordinates(
    n_points: int,
    area_size_km: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate uniform random 2D positions in a square area.

    Args:
        area_size_km: Side length of the square in km.

    Returns:
        (n_points, 2) x, y coordinates in km.
    """
    return rng.uniform(0, area_size_km, size=(n_points, 2))


def generate_data(
    config: dict,
) -> tuple[FieldData, HostData, CostData]:
    """
    Generate the synthetic dataset from the data_generation seed.

    Args:
        config: Full config dict, including 'seeds' from load_config.

    Returns:
        Tuple of (field_data, host_data, cost_data).
    """
    seed = config["seeds"]["data_generation"]
    rng = np.random.default_rng(seed)

    logger.info(
        "Generating data: %d fields, %d hosts, %d years (seed=%d)",
        config["n_fields"], config["n_hosts"], config["n_years"], seed,
    )

    dg = config["data_generation"]
    area_size_km = dg["area_size_km"]

    field_coords = generate_coordinates(config["n_fields"], area_size_km, rng)
    # Drawn even when overridden below, so later RNG draws do not depend
    # on host_placement
    host_coords = generate_coordinates(config["n_hosts"], area_size_km, rng)

    if dg.get("host_placement", "random") == "clustered":
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=config["n_hosts"], random_state=0, n_init=10,
        ).fit(field_coords)
        host_coords = kmeans.cluster_centers_
        logger.info("Host placement: clustered (K-means on field positions)")

    (
        oil, gas, water, field_names,
        ur, complexity_factor, depth, n_wells, gor,
        ramp_years, decline_rate,
        wc_initial, wc_final, wc_midpoint, wc_steepness,
    ) = generate_field_profiles(config, rng)

    field_data = FieldData(
        oil=oil, gas=gas, water=water, names=field_names, coords=field_coords,
        ultimate_recovery=ur,
        complexity_factor=complexity_factor,
        depth=depth,
        n_wells=n_wells,
        gor=gor,
        ramp_years=ramp_years,
        decline_rate=decline_rate,
        wc_initial=wc_initial,
        wc_final=wc_final,
        wc_midpoint=wc_midpoint,
        wc_steepness=wc_steepness,
    )

    oil_cap, gas_cap, water_cap, host_names = generate_host_capacities(config, rng)
    host_data = HostData(
        oil_capacity=oil_cap,
        gas_capacity=gas_cap,
        water_capacity=water_cap,
        names=host_names,
        coords=host_coords,
    )

    # CAPEX is assembled during evaluation from the chosen n_wells
    (
        pipeline_cost, subsea_base, well_cost_per_well,
        opex_rate, variable_opex_rate, abex,
    ) = generate_costs(config, rng, field_coords, host_coords, depth)
    cost_data = CostData(
        pipeline_cost=pipeline_cost,
        subsea_base=subsea_base,
        well_cost_per_well=well_cost_per_well,
        opex_rate=opex_rate,
        variable_opex_rate=variable_opex_rate,
        abex=abex,
    )

    logger.info("Data generation complete.")
    return field_data, host_data, cost_data
