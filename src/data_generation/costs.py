"""
Cost generation for field-host tiebacks.

CAPEX has three components:
  - Drilling: (well_cost_base + well_cost_depth_factor × depth) × n_wells
  - Subsea infrastructure: subsea_base per tieback
  - Pipeline: pipeline_cost_per_km2 × distance²

OPEX has a fixed part (MNOK/y) and a variable part (USD/bbl) per host.
ABEX is per host. Both are sampled uniformly from config ranges.
"""

import logging

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def generate_costs(
    config: dict,
    rng: np.random.Generator,
    field_coords: np.ndarray,
    host_coords: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate CAPEX components and operating and abandonment costs.

    CAPEX is returned as components so the optimizer can assemble it for
    the chosen n_wells:
      capex = well_cost_per_well[field] × n_wells + subsea_base + pipeline_cost[field, host]

    Args:
        config: Full config dict.
        field_coords: (n_fields, 2) field positions in km.
        host_coords: (n_hosts, 2) host positions in km.
        depth: (n_fields,) reservoir depth in m.

    Returns:
        Tuple of (pipeline_cost, subsea_base, well_cost_per_well,
        opex_rate, variable_opex_rate, abex).
    """
    n_hosts = config["n_hosts"]
    dg = config["data_generation"]

    well_cost_per_well = dg["well_cost_base"] + dg["well_cost_depth_factor"] * depth

    # (n_fields, n_hosts) distances in km
    dist_matrix = cdist(field_coords, host_coords)
    pipeline_cost = dg["pipeline_cost_per_km2"] * dist_matrix ** 2

    subsea_base = float(dg["subsea_base"])

    opex_rate = rng.uniform(dg["opex_rate_min"], dg["opex_rate_max"], size=n_hosts)
    variable_opex_rate = rng.uniform(
        dg["variable_opex_min"], dg["variable_opex_max"], size=n_hosts,
    )
    abex = rng.uniform(dg["abex_min"], dg["abex_max"], size=n_hosts)

    for j in range(n_hosts):
        logger.info(
            "Host_%d: OPEX=%.0f MNOK/y, var_OPEX=%.1f USD/bbl, ABEX=%.0f MNOK",
            j + 1, opex_rate[j], variable_opex_rate[j], abex[j],
        )

    logger.info(
        "CAPEX components: well_cost=%.0f–%.0f MNOK/well, subsea=%.0f, "
        "pipeline=%.0f–%.0f MNOK (dist=%.0f–%.0f km)",
        well_cost_per_well.min(), well_cost_per_well.max(),
        subsea_base,
        pipeline_cost.min(), pipeline_cost.max(),
        dist_matrix.min(), dist_matrix.max(),
    )

    return (
        pipeline_cost, subsea_base, well_cost_per_well,
        opex_rate, variable_opex_rate, abex,
    )
