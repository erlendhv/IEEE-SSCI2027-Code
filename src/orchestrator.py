"""
End-to-end pipeline: config, data generation, price scenarios, optimizer,
and re-evaluation of the final front on a larger scenario set.
"""

import logging
from typing import Any

import numpy as np

from src.config import load_config
from src.data_generation.generator import generate_data
from src.models import (
    CostData,
    FieldData,
    HostData,
    PortfolioAllocation,
    PriceScenarios,
)
from src.optimizer.bpso import StandardBPSO
from src.optimizer.dsbpso import DSBPSO
from src.optimizer.encoding import decode_particle
from src.optimizer.nsga2 import NSGA2
from src.optimizer.evaluation import evaluate_portfolio
from src.optimizer.utils import get_non_dominated
from src.price.schwartz_smith import (
    generate_price_scenarios,
    generate_price_scenarios_final,
)

logger = logging.getLogger(__name__)


def run_optimization(
    config_path: str = "config/default.yaml",
    overrides: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full optimization pipeline.

    Args:
        config_path: Path to the YAML config file.
        overrides: Dot-notation overrides, e.g. {"pso.n_particles": 50}.
        verbose: Log progress.

    Returns:
        Dict with keys config, archive, pareto_front, and history, plus
        population and population_front for NSGA-II. Each front entry has
        'objectives_final', the [expected_rov, cvar] on the final scenario set.
    """
    config = load_config(config_path, overrides)
    if verbose:
        logger.info(
            "Config loaded: %s (%d fields, %d hosts, %d years)",
            config["experiment_name"],
            config["n_fields"], config["n_hosts"], config["n_years"],
        )

    field_data, host_data, cost_data = generate_data(config)

    prices = generate_price_scenarios(config)
    if verbose:
        logger.info("Generated %d price scenarios for PSO", prices.n_scenarios)

    algorithm = config["pso"].get("algorithm", "dsbpso")
    if algorithm == "bpso":
        optimizer = StandardBPSO(config, field_data, host_data, cost_data, prices)
    elif algorithm == "nsga2":
        optimizer = NSGA2(config, field_data, host_data, cost_data, prices)
    else:
        optimizer = DSBPSO(config, field_data, host_data, cost_data, prices)
    n_workers = config["pso"].get("n_workers", 1)
    archive = optimizer.optimize(verbose=verbose, n_workers=n_workers)

    if verbose:
        logger.info("Optimization complete: %d archive members", len(archive))

    if not archive:
        logger.warning("Optimization produced an empty archive — no feasible solutions found")
        return {"config": config, "archive": [], "pareto_front": [], "history": optimizer.history}

    prices_final = generate_price_scenarios_final(config)
    if verbose:
        logger.info(
            "Re-evaluating archive with %d scenarios", prices_final.n_scenarios,
        )

    pareto_front = _reevaluate(
        archive, config, field_data, host_data, cost_data, prices_final,
    )

    n_removed = len(archive) - len(pareto_front)
    if n_removed > 0 and verbose:
        logger.info(
            "Removed %d dominated solutions after re-evaluation (%d → %d)",
            n_removed, len(archive), len(pareto_front),
        )

    results = {
        "config": config,
        "archive": archive,
        "pareto_front": pareto_front,
        "history": optimizer.history,
    }

    # NSGA-II also reports the front of its final population, the
    # algorithm's standard output
    if algorithm == "nsga2":
        population = optimizer.final_population
        results["population"] = population
        results["population_front"] = _reevaluate(
            population, config, field_data, host_data, cost_data, prices_final,
        )

    return results


def _reevaluate(
    entries: list[dict],
    config: dict,
    field_data: FieldData,
    host_data: HostData,
    cost_data: CostData,
    prices_final: PriceScenarios,
) -> list[dict]:
    """
    Re-evaluate solutions on the final scenario set.

    Adds 'objectives_final' ([expected_rov, cvar]) to each entry and drops
    entries that become dominated under the new objectives.

    Args:
        entries: Solutions with a binary 'position' array.
        prices_final: Price scenarios for the final evaluation.

    Returns:
        The non-dominated entries.
    """
    encoding = config["pso"].get("encoding", None)
    use_lsmc = config["pso"].get("use_lsmc", True)
    use_capping = config["pso"].get("constraint_handling", "capping") == "capping"
    for entry in entries:
        allocation = decode_particle(
            entry["position"],
            config["n_fields"], config["n_hosts"], config["max_start_year"],
            config["data_generation"]["max_wells"],
            encoding=encoding,
        )
        result = evaluate_portfolio(
            allocation, field_data, host_data, cost_data, prices_final, config,
            use_lsmc=use_lsmc,
            use_capping=use_capping,
        )
        entry["objectives_final"] = np.array([result.expected_rov, result.cvar])

    obj_final = np.array([e["objectives_final"] for e in entries])
    nd_mask = get_non_dominated(obj_final, maximize=[True, True])
    return [e for e, keep in zip(entries, nd_mask) if keep]
