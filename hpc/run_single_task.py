#!/usr/bin/env python
"""Run one (config, seed) pair of an experiment, mapped from a Slurm array task ID.

Usage:
    python hpc/run_single_task.py E5.2 --task-id 0 --n-seeds 10
    python hpc/run_single_task.py E5.2 --task-id 42 --n-seeds 10 --workers 20
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from src.experiments.configs import get_experiment_configs
from src.orchestrator import run_optimization
from src.experiments.runner import _save_run_results

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single experiment task (for HPC array jobs).",
    )
    parser.add_argument("experiment", help="Experiment ID (e.g., E5.2)")
    parser.add_argument(
        "--task-id", type=int, required=True,
        help="Array task ID (maps to config_index * n_seeds + seed_offset)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10,
        help="Number of seeds (default: 10). Seeds are 1..n_seeds.",
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Base config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--workers", type=int, default=20,
        help="Number of parallel workers (default: 20)",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Results output directory (default: results/)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    configs = get_experiment_configs(args.experiment)
    n_configs = len(configs)
    n_seeds = args.n_seeds

    config_index = args.task_id // n_seeds
    seed_offset = args.task_id % n_seeds
    # Seeds are 1-based
    seed = seed_offset + 1

    if config_index >= n_configs:
        logger.error(
            "task-id %d exceeds max (%d configs x %d seeds = %d tasks)",
            args.task_id, n_configs, n_seeds, n_configs * n_seeds,
        )
        sys.exit(1)

    config_name, overrides = configs[config_index]
    run_id = f"{config_name}_seed{seed}"
    out_dir = (Path(args.results_dir) if args.results_dir else RESULTS_DIR) / args.experiment
    run_dir = out_dir / run_id

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        import json
        with open(summary_path) as f:
            existing = json.load(f)
        if existing.get("status") == "completed":
            logger.info("SKIP %s — already completed", run_id)
            return

    logger.info(
        "Task %d → config=%s, seed=%d (config %d/%d)",
        args.task_id, config_name, seed, config_index + 1, n_configs,
    )

    run_overrides = dict(overrides)
    run_overrides["master_seed"] = seed
    run_overrides["experiment_name"] = f"{args.experiment}_{run_id}"
    run_overrides["pso.cache_enabled"] = True
    run_overrides["pso.cache_persist"] = False
    run_overrides["pso.n_workers"] = args.workers

    try:
        t_start = time.perf_counter()
        result = run_optimization(
            args.config, overrides=run_overrides, verbose=True,
        )
        t_elapsed = time.perf_counter() - t_start
    except Exception:
        logger.exception("FAILED: %s", run_id)
        sys.exit(1)

    summary = _save_run_results(
        run_dir, result, args.experiment, config_name,
        seed, run_id, t_elapsed,
    )

    logger.info(
        "DONE %s in %.1fs — archive: %d, pareto: %d",
        run_id, t_elapsed,
        summary.get("n_archive", 0),
        summary.get("n_pareto_front", 0),
    )


if __name__ == "__main__":
    main()
