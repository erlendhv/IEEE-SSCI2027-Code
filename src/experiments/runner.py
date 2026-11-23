"""Experiment runner: runs every config variant and seed, saving full results as pickle and a summary as JSON."""

import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from src.experiments.configs import get_experiment_configs
from src.orchestrator import run_optimization

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _is_run_completed(run_dir: Path) -> bool:
    """Check if a run directory contains a completed result."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        with open(summary_path) as f:
            summary = json.load(f)
        return summary.get("status") == "completed"
    except (json.JSONDecodeError, OSError):
        return False


def run_experiment(
    experiment_id: str,
    seeds: list[int],
    config_path: str = "config/default.yaml",
    base_overrides: dict | None = None,
    results_dir: Path | None = None,
    n_workers: int | None = None,
    cache_enabled: bool = True,
    cache_persist: bool = False,
    force: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """Run all config variants × seeds for one experiment.

    Args:
        experiment_id: Experiment to run, e.g. "E1.1".
        seeds: Master seeds.
        config_path: Base YAML config.
        base_overrides: Overrides applied to all runs, e.g. fewer iterations for testing.
        results_dir: Defaults to project_root/results/.
        n_workers: Overrides pso.n_workers if given.
        cache_persist: Off by default so runs do not reuse each other's evaluations, which keeps timings comparable.
        force: Re-run even if results already exist.

    Returns:
        List of result summary dicts (one per run).
    """
    configs = get_experiment_configs(experiment_id)
    out_dir = (results_dir or RESULTS_DIR) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(configs) * len(seeds)
    summaries = []

    if verbose:
        logger.info(
            "Experiment %s: %d configs × %d seeds = %d runs",
            experiment_id, len(configs), len(seeds), total_runs,
        )

    run_number = 0
    for seed in seeds:
        for config_name, overrides in configs:
            run_number += 1
            run_id = f"{config_name}_seed{seed}"
            run_dir = out_dir / run_id

            if not force and _is_run_completed(run_dir):
                if verbose:
                    logger.info(
                        "[%d/%d] %s — %s (skipped, already completed)",
                        run_number, total_runs, experiment_id, run_id,
                    )
                with open(run_dir / "summary.json") as f:
                    summaries.append(json.load(f))
                continue

            if verbose:
                logger.info(
                    "[%d/%d] %s — %s",
                    run_number, total_runs, experiment_id, run_id,
                )

            run_overrides = dict(base_overrides or {})
            run_overrides.update(overrides)
            run_overrides["master_seed"] = seed
            run_overrides["experiment_name"] = f"{experiment_id}_{run_id}"
            run_overrides["pso.cache_enabled"] = cache_enabled
            run_overrides["pso.cache_persist"] = cache_persist
            if n_workers is not None:
                run_overrides["pso.n_workers"] = n_workers

            try:
                t_start = time.perf_counter()
                result = run_optimization(
                    config_path, overrides=run_overrides, verbose=verbose,
                )
                t_elapsed = time.perf_counter() - t_start
            except Exception:
                logger.exception("Run %s FAILED", run_id)
                summaries.append({
                    "experiment_id": experiment_id,
                    "config_name": config_name,
                    "seed": seed,
                    "run_id": run_id,
                    "status": "failed",
                })
                continue

            summary = _save_run_results(
                run_dir, result, experiment_id, config_name,
                seed, run_id, t_elapsed,
            )
            summaries.append(summary)

            if verbose:
                logger.info(
                    "  Done in %.1fs — archive: %d, pareto: %d,"
                    " evaluations: %d",
                    t_elapsed,
                    len(result.get("archive", [])),
                    len(result.get("pareto_front", [])),
                    summary.get("n_evaluations", 0),
                )

    if verbose:
        n_ok = sum(1 for s in summaries if s["status"] == "completed")
        n_fail = sum(1 for s in summaries if s["status"] == "failed")
        logger.info(
            "Experiment %s complete: %d/%d succeeded, %d failed",
            experiment_id, n_ok, total_runs, n_fail,
        )

    return summaries


def _save_run_results(
    run_dir: Path,
    result: dict,
    experiment_id: str,
    config_name: str,
    seed: int,
    run_id: str,
    elapsed_seconds: float,
) -> dict:
    """Save full results (pickle) and summary (JSON) for one run."""
    run_dir.mkdir(parents=True, exist_ok=True)

    archive = result.get("archive", [])
    pareto_front = result.get("pareto_front", [])
    history = result.get("history", {})
    config = result.get("config", {})

    n_evaluations = len(archive)

    summary = {
        "experiment_id": experiment_id,
        "config_name": config_name,
        "seed": seed,
        "run_id": run_id,
        "status": "completed",
        "elapsed_seconds": round(elapsed_seconds, 2),
        "n_archive": len(archive),
        "n_pareto_front": len(pareto_front),
        "n_iterations": config.get("pso", {}).get("max_iterations", 0),
        "n_particles": config.get("pso", {}).get("n_particles", 0),
        "n_workers": config.get("pso", {}).get("n_workers", 1),
    }

    if pareto_front:
        obj_final = np.array([
            e.get("objectives_final", e["objectives"])
            for e in pareto_front
        ])
        summary["best_rov"] = round(float(obj_final[:, 0].max()), 2)
        summary["best_cvar"] = round(float(obj_final[:, 1].max()), 2)

    pickle_path = run_dir / "results.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(result, f)

    json_path = run_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Stored separately so analysis can load the history without the full results
    if history:
        history_path = run_dir / "history.pkl"
        with open(history_path, "wb") as f:
            pickle.dump(history, f)

    return summary


def load_experiment_results(
    experiment_id: str,
    results_dir: Path | None = None,
) -> list[dict]:
    """Load the run summaries of an experiment.

    Args:
        experiment_id: Experiment to load, e.g. "E1.1".
        results_dir: Defaults to project_root/results/.

    Returns:
        One dict per run directory with a summary.json, with keys run_dir, summary and pickle_path. Full results are loaded later with load_run_result.
    """
    base_dir = (results_dir or RESULTS_DIR) / experiment_id
    if not base_dir.exists():
        return []

    runs = []
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        entry = {"run_dir": run_dir}

        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                entry["summary"] = json.load(f)
        else:
            continue

        pickle_path = run_dir / "results.pkl"
        entry["pickle_path"] = pickle_path

        runs.append(entry)

    return runs


def load_run_result(pickle_path: Path) -> dict:
    """Load full results from a pickle file."""
    with open(pickle_path, "rb") as f:
        return pickle.load(f)
