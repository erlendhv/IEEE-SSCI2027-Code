#!/usr/bin/env python
"""CLI entry point for running experiments.

Usage:
    python run_experiment.py E1.1 --seeds 1-30
    python run_experiment.py E4.1 --seeds 42,43,44
    python run_experiment.py E3.1 --seeds 1-5 --workers 10 --no-cache
"""

import argparse
import logging
import sys

from src.experiments.runner import run_experiment


def parse_seeds(seed_str: str) -> list[int]:
    """Parse seed specification: '1-30', '42,43,44', or '1-5,10,20-25'."""
    seeds = []
    for part in seed_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run portfolio optimization experiments.",
    )
    parser.add_argument(
        "experiment",
        help="Experiment ID, e.g. E1.1 or E7.3 (see src/experiments/configs.py)",
    )
    parser.add_argument(
        "--seeds", required=True,
        help="Seed specification: '1-30', '42,43,44', or '1-5,10'",
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Base config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (overrides config)",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Results output directory (default: results/)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable evaluation caching entirely (for timing benchmarks)",
    )
    parser.add_argument(
        "--persist-cache", action="store_true",
        help="Persist cache to disk between runs (default: in-memory only)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all configs even if results already exist",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce logging output",
    )

    args = parser.parse_args()

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    seeds = parse_seeds(args.seeds)

    from pathlib import Path
    results_dir = Path(args.results_dir) if args.results_dir else None

    summaries = run_experiment(
        experiment_id=args.experiment,
        seeds=seeds,
        config_path=args.config,
        results_dir=results_dir,
        n_workers=args.workers,
        cache_enabled=not args.no_cache,
        cache_persist=args.persist_cache,
        force=args.force,
        verbose=not args.quiet,
    )

    n_ok = sum(1 for s in summaries if s["status"] == "completed")
    n_fail = sum(1 for s in summaries if s["status"] == "failed")
    print(f"\n{args.experiment}: {n_ok} completed, {n_fail} failed")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
