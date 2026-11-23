"""Evaluation cache keyed on decoded allocations (assignment, start year, well count).

The cache file is named by a hash of the config keys that affect
evaluation. Optimizer parameters (n_particles, max_iterations, etc.) do
not invalidate it; prices, valuation, data generation and the master seed do.
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from src.models import PortfolioAllocation

logger = logging.getLogger(__name__)

# Anchored to the project root so the location does not depend on the working directory
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"


def _evaluation_config_hash(config: dict) -> str:
    """Hash the config keys that change what evaluate_portfolio returns.

    Included: master_seed, problem dimensions, price (except
    n_scenarios_final, used only in post-optimization re-evaluation),
    valuation, data_generation, pso.use_lsmc and pso.constraint_handling.
    Seeds are derived from master_seed, and other pso keys only control
    the optimizer.

    Returns:
        16-character hex string (truncated SHA-256).
    """
    price_relevant = {
        k: v for k, v in config["price"].items()
        if k != "n_scenarios_final"
    }

    pso = config.get("pso", {})

    relevant = {
        "master_seed": config["master_seed"],
        "n_fields": config["n_fields"],
        "n_hosts": config["n_hosts"],
        "n_years": config["n_years"],
        "price": price_relevant,
        "valuation": config["valuation"],
        "data_generation": config["data_generation"],
        "use_lsmc": pso.get("use_lsmc", True),
        "constraint_handling": pso.get("constraint_handling", "capping"),
    }

    json_str = json.dumps(relevant, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def _allocation_key(
    allocation: PortfolioAllocation,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Build a hashable cache key from an allocation."""
    return (
        tuple(int(x) for x in allocation.assignment),
        tuple(int(x) for x in allocation.start_year),
        tuple(int(x) for x in allocation.n_wells),
    )


class EvaluationCache:
    """Cache of objective arrays keyed on decoded allocations.

    Modes:
      - enabled=True, persist=True: in memory, loaded from and saved to disk.
      - enabled=True, persist=False: in memory only, so runs do not share
        results and timings stay comparable.
      - enabled=False: no caching; every evaluation runs the full pipeline.

    Args:
        config: Full config. Only evaluation-relevant keys enter the hash.
        enabled: Whether to cache evaluations in memory.
        persist: Whether to load from and save to disk. Ignored when
            enabled=False.
    """

    def __init__(
        self, config: dict, enabled: bool = True, persist: bool = True,
    ) -> None:
        self._enabled = enabled
        self._persist = persist and enabled

        self._config_hash = _evaluation_config_hash(config)
        self._cache_path = CACHE_DIR / f"eval_{self._config_hash}.pkl"

        self._cache: dict[tuple, np.ndarray] = {}
        self._hits = 0
        self._misses = 0
        self._hits_this_iter = 0

        if self._persist:
            self._load()

    def _load(self) -> None:
        """Load cached entries from disk, if the file exists and is readable."""
        if not self._cache_path.exists():
            return

        try:
            with open(self._cache_path, "rb") as f:
                self._cache = pickle.load(f)
            logger.info(
                "Loaded %d cached evaluations from %s",
                len(self._cache), self._cache_path,
            )
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            logger.warning("Failed to load cache file %s: %s", self._cache_path, e)
            self._cache = {}

    def get(self, allocation: PortfolioAllocation) -> np.ndarray | None:
        """Return a copy of the cached objectives, or None if absent or disabled."""
        if not self._enabled:
            self._misses += 1
            return None

        key = _allocation_key(allocation)
        if key in self._cache:
            self._hits += 1
            self._hits_this_iter += 1
            return self._cache[key].copy()

        self._misses += 1
        return None

    def put(
        self, allocation: PortfolioAllocation, objectives: np.ndarray,
    ) -> None:
        """Store objectives for an allocation."""
        if not self._enabled:
            return

        key = _allocation_key(allocation)
        self._cache[key] = objectives.copy()

    def save(self) -> None:
        """Save the cache to disk when persist=True."""
        if not self._persist:
            return

        CACHE_DIR.mkdir(exist_ok=True)
        with open(self._cache_path, "wb") as f:
            pickle.dump(self._cache, f)
        logger.info(
            "Saved %d cached evaluations to %s",
            len(self._cache), self._cache_path,
        )

    def reset_iteration_hits(self) -> None:
        """Reset the per-iteration hit counter."""
        self._hits_this_iter = 0

    @property
    def hits_this_iter(self) -> int:
        """Number of cache hits in the current iteration."""
        return self._hits_this_iter

    @property
    def total_hits(self) -> int:
        """Cumulative cache hits across all iterations."""
        return self._hits

    @property
    def total_lookups(self) -> int:
        """Total cache lookups (hits + misses)."""
        return self._hits + self._misses

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._cache)
