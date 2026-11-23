"""Base class for multi-objective binary PSO variants.

Holds the shared evaluation, personal best, archive, leader selection and
main loop. Subclasses (DSBPSO, StandardBPSO) implement the position update
and dynamic parameter schedule.
"""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from src.models import (
    CostData,
    FieldData,
    HostData,
    PriceScenarios,
)
from src.optimizer.cache import EvaluationCache
from src.optimizer.encoding import (
    allocation_bits,
    decode_particle,
    particle_length,
    wells_bits,
    year_bits,
)
from src.optimizer.evaluation import evaluate_portfolio
from src.optimizer.utils import (
    binary_tournament_selection,
    constraint_dominates,
    crowding_distance,
    update_archive,
)

logger = logging.getLogger(__name__)


# ── Parallel evaluation support ──────────────────────────────────
# Worker function and initializer live at module level so they can be pickled.

_worker_data: dict | None = None


def _init_eval_worker(
    field_data: FieldData,
    host_data: HostData,
    cost_data: CostData,
    prices: PriceScenarios,
    config: dict,
    n_fields: int,
    n_hosts: int,
    max_start_year: int,
    max_wells: int,
    encoding: dict | None = None,
    use_lsmc: bool = True,
    use_capping: bool = True,
) -> None:
    """Initialize worker process with shared evaluation data."""
    global _worker_data
    _worker_data = {
        "field_data": field_data,
        "host_data": host_data,
        "cost_data": cost_data,
        "prices": prices,
        "config": config,
        "n_fields": n_fields,
        "n_hosts": n_hosts,
        "max_start_year": max_start_year,
        "max_wells": max_wells,
        "encoding": encoding,
        "use_lsmc": use_lsmc,
        "use_capping": use_capping,
    }


def _eval_position(position: np.ndarray) -> np.ndarray:
    """Decode a position and evaluate it. Returns [expected_rov, cvar, violation]."""
    d = _worker_data
    allocation = decode_particle(
        position, d["n_fields"], d["n_hosts"],
        d["max_start_year"], d["max_wells"],
        encoding=d["encoding"],
    )
    result = evaluate_portfolio(
        allocation, d["field_data"], d["host_data"],
        d["cost_data"], d["prices"], d["config"],
        use_lsmc=d["use_lsmc"],
        use_capping=d["use_capping"],
    )
    return np.array([result.expected_rov, result.cvar, result.violation])


# ── Batch evaluation ─────────────────────────────────────────────


class BatchEvaluator:
    """Evaluates batches of binary positions with caching and optional parallelism.

    Shared by all optimizers, so every algorithm uses the same decoding,
    valuation pipeline and cache.
    """

    def __init__(
        self,
        config: dict,
        field_data: FieldData,
        host_data: HostData,
        cost_data: CostData,
        prices: PriceScenarios,
    ) -> None:
        self.config = config
        self.field_data = field_data
        self.host_data = host_data
        self.cost_data = cost_data
        self.prices = prices

        self.n_fields = config["n_fields"]
        self.n_hosts = config["n_hosts"]
        self.max_start_year = config["max_start_year"]
        self.max_wells = config["data_generation"]["max_wells"]

        pso = config["pso"]
        self.encoding = pso.get("encoding", None)
        self.use_lsmc = pso.get("use_lsmc", True)
        self.use_capping = pso.get("constraint_handling", "capping") == "capping"

        self.cache = EvaluationCache(
            config,
            enabled=pso.get("cache_enabled", True),
            persist=pso.get("cache_persist", True),
        )
        self.n_evaluations = 0

    def create_pool(self, n_workers: int) -> ProcessPoolExecutor | None:
        """Create a worker pool, or return None when n_workers <= 1 (sequential)."""
        if n_workers <= 1:
            return None
        return ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_eval_worker,
            initargs=(
                self.field_data, self.host_data, self.cost_data,
                self.prices, self.config,
                self.n_fields, self.n_hosts,
                self.max_start_year, self.max_wells,
                self.encoding, self.use_lsmc,
                self.use_capping,
            ),
        )

    def evaluate(
        self,
        positions: list[np.ndarray],
        pool: ProcessPoolExecutor | None,
    ) -> list[np.ndarray]:
        """Evaluate positions, using cached results where available.

        Args:
            positions: Binary position arrays, each of shape (n_dim,).
            pool: Worker pool from create_pool, or None for sequential.

        Returns:
            One (3,) array [expected_rov, cvar, violation] per position.
        """
        allocations = []
        results: list[np.ndarray | None] = [None] * len(positions)
        uncached_indices: list[int] = []

        for i, position in enumerate(positions):
            alloc = decode_particle(
                position, self.n_fields, self.n_hosts,
                self.max_start_year, self.max_wells,
                encoding=self.encoding,
            )
            allocations.append(alloc)
            cached = self.cache.get(alloc)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        if uncached_indices and pool is not None:
            uncached_positions = [positions[i] for i in uncached_indices]
            evaluated = list(pool.map(_eval_position, uncached_positions))
            for idx, obj_viol in zip(uncached_indices, evaluated):
                self.n_evaluations += 1
                self.cache.put(allocations[idx], obj_viol)
                results[idx] = obj_viol.copy()
        else:
            for idx in uncached_indices:
                self.n_evaluations += 1
                result = evaluate_portfolio(
                    allocations[idx], self.field_data, self.host_data,
                    self.cost_data, self.prices, self.config,
                    use_lsmc=self.use_lsmc,
                    use_capping=self.use_capping,
                )
                obj_viol = np.array([
                    result.expected_rov, result.cvar, result.violation,
                ])
                self.cache.put(allocations[idx], obj_viol)
                results[idx] = obj_viol.copy()

        return results


# ── Particle ─────────────────────────────────────────────────────


@dataclass
class Particle:
    """One particle in a binary PSO swarm.

    Positions are binary arrays of shape (n_dim,); objectives are
    [expected_rov, cvar]. Stickiness (DSBPSO, values in [0, 1]) and
    velocity (BPSO, real-valued) are set only by the variant that uses them.
    """

    position: np.ndarray
    pbest_position: np.ndarray
    pbest_objectives: np.ndarray
    pbest_feasible: bool
    pbest_violation: float
    objectives: np.ndarray
    feasible: bool
    violation: float
    stickiness: np.ndarray | None = None
    velocity: np.ndarray | None = None


# ── Base MOPSO ───────────────────────────────────────────────────


class BaseMOPSO(ABC):
    """Abstract base class for multi-objective binary PSO.

    Subclasses implement particle creation, position updates and
    parameter schedules.
    """

    algorithm_name: str = "MOPSO"

    def __init__(
        self,
        config: dict,
        field_data: FieldData,
        host_data: HostData,
        cost_data: CostData,
        prices: PriceScenarios,
    ) -> None:
        self.n_fields = config["n_fields"]
        self.n_hosts = config["n_hosts"]
        self.n_years = config["n_years"]
        self.max_start_year = config["max_start_year"]
        self.max_wells = config["data_generation"]["max_wells"]
        self.n_dim = particle_length(
            self.n_fields, self.n_hosts, self.max_start_year, self.max_wells,
        )

        self.field_data = field_data
        self.host_data = host_data
        self.cost_data = cost_data
        self.prices = prices
        self.config = config

        pso = config["pso"]
        self.encoding = pso.get("encoding", None)
        self.use_lsmc = pso.get("use_lsmc", True)
        constraint_handling = pso.get("constraint_handling", "capping")
        self.use_capping = constraint_handling == "capping"
        self.feasibility_tol = pso.get("feasibility_tol", 1e-6)

        self.n_particles = pso["n_particles"]
        self.max_iterations = pso["max_iterations"]
        self.max_archive_size = pso["archive_size"]

        # Bit indices of each variable type across all fields, for entropy logging
        n_alloc = allocation_bits(self.n_hosts)
        n_year = year_bits(self.max_start_year)
        n_wells = wells_bits(self.max_wells)
        bits_per_field = n_alloc + n_year + n_wells
        self._assign_bit_indices = []
        self._timing_bit_indices = []
        self._wells_bit_indices = []
        for f in range(self.n_fields):
            base = f * bits_per_field
            self._assign_bit_indices.extend(range(base, base + n_alloc))
            self._timing_bit_indices.extend(
                range(base + n_alloc, base + n_alloc + n_year),
            )
            self._wells_bit_indices.extend(
                range(base + n_alloc + n_year, base + bits_per_field),
            )

        self.rng = np.random.default_rng(config["seeds"]["pso"])

        # Entries are dicts with keys: position, objectives, feasible, violation
        self.archive: list[dict] = []

        self.history: dict[str, list] = {
            "archive_size": [],
            "best_rov": [],
            "best_cvar": [],
            "cache_hits": [],
            "hamming_distance": [],
            "entropy_assignment": [],
            "entropy_timing": [],
            "entropy_wells": [],
        }

        self.evaluator = BatchEvaluator(
            config, field_data, host_data, cost_data, prices,
        )
        self.cache = self.evaluator.cache

    # ── Abstract methods ─────────────────────────────────────────

    @abstractmethod
    def _make_particle(self) -> Particle:
        """Create a randomly initialized particle."""

    @abstractmethod
    def _update_particle(self, particle: Particle) -> None:
        """Update the particle position."""

    @abstractmethod
    def _update_dynamic_parameters(self, iteration: int) -> None:
        """Update algorithm-specific parameters for this iteration."""

    # ── Evaluation ───────────────────────────────────────────────

    def _evaluate_particles_batch(
        self,
        particles: list[Particle],
        pool: ProcessPoolExecutor | None,
    ) -> None:
        """Evaluate all particles and assign objectives and feasibility."""
        objectives_list = self.evaluator.evaluate(
            [p.position for p in particles], pool,
        )

        for i, p in enumerate(particles):
            obj_viol = objectives_list[i]
            p.objectives = obj_viol[:2]
            p.violation = float(obj_viol[2])
            p.feasible = p.violation <= self.feasibility_tol

    # ── Personal best ────────────────────────────────────────────

    def _update_personal_best(self, particle: Particle) -> None:
        """Update pbest if current position dominates it (CDP)."""
        if constraint_dominates(
            particle.objectives, particle.feasible, particle.violation,
            particle.pbest_objectives, particle.pbest_feasible,
            particle.pbest_violation,
        ):
            particle.pbest_position = particle.position.copy()
            particle.pbest_objectives = particle.objectives.copy()
            particle.pbest_feasible = particle.feasible
            particle.pbest_violation = particle.violation

    # ── Archive management ───────────────────────────────────────

    def _update_archive(
        self,
        position: np.ndarray,
        objectives: np.ndarray,
        feasible: bool,
        violation: float,
    ) -> None:
        """Add a solution to the archive if it is non-dominated."""
        self.archive = update_archive(
            self.archive, position, objectives, feasible, violation,
            self.max_archive_size,
        )

    # ── Leader selection ─────────────────────────────────────────

    def _select_leader(self) -> np.ndarray:
        """Select a leader position from the archive via binary tournament."""
        if not self.archive:
            return self.rng.integers(0, 2, self.n_dim)

        objectives = np.array([s["objectives"] for s in self.archive])
        distances = crowding_distance(objectives)
        idx = binary_tournament_selection(objectives, distances, self.rng)
        return self.archive[idx]["position"]

    # ── Main loop ────────────────────────────────────────────────

    def optimize(
        self, verbose: bool = True, n_workers: int = 1,
    ) -> list[dict]:
        """Run the full optimization.

        Each iteration updates all positions, evaluates all particles
        (in parallel when n_workers > 1), then updates personal bests and
        the archive.

        Args:
            verbose: Whether to log progress.
            n_workers: Number of parallel worker processes.

        Returns:
            The final Pareto archive.
        """
        if verbose:
            logger.info(
                "%s: %d particles, %d iterations, %d dimensions"
                ", %d workers",
                self.algorithm_name,
                self.n_particles, self.max_iterations, self.n_dim, n_workers,
            )

        pool = self.evaluator.create_pool(n_workers)

        try:
            self._run_optimization_loop(pool, verbose)
        finally:
            if pool is not None:
                pool.shutdown()
            self.cache.save()

        if verbose:
            total_lookups = self.cache.total_lookups
            total_hits = self.cache.total_hits
            logger.info(
                "%s complete: %d evaluations (%d cached of %d total,"
                " %.0f%% hit rate), %d archive members",
                self.algorithm_name,
                self.evaluator.n_evaluations,
                total_hits,
                total_lookups,
                100 * total_hits / max(1, total_lookups),
                len(self.archive),
            )

        return self.archive

    def _run_optimization_loop(
        self, pool: ProcessPoolExecutor | None, verbose: bool,
    ) -> None:
        """Evaluate the initial swarm, then run the main iterations."""
        self._evaluate_particles_batch(self.particles, pool)
        for particle in self.particles:
            particle.pbest_position = particle.position.copy()
            particle.pbest_objectives = particle.objectives.copy()
            particle.pbest_feasible = particle.feasible
            particle.pbest_violation = particle.violation
            self._update_archive(
                particle.position, particle.objectives,
                particle.feasible, particle.violation,
            )

        for it in range(self.max_iterations):
            self._update_dynamic_parameters(it)
            self.cache.reset_iteration_hits()

            # Position updates stay sequential: they read the archive and share the RNG
            for particle in self.particles:
                self._update_particle(particle)

            self._evaluate_particles_batch(self.particles, pool)

            for particle in self.particles:
                self._update_personal_best(particle)
                self._update_archive(
                    particle.position, particle.objectives,
                    particle.feasible, particle.violation,
                )

            self._record_history()

            if verbose and (it % 10 == 0 or it == self.max_iterations - 1):
                hit_pct = (
                    100 * self.cache.hits_this_iter / self.n_particles
                )
                logger.info(
                    "  Iteration %d/%d — archive: %d, best ROV: %.1f MNOK"
                    ", cache: %d/%d (%.0f%%)",
                    it + 1, self.max_iterations,
                    len(self.archive),
                    self.history["best_rov"][-1],
                    self.cache.hits_this_iter, self.n_particles, hit_pct,
                )

    def _record_history(self) -> None:
        """Record per-iteration metrics including search dynamics."""
        self.history["cache_hits"].append(self.cache.hits_this_iter)
        if self.archive:
            objectives = np.array([s["objectives"] for s in self.archive])
            self.history["archive_size"].append(len(self.archive))
            self.history["best_rov"].append(float(objectives[:, 0].max()))
            self.history["best_cvar"].append(float(objectives[:, 1].max()))
        else:
            self.history["archive_size"].append(0)
            self.history["best_rov"].append(0.0)
            self.history["best_cvar"].append(0.0)

        # Average pairwise Hamming distance: with p_d the fraction of particles
        # with bit d set, bit d contributes 2 * p_d * (1 - p_d)
        positions = np.array([p.position for p in self.particles])
        p_d = positions.mean(axis=0)
        hamming = float(np.sum(2.0 * p_d * (1.0 - p_d)))
        self.history["hamming_distance"].append(hamming)

        self.history["entropy_assignment"].append(
            self._bit_entropy(p_d, self._assign_bit_indices),
        )
        self.history["entropy_timing"].append(
            self._bit_entropy(p_d, self._timing_bit_indices),
        )
        self.history["entropy_wells"].append(
            self._bit_entropy(p_d, self._wells_bit_indices),
        )

    @staticmethod
    def _bit_entropy(p_d: np.ndarray, indices: list[int]) -> float:
        """Average binary entropy H(p) = -p log2 p - (1-p) log2(1-p) over the given bits."""
        if not indices:
            return 0.0
        p = p_d[indices]
        p_safe = np.clip(p, 1e-15, 1.0 - 1e-15)
        entropy = -p_safe * np.log2(p_safe) - (1.0 - p_safe) * np.log2(1.0 - p_safe)
        return float(entropy.mean())
