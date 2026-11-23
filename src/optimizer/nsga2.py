"""NSGA-II (Deb et al., 2002) benchmark for the binary PSO variants.

Uses the same chromosome, evaluation pipeline, constraint handling and
evaluation budget as the PSO variants, so only the search operators differ.

Per generation:
  1. Select parents by crowded binary tournament (lower front, then larger
     crowding distance).
  2. Create offspring by uniform crossover and bit-flip mutation.
  3. Evaluate offspring.
  4. Keep the best population_size of parents + offspring, by front and then
     by crowding distance within the last front that fits.

A passive external Pareto archive, identical to the PSO archive, records
every non-dominated solution evaluated; the search never reads it. A run
yields both the standard NSGA-II result (non-dominated members of the final
population) and an archive comparable to the PSO archive.
"""

import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.models import (
    CostData,
    FieldData,
    HostData,
    PriceScenarios,
)
from src.optimizer.base_pso import BatchEvaluator
from src.optimizer.encoding import particle_length
from src.optimizer.utils import (
    crowding_distance,
    non_dominated_sort,
    update_archive,
)

logger = logging.getLogger(__name__)


class NSGA2:
    """NSGA-II with binary chromosomes.

    Population size, generations and archive capacity come from the same
    config keys as the PSO variants (pso.n_particles, pso.max_iterations,
    pso.archive_size), so all algorithms share one evaluation budget.
    """

    algorithm_name = "NSGA-II"

    def __init__(
        self,
        config: dict,
        field_data: FieldData,
        host_data: HostData,
        cost_data: CostData,
        prices: PriceScenarios,
    ) -> None:
        pso = config["pso"]
        nsga2 = pso.get("nsga2", {})

        self.n_dim = particle_length(
            config["n_fields"], config["n_hosts"],
            config["max_start_year"], config["data_generation"]["max_wells"],
        )
        self.population_size = pso["n_particles"]
        self.max_generations = pso["max_iterations"]
        self.max_archive_size = pso["archive_size"]
        self.feasibility_tol = pso.get("feasibility_tol", 1e-6)

        self.crossover_prob = nsga2.get("crossover_prob", 0.9)
        self.mutation_rate = nsga2.get("mutation_factor", 1.0) / self.n_dim

        self.rng = np.random.default_rng(config["seeds"]["pso"])

        self.evaluator = BatchEvaluator(
            config, field_data, host_data, cost_data, prices,
        )
        self.cache = self.evaluator.cache

        # Population state, one row per individual
        self.positions = self.rng.integers(
            0, 2, (self.population_size, self.n_dim),
        )
        self.objectives = np.zeros((self.population_size, 2))
        self.violation = np.zeros(self.population_size)
        self.feasible = np.zeros(self.population_size, dtype=bool)
        self.ranks = np.zeros(self.population_size, dtype=int)
        self.distances = np.zeros(self.population_size)

        # Entries are dicts with keys: position, objectives, feasible, violation
        self.archive: list[dict] = []
        self.final_population: list[dict] = []

        self.history: dict[str, list] = {
            "archive_size": [],
            "best_rov": [],
            "best_cvar": [],
            "cache_hits": [],
            "hamming_distance": [],
        }

    # ── Evaluation ───────────────────────────────────────────────

    def _evaluate(
        self,
        positions: np.ndarray,
        pool: ProcessPoolExecutor | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a batch of positions.

        Args:
            positions: Binary positions, shape (n, n_dim).
            pool: Worker pool, or None for sequential evaluation.

        Returns:
            Tuple (objectives (n, 2), violation (n,), feasible (n,)).
        """
        results = np.array(self.evaluator.evaluate(list(positions), pool))
        objectives = results[:, :2]
        violation = results[:, 2]
        feasible = violation <= self.feasibility_tol
        return objectives, violation, feasible

    # ── Ranking ──────────────────────────────────────────────────

    @staticmethod
    def _rank_and_crowd(
        objectives: np.ndarray,
        feasible: np.ndarray,
        violation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Front index and within-front crowding distance per solution.

        Args:
            objectives: Shape (n, 2).
            feasible: Shape (n,).
            violation: Shape (n,).

        Returns:
            Tuple (ranks (n,), distances (n,)).
        """
        ranks = non_dominated_sort(objectives, feasible, violation)
        distances = np.zeros(len(objectives))
        for rank in np.unique(ranks):
            front = np.flatnonzero(ranks == rank)
            distances[front] = crowding_distance(objectives[front])
        return ranks, distances

    # ── Variation ────────────────────────────────────────────────

    def _select_parents(self, n_parents: int) -> np.ndarray:
        """Crowded binary tournament selection. Returns population indices, shape (n_parents,)."""
        first = self.rng.integers(0, self.population_size, n_parents)
        second = self.rng.integers(0, self.population_size, n_parents)

        second_wins = (self.ranks[second] < self.ranks[first]) | (
            (self.ranks[second] == self.ranks[first])
            & (self.distances[second] > self.distances[first])
        )
        return np.where(second_wins, second, first)

    def _make_offspring(self) -> np.ndarray:
        """Create offspring positions, shape (population_size, n_dim)."""
        n_pairs = (self.population_size + 1) // 2
        parents = self.positions[self._select_parents(2 * n_pairs)]
        parent_a = parents[:n_pairs]
        parent_b = parents[n_pairs:]

        # Uniform crossover: in a crossed pair, each bit is swapped with probability 0.5
        crossed = self.rng.random(n_pairs) < self.crossover_prob
        swap = (self.rng.random((n_pairs, self.n_dim)) < 0.5) & crossed[:, None]
        child_a = np.where(swap, parent_b, parent_a)
        child_b = np.where(swap, parent_a, parent_b)
        offspring = np.vstack([child_a, child_b])[:self.population_size]

        flips = self.rng.random(offspring.shape) < self.mutation_rate
        return np.where(flips, 1 - offspring, offspring)

    # ── Survivor selection ───────────────────────────────────────

    def _select_survivors(
        self,
        offspring: np.ndarray,
        offspring_objectives: np.ndarray,
        offspring_violation: np.ndarray,
        offspring_feasible: np.ndarray,
    ) -> None:
        """Keep the best population_size of parents + offspring.

        Fronts are taken in order. The last front that fits only partly is
        filled by descending crowding distance.
        """
        positions = np.vstack([self.positions, offspring])
        objectives = np.vstack([self.objectives, offspring_objectives])
        violation = np.concatenate([self.violation, offspring_violation])
        feasible = np.concatenate([self.feasible, offspring_feasible])

        ranks, distances = self._rank_and_crowd(objectives, feasible, violation)

        order = np.lexsort((-distances, ranks))
        keep = order[:self.population_size]

        self.positions = positions[keep]
        self.objectives = objectives[keep]
        self.violation = violation[keep]
        self.feasible = feasible[keep]
        self.ranks = ranks[keep]
        self.distances = distances[keep]

    # ── Archive and history ──────────────────────────────────────

    def _update_archive(
        self,
        positions: np.ndarray,
        objectives: np.ndarray,
        violation: np.ndarray,
        feasible: np.ndarray,
    ) -> None:
        """Offer a batch of evaluated solutions to the external archive."""
        for i in range(len(positions)):
            self.archive = update_archive(
                self.archive, positions[i], objectives[i],
                bool(feasible[i]), float(violation[i]),
                self.max_archive_size,
            )

    def _non_dominated_population(self) -> list[dict]:
        """Non-dominated members of the current population, without duplicates."""
        front: list[dict] = []
        for i in np.flatnonzero(self.ranks == 0):
            front = update_archive(
                front, self.positions[i], self.objectives[i],
                bool(self.feasible[i]), float(self.violation[i]),
                self.population_size,
            )
        return front

    def _record_history(self) -> None:
        """Record per-generation archive metrics and population diversity."""
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

        # Average pairwise Hamming distance, same formula as BaseMOPSO
        p_d = self.positions.mean(axis=0)
        self.history["hamming_distance"].append(
            float(np.sum(2.0 * p_d * (1.0 - p_d))),
        )

    # ── Main loop ────────────────────────────────────────────────

    def optimize(
        self, verbose: bool = True, n_workers: int = 1,
    ) -> list[dict]:
        """Run the full optimization.

        Args:
            verbose: Whether to log progress.
            n_workers: Number of parallel worker processes.

        Returns:
            The external Pareto archive. The non-dominated members of the
            final population are stored in self.final_population.
        """
        if verbose:
            logger.info(
                "%s: population %d, %d generations, %d dimensions"
                ", %d workers",
                self.algorithm_name,
                self.population_size, self.max_generations, self.n_dim,
                n_workers,
            )

        pool = self.evaluator.create_pool(n_workers)

        try:
            self._run_optimization_loop(pool, verbose)
        finally:
            if pool is not None:
                pool.shutdown()
            self.cache.save()

        self.final_population = self._non_dominated_population()

        if verbose:
            logger.info(
                "%s complete: %d evaluations, %d archive members"
                ", %d in final population front",
                self.algorithm_name,
                self.evaluator.n_evaluations,
                len(self.archive),
                len(self.final_population),
            )

        return self.archive

    def _run_optimization_loop(
        self, pool: ProcessPoolExecutor | None, verbose: bool,
    ) -> None:
        """Evaluate and rank the initial population, then run the generations."""
        self.objectives, self.violation, self.feasible = self._evaluate(
            self.positions, pool,
        )
        self.ranks, self.distances = self._rank_and_crowd(
            self.objectives, self.feasible, self.violation,
        )
        self._update_archive(
            self.positions, self.objectives, self.violation, self.feasible,
        )

        for generation in range(self.max_generations):
            self.cache.reset_iteration_hits()

            offspring = self._make_offspring()
            objectives, violation, feasible = self._evaluate(offspring, pool)
            self._update_archive(offspring, objectives, violation, feasible)
            self._select_survivors(offspring, objectives, violation, feasible)

            self._record_history()

            if verbose and (
                generation % 10 == 0 or generation == self.max_generations - 1
            ):
                logger.info(
                    "  Generation %d/%d — archive: %d, best ROV: %.1f MNOK"
                    ", cache: %d/%d",
                    generation + 1, self.max_generations,
                    len(self.archive),
                    self.history["best_rov"][-1],
                    self.cache.hits_this_iter, self.population_size,
                )
