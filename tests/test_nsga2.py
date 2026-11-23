"""Tests for the NSGA-II optimizer and its sorting and archive utilities."""

import numpy as np

from src.optimizer.nsga2 import NSGA2
from src.optimizer.utils import (
    non_dominated_sort,
    pareto_dominates,
    update_archive,
)
from test_bpso import _make_test_config, _make_test_data


def _make_optimizer(**nsga2_overrides) -> NSGA2:
    """Small NSGA-II instance on the shared optimizer test data."""
    config = _make_test_config()
    config["pso"]["nsga2"] = nsga2_overrides
    fd, hd, cd, pr = _make_test_data(config)
    return NSGA2(config, fd, hd, cd, pr)


class TestNonDominatedSort:
    """Test front ranking under CDP."""

    def test_fronts_ordered_by_dominance(self):
        objectives = np.array([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
        feasible = np.ones(3, dtype=bool)
        violation = np.zeros(3)
        ranks = non_dominated_sort(objectives, feasible, violation)
        np.testing.assert_array_equal(ranks, [0, 1, 2])

    def test_mutually_non_dominated_share_front(self):
        objectives = np.array([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]])
        feasible = np.ones(3, dtype=bool)
        violation = np.zeros(3)
        ranks = non_dominated_sort(objectives, feasible, violation)
        np.testing.assert_array_equal(ranks, [0, 0, 0])

    def test_feasible_ranks_before_infeasible(self):
        objectives = np.array([[9.0, 9.0], [1.0, 1.0]])
        feasible = np.array([False, True])
        violation = np.array([0.5, 0.0])
        ranks = non_dominated_sort(objectives, feasible, violation)
        np.testing.assert_array_equal(ranks, [1, 0])

    def test_infeasible_ranked_by_violation(self):
        objectives = np.array([[1.0, 1.0], [9.0, 9.0], [5.0, 5.0]])
        feasible = np.zeros(3, dtype=bool)
        violation = np.array([0.1, 0.3, 0.2])
        ranks = non_dominated_sort(objectives, feasible, violation)
        np.testing.assert_array_equal(ranks, [0, 2, 1])


class TestUpdateArchive:
    """Test the shared Pareto archive update."""

    def test_dominated_solution_rejected(self):
        position = np.array([0, 1])
        archive = update_archive([], position, np.array([2.0, 2.0]), True, 0.0, 10)
        archive = update_archive(archive, position, np.array([1.0, 1.0]), True, 0.0, 10)
        assert len(archive) == 1
        np.testing.assert_array_equal(archive[0]["objectives"], [2.0, 2.0])

    def test_dominating_solution_replaces_members(self):
        position = np.array([0, 1])
        archive = update_archive([], position, np.array([1.0, 2.0]), True, 0.0, 10)
        archive = update_archive(archive, position, np.array([2.0, 1.0]), True, 0.0, 10)
        archive = update_archive(archive, position, np.array([3.0, 3.0]), True, 0.0, 10)
        assert len(archive) == 1
        np.testing.assert_array_equal(archive[0]["objectives"], [3.0, 3.0])

    def test_pruned_to_capacity_keeping_extremes(self):
        position = np.array([0, 1])
        archive: list[dict] = []
        for x in [1.0, 2.0, 2.1, 3.0, 4.0]:
            archive = update_archive(
                archive, position, np.array([x, 5.0 - x]), True, 0.0, 4,
            )
        assert len(archive) == 4
        first_objective = sorted(e["objectives"][0] for e in archive)
        assert first_objective[0] == 1.0
        assert first_objective[-1] == 4.0


class TestNSGA2Construction:
    """Test NSGA-II initialization."""

    def test_population_shape(self):
        opt = _make_optimizer()
        assert opt.positions.shape == (6, opt.n_dim)
        assert set(np.unique(opt.positions)) <= {0, 1}

    def test_default_operator_params(self):
        opt = _make_optimizer()
        assert opt.crossover_prob == 0.9
        assert abs(opt.mutation_rate - 1.0 / opt.n_dim) < 1e-12

    def test_operator_params_from_config(self):
        opt = _make_optimizer(crossover_prob=0.5, mutation_factor=2.0)
        assert opt.crossover_prob == 0.5
        assert abs(opt.mutation_rate - 2.0 / opt.n_dim) < 1e-12


class TestNSGA2Variation:
    """Test crossover and mutation."""

    def test_offspring_shape_and_binary(self):
        opt = _make_optimizer()
        offspring = opt._make_offspring()
        assert offspring.shape == (6, opt.n_dim)
        assert set(np.unique(offspring)) <= {0, 1}

    def test_no_variation_copies_parents(self):
        opt = _make_optimizer(crossover_prob=0.0, mutation_factor=0.0)
        offspring = opt._make_offspring()
        parents = {tuple(p) for p in opt.positions}
        for child in offspring:
            assert tuple(child) in parents

    def test_odd_population_size(self):
        config = _make_test_config()
        config["pso"]["n_particles"] = 5
        fd, hd, cd, pr = _make_test_data(config)
        opt = NSGA2(config, fd, hd, cd, pr)
        assert opt._make_offspring().shape == (5, opt.n_dim)


class TestNSGA2Optimize:
    """Integration tests for NSGA-II optimization."""

    def test_produces_archive_and_final_population(self):
        opt = _make_optimizer()
        archive = opt.optimize(verbose=False)
        assert len(archive) > 0
        assert len(opt.final_population) > 0
        assert opt.positions.shape == (6, opt.n_dim)

    def test_archive_and_population_non_dominated(self):
        opt = _make_optimizer()
        archive = opt.optimize(verbose=False)

        for solutions in (archive, opt.final_population):
            objectives = np.array([s["objectives"] for s in solutions])
            for i in range(len(solutions)):
                for j in range(len(solutions)):
                    if i == j:
                        continue
                    assert not pareto_dominates(objectives[j], objectives[i])

    def test_archive_covers_final_population(self):
        """No final-population solution may dominate an archive member."""
        opt = _make_optimizer()
        archive = opt.optimize(verbose=False)

        for member in opt.final_population:
            for entry in archive:
                assert not pareto_dominates(
                    member["objectives"], entry["objectives"],
                )

    def test_reproducible_with_same_seed(self):
        archive1 = _make_optimizer().optimize(verbose=False)
        archive2 = _make_optimizer().optimize(verbose=False)

        obj1 = np.array([s["objectives"] for s in archive1])
        obj2 = np.array([s["objectives"] for s in archive2])
        np.testing.assert_array_equal(obj1, obj2)

    def test_history_recorded_per_generation(self):
        opt = _make_optimizer()
        opt.optimize(verbose=False)
        for values in opt.history.values():
            assert len(values) == 5
