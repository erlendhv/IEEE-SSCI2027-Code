"""Tests for experiment infrastructure."""

import numpy as np
import pytest

from src.experiments.configs import get_experiment_configs
from src.experiments.metrics import (
    construct_reference_front,
    hypervolume_2d,
    inverted_generational_distance,
    per_instance_metrics,
)
from src.optimizer.cache import EvaluationCache


# ── Experiment configs ───────────────────────────────────────────


class TestExperimentConfigs:
    """Test experiment config variant definitions."""

    def test_e1_1_has_five_configs(self):
        configs = get_experiment_configs("E1.1")
        assert len(configs) == 5

    def test_e2_1_has_two_configs(self):
        configs = get_experiment_configs("E2.1")
        assert len(configs) == 2

    def test_e3_1_has_two_configs(self):
        configs = get_experiment_configs("E3.1")
        assert len(configs) == 2

    def test_e4_1_has_three_configs(self):
        configs = get_experiment_configs("E4.1")
        assert len(configs) == 3

    def test_unknown_experiment_raises(self):
        with pytest.raises(ValueError, match="Unknown experiment"):
            get_experiment_configs("E99")

    def test_configs_are_name_override_tuples(self):
        for exp_id in ["E1.1", "E2.1", "E3.1", "E4.1"]:
            configs = get_experiment_configs(exp_id)
            for name, overrides in configs:
                assert isinstance(name, str)
                assert isinstance(overrides, dict)


# ── Metrics ──────────────────────────────────────────────────────


class TestHypervolume:
    """Test 2D hypervolume indicator."""

    def test_single_point(self):
        front = np.array([[5.0, 5.0]])
        ref = np.array([0.0, 0.0])
        assert hypervolume_2d(front, ref) == 25.0

    def test_two_points(self):
        front = np.array([[10.0, 2.0], [2.0, 10.0]])
        ref = np.array([0.0, 0.0])
        # 10*2 + 2*(10-2) = 20 + 16 = 36
        assert hypervolume_2d(front, ref) == 36.0

    def test_empty_front(self):
        assert hypervolume_2d(np.empty((0, 2)), np.array([0.0, 0.0])) == 0.0

    def test_dominated_points_ignored(self):
        front = np.array([[10.0, 10.0], [5.0, 5.0]])
        ref = np.array([0.0, 0.0])
        assert hypervolume_2d(front, ref) == 100.0


class TestIGD:
    """Test inverted generational distance."""

    def test_perfect_coverage(self):
        """When approximation equals reference, IGD = 0."""
        ref = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert inverted_generational_distance(ref, ref) == 0.0

    def test_empty_approximation(self):
        ref = np.array([[1.0, 2.0]])
        assert inverted_generational_distance(np.empty((0, 2)), ref) == float("inf")

    def test_farther_is_worse(self):
        ref = np.array([[0.0, 0.0]])
        close = np.array([[1.0, 0.0]])
        far = np.array([[10.0, 0.0]])
        assert inverted_generational_distance(close, ref) < inverted_generational_distance(far, ref)


class TestReferenceFront:
    """Test reference front construction."""

    def test_combines_and_filters(self):
        # [3, 3] from the second run is dominated by [5, 5] from the first
        all_obj = [
            np.array([[10.0, 1.0], [5.0, 5.0]]),
            np.array([[1.0, 10.0], [3.0, 3.0]]),
        ]
        rf = construct_reference_front(all_obj)
        assert len(rf) == 3


class TestPerInstanceMetrics:
    """Test per-seed normalised HVI and IGD."""

    def test_scale_of_instance_does_not_matter(self):
        good = np.array([[10.0, 1.0], [5.0, 5.0], [1.0, 10.0]])
        poor = np.array([[4.0, 1.0], [1.0, 4.0]])
        fronts = {
            "a": {1: good, 2: good * 1000.0},
            "b": {1: poor, 2: poor * 1000.0},
        }
        hvi, igd = per_instance_metrics(fronts)
        for config in fronts:
            assert hvi[config][1] == pytest.approx(hvi[config][2])
            assert igd[config][1] == pytest.approx(igd[config][2])

    def test_dominating_config_is_the_reference(self):
        fronts = {
            "a": {1: np.array([[10.0, 1.0], [5.0, 5.0], [1.0, 10.0]])},
            "b": {1: np.array([[4.0, 1.0], [1.0, 4.0]])},
        }
        hvi, igd = per_instance_metrics(fronts)
        assert igd["a"][1] == pytest.approx(0.0)
        assert igd["b"][1] > 0.0
        assert hvi["a"][1] > hvi["b"][1]

    def test_single_point_filling_the_range_gets_maximum_hvi(self):
        fronts = {
            "a": {1: np.array([[10.0, 10.0]])},
            "b": {1: np.array([[0.0, 0.0]])},
        }
        hvi, _ = per_instance_metrics(fronts)
        assert hvi["a"][1] == pytest.approx(1.1 ** 2)
        assert hvi["b"][1] == pytest.approx(0.1 ** 2)

    def test_seeds_are_scored_separately_and_empty_fronts_handled(self):
        fronts = {
            "a": {1: np.array([[2.0, 2.0]]), 2: np.empty((0, 2))},
            "b": {1: np.array([[1.0, 1.0]])},
        }
        hvi, igd = per_instance_metrics(fronts)
        assert set(hvi["a"]) == {1, 2}
        assert set(hvi["b"]) == {1}
        assert hvi["a"][2] == 0.0
        assert igd["a"][2] == float("inf")


# ── Cache modes ──────────────────────────────────────────────────


def _make_cache_config():
    return {
        "master_seed": 42,
        "n_fields": 3, "n_hosts": 2, "n_years": 10,
        "price": {"n_scenarios": 10, "model": "ss"},
        "valuation": {"discount_rate": 0.08},
        "data_generation": {"area_size_km": 200},
        "pso": {},
    }


class TestCacheModes:
    """Test cache enabled/persist modes."""

    def test_disabled_cache_never_hits(self):
        from src.models import PortfolioAllocation
        config = _make_cache_config()
        cache = EvaluationCache(config, enabled=False)
        alloc = PortfolioAllocation(
            assignment=np.array([0, 1, -1]),
            start_year=np.array([0, 0, 0]),
            n_wells=np.ones(3, dtype=int),
        )
        cache.put(alloc, np.array([100.0, 50.0, 0.0]))
        assert cache.get(alloc) is None
        assert cache.size == 0

    def test_memory_only_cache_works(self):
        from src.models import PortfolioAllocation
        config = _make_cache_config()
        cache = EvaluationCache(config, enabled=True, persist=False)
        alloc = PortfolioAllocation(
            assignment=np.array([0, 1, -1]),
            start_year=np.array([0, 0, 0]),
            n_wells=np.ones(3, dtype=int),
        )
        cache.put(alloc, np.array([100.0, 50.0, 0.0]))
        result = cache.get(alloc)
        assert result is not None
        np.testing.assert_array_equal(result, [100.0, 50.0, 0.0])
