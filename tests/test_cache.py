"""Tests for evaluation cache module."""

import numpy as np
import pytest

from src.models import PortfolioAllocation
from src.optimizer.cache import CACHE_DIR, EvaluationCache, _evaluation_config_hash


def _make_config():
    """Minimal config with all keys needed for hashing."""
    return {
        "master_seed": 42,
        "n_fields": 3,
        "n_hosts": 2,
        "n_years": 10,
        "experiment_name": "test",
        "price": {
            "model": "schwartz_smith",
            "n_scenarios": 200,
            "n_scenarios_final": 10000,
            "oil": {"E0": 4.26, "kappa": 0.5},
            "gas": {"E0": 4.8, "kappa": 0.9},
            "rho_oil_gas": 0.64,
        },
        "valuation": {
            "discount_rate": 0.08,
            "cvar_alpha": 0.05,
            "barrel": 6.29,
            "btu": 37.913,
            "nok_per_usd": 10.0,
        },
        "data_generation": {
            "area_size_km": 200,
            "oil_in_place_mean": 35.0,
            "well_cost_base": 400,
        },
        "pso": {
            "n_particles": 30,
            "max_iterations": 100,
            "alpha": 2.0,
        },
    }


def _make_allocation(assignment, start_year):
    """Helper to create a PortfolioAllocation."""
    assign = np.array(assignment)
    return PortfolioAllocation(
        assignment=assign,
        start_year=np.array(start_year),
        n_wells=np.ones(len(assign), dtype=int),
    )


# ── Config hashing ───────────────────────────────────────────────


class TestConfigHash:
    """Test _evaluation_config_hash correctness."""

    def test_same_config_same_hash(self):
        """Identical configs produce the same hash."""
        config = _make_config()
        assert _evaluation_config_hash(config) == _evaluation_config_hash(config)

    def test_different_master_seed(self):
        """Changing master_seed changes the hash."""
        config_a = _make_config()
        config_b = _make_config()
        config_b["master_seed"] = 99
        assert _evaluation_config_hash(config_a) != _evaluation_config_hash(config_b)

    def test_different_discount_rate(self):
        """Changing a valuation parameter changes the hash."""
        config_a = _make_config()
        config_b = _make_config()
        config_b["valuation"]["discount_rate"] = 0.10
        assert _evaluation_config_hash(config_a) != _evaluation_config_hash(config_b)

    def test_different_data_generation(self):
        """Changing a data_generation parameter changes the hash."""
        config_a = _make_config()
        config_b = _make_config()
        config_b["data_generation"]["well_cost_base"] = 500
        assert _evaluation_config_hash(config_a) != _evaluation_config_hash(config_b)

    def test_different_price_param(self):
        """Changing a price model parameter changes the hash."""
        config_a = _make_config()
        config_b = _make_config()
        config_b["price"]["oil"]["E0"] = 5.0
        assert _evaluation_config_hash(config_a) != _evaluation_config_hash(config_b)

    def test_different_n_scenarios(self):
        """Changing n_scenarios changes the hash (different price arrays)."""
        config_a = _make_config()
        config_b = _make_config()
        config_b["price"]["n_scenarios"] = 500
        assert _evaluation_config_hash(config_a) != _evaluation_config_hash(config_b)

    def test_pso_change_no_effect(self):
        """Changing PSO parameters does NOT change the hash."""
        config_a = _make_config()
        config_b = _make_config()
        config_b["pso"]["n_particles"] = 50
        config_b["pso"]["max_iterations"] = 200
        config_b["pso"]["alpha"] = 3.0
        assert _evaluation_config_hash(config_a) == _evaluation_config_hash(config_b)

    def test_experiment_name_no_effect(self):
        """Changing experiment_name does NOT change the hash."""
        config_a = _make_config()
        config_b = _make_config()
        config_b["experiment_name"] = "different_name"
        assert _evaluation_config_hash(config_a) == _evaluation_config_hash(config_b)

    def test_n_scenarios_final_no_effect(self):
        """Changing n_scenarios_final does NOT change the hash."""
        config_a = _make_config()
        config_b = _make_config()
        config_b["price"]["n_scenarios_final"] = 50000
        assert _evaluation_config_hash(config_a) == _evaluation_config_hash(config_b)

    def test_use_lsmc_changes_hash(self):
        """Changing use_lsmc changes the hash (different valuation method)."""
        config_a = _make_config()
        config_b = _make_config()
        config_b["pso"]["use_lsmc"] = False
        assert _evaluation_config_hash(config_a) != _evaluation_config_hash(config_b)

    def test_constraint_handling_changes_hash(self):
        """Changing constraint_handling changes the hash."""
        config_a = _make_config()
        config_b = _make_config()
        config_b["pso"]["constraint_handling"] = "cdp"
        assert _evaluation_config_hash(config_a) != _evaluation_config_hash(config_b)


# ── In-memory cache ──────────────────────────────────────────────


class TestCacheGetPut:
    """Test in-memory get/put operations."""

    def test_get_miss_returns_none(self):
        """Getting an uncached allocation returns None."""
        cache = EvaluationCache(_make_config())
        alloc = _make_allocation([0, 1, -1], [2, 3, 0])
        assert cache.get(alloc) is None

    def test_get_after_put(self):
        """Put then get returns the same objectives."""
        cache = EvaluationCache(_make_config())
        alloc = _make_allocation([0, 1, -1], [2, 3, 0])
        objectives = np.array([100.0, -50.0])

        cache.put(alloc, objectives)
        result = cache.get(alloc)

        np.testing.assert_array_equal(result, objectives)

    def test_get_returns_copy(self):
        """Mutating the returned array does not affect the cache."""
        cache = EvaluationCache(_make_config())
        alloc = _make_allocation([0, 1, -1], [2, 3, 0])
        cache.put(alloc, np.array([100.0, -50.0]))

        result = cache.get(alloc)
        result[0] = 999.0

        np.testing.assert_array_equal(
            cache.get(alloc), np.array([100.0, -50.0]),
        )

    def test_different_allocations_independent(self):
        """Different allocations are stored independently."""
        cache = EvaluationCache(_make_config())
        alloc_a = _make_allocation([0, 1, -1], [2, 3, 0])
        alloc_b = _make_allocation([1, 0, -1], [2, 3, 0])

        cache.put(alloc_a, np.array([100.0, -50.0]))
        cache.put(alloc_b, np.array([200.0, -30.0]))

        np.testing.assert_array_equal(
            cache.get(alloc_a), np.array([100.0, -50.0]),
        )
        np.testing.assert_array_equal(
            cache.get(alloc_b), np.array([200.0, -30.0]),
        )

    def test_size(self):
        """Size reflects number of cached entries."""
        cache = EvaluationCache(_make_config())
        assert cache.size == 0

        cache.put(
            _make_allocation([0, 1, -1], [2, 3, 0]),
            np.array([100.0, -50.0]),
        )
        assert cache.size == 1


# ── Hit/miss tracking ────────────────────────────────────────────


class TestCacheCounters:
    """Test hit/miss counter tracking."""

    def test_miss_increments(self):
        """A cache miss increments the miss count."""
        cache = EvaluationCache(_make_config())
        alloc = _make_allocation([0, 1, -1], [2, 3, 0])
        cache.get(alloc)

        assert cache.total_hits == 0
        assert cache.total_lookups == 1

    def test_hit_increments(self):
        """A cache hit increments the hit count."""
        cache = EvaluationCache(_make_config())
        alloc = _make_allocation([0, 1, -1], [2, 3, 0])
        cache.put(alloc, np.array([100.0, -50.0]))
        cache.get(alloc)

        assert cache.total_hits == 1
        assert cache.total_lookups == 1

    def test_hits_this_iter(self):
        """Per-iteration hit counter tracks within-iteration hits."""
        cache = EvaluationCache(_make_config())
        alloc = _make_allocation([0, 1, -1], [2, 3, 0])
        cache.put(alloc, np.array([100.0, -50.0]))

        cache.get(alloc)
        cache.get(alloc)
        assert cache.hits_this_iter == 2

    def test_reset_iteration_hits(self):
        """reset_iteration_hits clears per-iteration counter but not total."""
        cache = EvaluationCache(_make_config())
        alloc = _make_allocation([0, 1, -1], [2, 3, 0])
        cache.put(alloc, np.array([100.0, -50.0]))

        cache.get(alloc)
        cache.reset_iteration_hits()

        assert cache.hits_this_iter == 0
        assert cache.total_hits == 1


# ── Disk persistence ─────────────────────────────────────────────


class TestCachePersistence:
    """Test save/load roundtrip to disk."""

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        """Saved cache entries are recovered by a new cache instance."""
        monkeypatch.setattr("src.optimizer.cache.CACHE_DIR", tmp_path)

        config = _make_config()
        alloc = _make_allocation([0, 1, -1], [2, 3, 0])
        objectives = np.array([100.0, -50.0])

        cache1 = EvaluationCache(config)
        cache1.put(alloc, objectives)
        cache1.save()

        cache2 = EvaluationCache(config)
        result = cache2.get(alloc)

        np.testing.assert_array_equal(result, objectives)

    def test_different_config_different_file(self, tmp_path, monkeypatch):
        """Different configs produce separate cache files."""
        monkeypatch.setattr("src.optimizer.cache.CACHE_DIR", tmp_path)

        config_a = _make_config()
        config_b = _make_config()
        config_b["master_seed"] = 99

        alloc = _make_allocation([0, 1, -1], [2, 3, 0])

        cache_a = EvaluationCache(config_a)
        cache_a.put(alloc, np.array([100.0, -50.0]))
        cache_a.save()

        cache_b = EvaluationCache(config_b)
        assert cache_b.get(alloc) is None

    def test_load_nonexistent_is_empty(self, tmp_path, monkeypatch):
        """Loading when no cache file exists starts with empty cache."""
        monkeypatch.setattr("src.optimizer.cache.CACHE_DIR", tmp_path)

        cache = EvaluationCache(_make_config())
        assert cache.size == 0

    def test_corrupted_file_handled(self, tmp_path, monkeypatch):
        """A corrupted cache file is ignored and the cache starts empty."""
        monkeypatch.setattr("src.optimizer.cache.CACHE_DIR", tmp_path)

        config = _make_config()
        config_hash = _evaluation_config_hash(config)
        corrupt_path = tmp_path / f"eval_{config_hash}.pkl"
        corrupt_path.write_bytes(b"not a pickle")

        cache =EvaluationCache(config)
        assert cache.size == 0
