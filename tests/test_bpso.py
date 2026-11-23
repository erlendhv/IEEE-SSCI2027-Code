"""Tests for Standard BPSO optimizer."""

import numpy as np
import pytest

from src.models import (
    CostData,
    FieldData,
    HostData,
    PriceScenarios,
)
from src.optimizer.bpso import StandardBPSO, _sigmoid
from src.optimizer.utils import pareto_dominates


def _make_test_config(**bpso_overrides):
    """Small config for fast optimizer tests."""
    bpso_cfg = {
        "c1": 2.0,
        "c2": 2.0,
        "v_max": 6.0,
        "w_schedule": "constant",
        "w_initial": 0.9,
        "w_final": 0.9,
    }
    bpso_cfg.update(bpso_overrides)
    return {
        "n_fields": 2,
        "n_hosts": 2,
        "n_years": 5,
        "max_start_year": 4,
        "master_seed": 42,
        "seeds": {"pso": 12345},
        "pso": {
            "n_particles": 6,
            "max_iterations": 5,
            "archive_size": 10,
            "alpha": 2.0,
            "ustk_initial_pct": 1,
            "ustk_final_pct": 10,
            "feasibility_tol": 1e-6,
            "bpso": bpso_cfg,
        },
        "valuation": {
            "discount_rate": 0.08,
            "cvar_alpha": 0.05,
            "barrel": 6.29,
            "btu": 37.913,
            "nok_per_usd": 10.0,
        },
        "price": {"n_scenarios": 20, "model": "schwartz_smith"},
        "data_generation": {"area_size_km": 200, "max_wells": 8, "per_well_rate": 0.15},
    }


def _make_test_data(config):
    """Small test data matching config dimensions."""
    n_f, n_h, n_y = config["n_fields"], config["n_hosts"], config["n_years"]
    n_s = 20

    field_data = FieldData(
        oil=np.full((n_f, n_y), 1.0),
        gas=np.full((n_f, n_y), 0.1),
        water=np.full((n_f, n_y), 0.5),
        names=[f"F{i}" for i in range(n_f)],
        coords=np.zeros((n_f, 2)),
        ultimate_recovery=np.ones(n_f) * 4.0,
        complexity_factor=np.ones(n_f),
        depth=np.ones(n_f) * 2000.0,
        n_wells=np.ones(n_f, dtype=int) * 5,
        gor=np.ones(n_f) * 250.0,
        ramp_years=np.ones(n_f, dtype=int),
        decline_rate=np.ones(n_f) * 0.10,
        wc_initial=np.zeros(n_f),
        wc_final=np.ones(n_f) * 0.6,
        wc_midpoint=np.ones(n_f) * 10.0,
        wc_steepness=np.ones(n_f) * 0.3,
    )

    host_data = HostData(
        oil_capacity=np.full((n_h, n_y), 5.0),
        gas_capacity=np.full((n_h, n_y), 1.0),
        water_capacity=np.full((n_h, n_y), 3.0),
        names=[f"H{i}" for i in range(n_h)],
        coords=np.zeros((n_h, 2)),
    )

    cost_data = CostData(
        pipeline_cost=np.array([[50.0, 70.0], [60.0, 80.0]]),
        subsea_base=30.0,
        well_cost_per_well=np.array([20.0, 20.0]),
        opex_rate=np.array([10.0, 12.0]),
        variable_opex_rate=np.zeros(n_h),
        abex=np.array([20.0, 25.0]),
    )

    prices = PriceScenarios(
        oil=np.full((n_s, n_y), 70.0) + np.random.default_rng(42).normal(0, 5, (n_s, n_y)),
        gas=np.full((n_s, n_y), 3.0) + np.random.default_rng(43).normal(0, 0.3, (n_s, n_y)),
    )

    return field_data, host_data, cost_data, prices


class TestSigmoid:
    """Test the sigmoid transfer function."""

    def test_zero_gives_half(self):
        assert abs(_sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10

    def test_large_positive_near_one(self):
        assert _sigmoid(np.array([10.0]))[0] > 0.999

    def test_large_negative_near_zero(self):
        assert _sigmoid(np.array([-10.0]))[0] < 0.001

    def test_monotonic(self):
        x = np.linspace(-6, 6, 100)
        s = _sigmoid(x)
        assert np.all(np.diff(s) > 0)


class TestBPSOConstruction:
    """Test BPSO initialization."""

    def test_creates_particles(self):
        config = _make_test_config()
        fd, hd, cd, pr = _make_test_data(config)
        opt = StandardBPSO(config, fd, hd, cd, pr)
        assert len(opt.particles) == 6

    def test_particles_have_velocity(self):
        config = _make_test_config()
        fd, hd, cd, pr = _make_test_data(config)
        opt = StandardBPSO(config, fd, hd, cd, pr)
        for p in opt.particles:
            assert p.velocity is not None
            assert p.velocity.shape == (opt.n_dim,)

    def test_default_bpso_params(self):
        """BPSO uses defaults when bpso config section is missing."""
        config = _make_test_config()
        del config["pso"]["bpso"]
        fd, hd, cd, pr = _make_test_data(config)
        opt = StandardBPSO(config, fd, hd, cd, pr)
        assert opt.c1 == 2.0
        assert opt.c2 == 2.0
        assert opt.v_max == 6.0


class TestInertiaWeight:
    """Test inertia weight schedules."""

    def test_constant_weight(self):
        config = _make_test_config(w_schedule="constant", w_initial=0.9)
        fd, hd, cd, pr = _make_test_data(config)
        opt = StandardBPSO(config, fd, hd, cd, pr)
        opt._update_dynamic_parameters(0)
        assert opt.w == 0.9
        opt._update_dynamic_parameters(4)
        assert opt.w == 0.9

    def test_increasing_weight(self):
        config = _make_test_config(
            w_schedule="increasing", w_initial=0.4, w_final=0.9,
        )
        fd, hd, cd, pr = _make_test_data(config)
        opt = StandardBPSO(config, fd, hd, cd, pr)

        opt._update_dynamic_parameters(0)
        assert abs(opt.w - 0.4) < 1e-10

        opt._update_dynamic_parameters(config["pso"]["max_iterations"] - 1)
        assert abs(opt.w - 0.9) < 1e-10


class TestBPSOOptimize:
    """Integration tests for BPSO optimization."""

    def test_produces_non_empty_archive(self):
        config = _make_test_config()
        fd, hd, cd, pr = _make_test_data(config)
        opt = StandardBPSO(config, fd, hd, cd, pr)
        archive = opt.optimize(verbose=False)
        assert len(archive) > 0

    def test_archive_members_non_dominated(self):
        config = _make_test_config()
        fd, hd, cd, pr = _make_test_data(config)
        opt = StandardBPSO(config, fd, hd, cd, pr)
        archive = opt.optimize(verbose=False)

        objectives = np.array([s["objectives"] for s in archive])
        for i in range(len(archive)):
            for j in range(len(archive)):
                if i == j:
                    continue
                assert not pareto_dominates(objectives[j], objectives[i])

    def test_reproducible_with_same_seed(self):
        config = _make_test_config()
        fd, hd, cd, pr = _make_test_data(config)

        opt1 = StandardBPSO(config, fd, hd, cd, pr)
        archive1 = opt1.optimize(verbose=False)

        opt2 = StandardBPSO(config, fd, hd, cd, pr)
        archive2 = opt2.optimize(verbose=False)

        obj1 = np.array([s["objectives"] for s in archive1])
        obj2 = np.array([s["objectives"] for s in archive2])
        np.testing.assert_array_equal(obj1, obj2)

    def test_velocity_clamped(self):
        """After optimization, velocities should be within [-v_max, v_max]."""
        config = _make_test_config()
        fd, hd, cd, pr = _make_test_data(config)
        opt = StandardBPSO(config, fd, hd, cd, pr)
        opt.optimize(verbose=False)

        for p in opt.particles:
            assert np.all(p.velocity >= -opt.v_max)
            assert np.all(p.velocity <= opt.v_max)
