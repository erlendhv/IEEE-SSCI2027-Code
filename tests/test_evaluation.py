"""Tests for portfolio evaluation bridge."""

import numpy as np
import pytest

from src.models import (
    CostData,
    FieldData,
    HostData,
    PortfolioAllocation,
    PriceScenarios,
    ValuationResult,
)
from src.optimizer.evaluation import evaluate_portfolio


def _make_test_data():
    """2 fields, 2 hosts, 5 years and 10 scenarios with constant profiles and prices."""
    n_fields, n_hosts, n_years, n_scenarios = 2, 2, 5, 10

    # Oil and water in MSm3/y, gas in BSm3/y
    field_data = FieldData(
        oil=np.full((n_fields, n_years), 1.0),
        gas=np.full((n_fields, n_years), 0.1),
        water=np.full((n_fields, n_years), 0.5),
        names=["F0", "F1"],
        coords=np.zeros((n_fields, 2)),
        ultimate_recovery=np.array([4.0, 4.0]),
        complexity_factor=np.array([1.0, 1.0]),
        depth=np.array([2000.0, 2000.0]),
        n_wells=np.array([5, 5]),
        gor=np.array([250.0, 250.0]),
        ramp_years=np.array([1, 1]),
        decline_rate=np.array([0.10, 0.10]),
        wc_initial=np.zeros(n_fields),
        wc_final=np.ones(n_fields) * 0.6,
        wc_midpoint=np.ones(n_fields) * 10.0,
        wc_steepness=np.ones(n_fields) * 0.3,
    )

    # Capacity well above total production
    host_data = HostData(
        oil_capacity=np.full((n_hosts, n_years), 5.0),
        gas_capacity=np.full((n_hosts, n_years), 1.0),
        water_capacity=np.full((n_hosts, n_years), 3.0),
        names=["H0", "H1"],
        coords=np.zeros((n_hosts, 2)),
    )

    cost_data = CostData(
        pipeline_cost=np.array([[50.0, 70.0],
                                [60.0, 80.0]]),
        subsea_base=30.0,
        well_cost_per_well=np.array([20.0, 20.0]),
        opex_rate=np.array([10.0, 12.0]),
        variable_opex_rate=np.zeros(n_hosts),
        abex=np.array([20.0, 25.0]),
    )

    # Constant prices (oil USD/bbl, gas USD/MMBtu) give a deterministic NPV
    prices = PriceScenarios(
        oil=np.full((n_scenarios, n_years), 70.0),
        gas=np.full((n_scenarios, n_years), 3.0),
    )

    config = {
        "n_years": n_years,
        "valuation": {
            "discount_rate": 0.08,
            "cvar_alpha": 0.05,
            "barrel": 6.29,
            "btu": 37.913,
            "nok_per_usd": 10.0,
        },
        "data_generation": {
            "per_well_rate": 0.15,
        },
    }

    return field_data, host_data, cost_data, prices, config


class TestEvaluatePortfolio:
    """Test the portfolio evaluation bridge."""

    def test_returns_valuation_result(self):
        """evaluate_portfolio returns a ValuationResult."""
        field_data, host_data, cost_data, prices, config = _make_test_data()
        allocation = PortfolioAllocation(
            assignment=np.array([0, 1]),
            start_year=np.array([0, 0]),
            n_wells=np.ones(2, dtype=int),
        )

        result = evaluate_portfolio(
            allocation, field_data, host_data, cost_data, prices, config,
        )
        assert isinstance(result, ValuationResult)

    def test_output_shapes(self):
        """Check shapes of field_values and scenario_values."""
        field_data, host_data, cost_data, prices, config = _make_test_data()
        allocation = PortfolioAllocation(
            assignment=np.array([0, 1]),
            start_year=np.array([0, 0]),
            n_wells=np.ones(2, dtype=int),
        )

        result = evaluate_portfolio(
            allocation, field_data, host_data, cost_data, prices, config,
        )
        assert result.field_values.shape == (2,)
        assert result.scenario_values.shape == (10,)

    def test_undeveloped_field_has_zero_value(self):
        """Fields with assignment=-1 contribute zero value."""
        field_data, host_data, cost_data, prices, config = _make_test_data()
        allocation = PortfolioAllocation(
            assignment=np.array([0, -1]),
            start_year=np.array([0, 0]),
            n_wells=np.ones(2, dtype=int),
        )

        result = evaluate_portfolio(
            allocation, field_data, host_data, cost_data, prices, config,
        )
        assert result.field_values[1] == 0.0

    def test_all_undeveloped_returns_zeros(self):
        """No developed fields → all zeros."""
        field_data, host_data, cost_data, prices, config = _make_test_data()
        allocation = PortfolioAllocation(
            assignment=np.array([-1, -1]),
            start_year=np.array([0, 0]),
            n_wells=np.ones(2, dtype=int),
        )

        result = evaluate_portfolio(
            allocation, field_data, host_data, cost_data, prices, config,
        )
        assert result.expected_rov == 0.0
        assert result.cvar == 0.0
        np.testing.assert_array_equal(result.field_values, [0.0, 0.0])
        np.testing.assert_array_equal(result.scenario_values, np.zeros(10))

    def test_positive_rov_with_profitable_fields(self):
        """With high prices and low costs, ROV should be positive."""
        field_data, host_data, cost_data, prices, config = _make_test_data()
        allocation = PortfolioAllocation(
            assignment=np.array([0, 1]),
            start_year=np.array([0, 0]),
            n_wells=np.ones(2, dtype=int),
        )

        result = evaluate_portfolio(
            allocation, field_data, host_data, cost_data, prices, config,
        )
        assert result.expected_rov > 0
        assert np.all(result.scenario_values > 0)

    def test_later_start_reduces_value(self):
        """Delaying development reduces discounted value."""
        field_data, host_data, cost_data, prices, config = _make_test_data()

        alloc_early = PortfolioAllocation(
            assignment=np.array([0, -1]),
            start_year=np.array([0, 0]),
            n_wells=np.ones(2, dtype=int),
        )
        alloc_late = PortfolioAllocation(
            assignment=np.array([0, -1]),
            start_year=np.array([3, 0]),
            n_wells=np.ones(2, dtype=int),
        )

        result_early = evaluate_portfolio(
            alloc_early, field_data, host_data, cost_data, prices, config,
        )
        result_late = evaluate_portfolio(
            alloc_late, field_data, host_data, cost_data, prices, config,
        )
        # Constant prices carry no option value from waiting
        assert result_early.expected_rov > result_late.expected_rov


class TestNpvOnlyEvaluation:
    """Test the static NPV evaluation path (use_lsmc=False)."""

    def test_npv_returns_valuation_result(self):
        """NPV-only path returns a ValuationResult."""
        field_data, host_data, cost_data, prices, config = _make_test_data()
        allocation = PortfolioAllocation(
            assignment=np.array([0, 1]),
            start_year=np.array([0, 0]),
            n_wells=np.ones(2, dtype=int),
        )
        result = evaluate_portfolio(
            allocation, field_data, host_data, cost_data, prices, config,
            use_lsmc=False,
        )
        assert isinstance(result, ValuationResult)
        assert result.field_values.shape == (2,)
        assert result.scenario_values.shape == (10,)

    def test_npv_all_undeveloped_returns_zeros(self):
        """NPV-only with no developed fields returns zeros."""
        field_data, host_data, cost_data, prices, config = _make_test_data()
        allocation = PortfolioAllocation(
            assignment=np.array([-1, -1]),
            start_year=np.array([0, 0]),
            n_wells=np.ones(2, dtype=int),
        )
        result = evaluate_portfolio(
            allocation, field_data, host_data, cost_data, prices, config,
            use_lsmc=False,
        )
        assert result.expected_rov == 0.0

    def test_npv_no_go_no_go_floor(self):
        """NPV-only path can produce negative values (no floor at 0)."""
        field_data, host_data, cost_data, prices, config = _make_test_data()
        # Prices far below breakeven give a negative NPV
        low_prices = PriceScenarios(
            oil=np.full((10, 5), 1.0),
            gas=np.full((10, 5), 0.01),
        )
        allocation = PortfolioAllocation(
            assignment=np.array([0, -1]),
            start_year=np.array([0, 0]),
            n_wells=np.ones(2, dtype=int),
        )
        result = evaluate_portfolio(
            allocation, field_data, host_data, cost_data, low_prices, config,
            use_lsmc=False,
        )
        assert result.expected_rov < 0

    def test_rov_at_least_as_good_as_npv(self):
        """ROV >= NPV, since LSMC adds a deferral option and a go/no-go floor."""
        field_data, host_data, cost_data, prices, config = _make_test_data()
        allocation = PortfolioAllocation(
            assignment=np.array([0, 1]),
            start_year=np.array([0, 0]),
            n_wells=np.ones(2, dtype=int),
        )
        result_rov = evaluate_portfolio(
            allocation, field_data, host_data, cost_data, prices, config,
            use_lsmc=True,
        )
        result_npv = evaluate_portfolio(
            allocation, field_data, host_data, cost_data, prices, config,
            use_lsmc=False,
        )
        assert result_rov.expected_rov >= result_npv.expected_rov


class TestUncappedEvaluation:
    """Test the uncapped evaluation path (use_capping=False) for CDP."""

    def test_uncapped_returns_zero_violation_within_capacity(self):
        """When production is within capacity, violation is 0."""
        field_data, host_data, cost_data, prices, config = _make_test_data()
        # One field with one well stays far below host capacity
        allocation = PortfolioAllocation(
            assignment=np.array([0, -1]),
            start_year=np.array([0, 0]),
            n_wells=np.ones(2, dtype=int),
        )
        result = evaluate_portfolio(
            allocation, field_data, host_data, cost_data, prices, config,
            use_capping=False,
        )
        assert result.violation == 0.0

    def test_uncapped_returns_positive_violation_over_capacity(self):
        """When production exceeds capacity, violation > 0."""
        field_data, host_data, cost_data, prices, config = _make_test_data()
        host_data_tight = HostData(
            oil_capacity=np.full((2, 5), 0.001),
            gas_capacity=np.full((2, 5), 0.0001),
            water_capacity=np.full((2, 5), 0.001),
            names=["H0", "H1"],
            coords=np.zeros((2, 2)),
        )
        allocation = PortfolioAllocation(
            assignment=np.array([0, 0]),
            start_year=np.array([0, 0]),
            n_wells=np.array([5, 5]),
        )
        result = evaluate_portfolio(
            allocation, field_data, host_data_tight, cost_data, prices, config,
            use_capping=False,
        )
        assert result.violation > 0.0

    def test_capped_always_zero_violation(self):
        """Capped evaluation always returns violation=0."""
        field_data, host_data, cost_data, prices, config = _make_test_data()
        allocation = PortfolioAllocation(
            assignment=np.array([0, 1]),
            start_year=np.array([0, 0]),
            n_wells=np.ones(2, dtype=int),
        )
        result = evaluate_portfolio(
            allocation, field_data, host_data, cost_data, prices, config,
            use_capping=True,
        )
        assert result.violation == 0.0

    def test_uncapped_produces_valid_objectives(self):
        """Uncapped path still produces valid ROV and CVaR."""
        field_data, host_data, cost_data, prices, config = _make_test_data()
        allocation = PortfolioAllocation(
            assignment=np.array([0, 1]),
            start_year=np.array([0, 0]),
            n_wells=np.ones(2, dtype=int),
        )
        result = evaluate_portfolio(
            allocation, field_data, host_data, cost_data, prices, config,
            use_capping=False,
        )
        assert np.isfinite(result.expected_rov)
        assert np.isfinite(result.cvar)
