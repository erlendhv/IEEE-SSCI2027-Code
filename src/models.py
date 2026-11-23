"""Dataclasses shared between modules."""

from dataclasses import dataclass

import numpy as np


@dataclass
class FieldData:
    """
    Field production profiles from year 0 and reservoir attributes.

    Profiles are the production if developed with the generated n_wells.
    The shape and water-cut parameters let evaluation rebuild profiles for
    other well counts.

    Attributes:
        oil: (n_fields, n_years) MSm3/y.
        gas: (n_fields, n_years) BSm3/y.
        water: (n_fields, n_years) MSm3/y.
        names: Field identifiers.
        coords: (n_fields, 2) x, y position in km.
        ultimate_recovery: (n_fields,) recoverable volume, MSm3.
        complexity_factor: (n_fields,) per-well productivity multiplier.
        depth: (n_fields,) reservoir depth, m.
        n_wells: (n_fields,) int, number of wells.
        gor: (n_fields,) gas-oil ratio, Sm3/Sm3.
        ramp_years: (n_fields,) int, build-up duration.
        decline_rate: (n_fields,) exponential decline rate.
        wc_initial: (n_fields,) initial water cut fraction.
        wc_final: (n_fields,) maximum water cut fraction.
        wc_midpoint: (n_fields,) logistic midpoint year.
        wc_steepness: (n_fields,) logistic steepness.
    """

    oil: np.ndarray
    gas: np.ndarray
    water: np.ndarray
    names: list[str]
    coords: np.ndarray

    ultimate_recovery: np.ndarray
    complexity_factor: np.ndarray
    depth: np.ndarray
    n_wells: np.ndarray
    gor: np.ndarray

    ramp_years: np.ndarray
    decline_rate: np.ndarray

    wc_initial: np.ndarray
    wc_final: np.ndarray
    wc_midpoint: np.ndarray
    wc_steepness: np.ndarray

    def __post_init__(self) -> None:
        if not (self.oil.shape == self.gas.shape == self.water.shape):
            raise ValueError(
                f"Shape mismatch: oil={self.oil.shape}, "
                f"gas={self.gas.shape}, water={self.water.shape}"
            )
        if self.oil.ndim != 2:
            raise ValueError(f"Expected 2D arrays, got {self.oil.ndim}D")
        n_fields = self.oil.shape[0]
        if len(self.names) != n_fields:
            raise ValueError(
                f"names length ({len(self.names)}) != n_fields ({n_fields})"
            )
        if self.coords.shape != (n_fields, 2):
            raise ValueError(
                f"coords shape {self.coords.shape} != expected ({n_fields}, 2)"
            )
        for attr_name in (
            "ultimate_recovery", "complexity_factor", "depth", "n_wells", "gor",
            "ramp_years", "decline_rate",
            "wc_initial", "wc_final", "wc_midpoint", "wc_steepness",
        ):
            arr = getattr(self, attr_name)
            if arr.shape != (n_fields,):
                raise ValueError(
                    f"{attr_name} shape {arr.shape} != expected ({n_fields},)"
                )

    @property
    def n_fields(self) -> int:
        return self.oil.shape[0]

    @property
    def n_years(self) -> int:
        return self.oil.shape[1]


@dataclass
class HostData:
    """
    Spare host processing capacity available to new tiebacks, per year.

    Attributes:
        oil_capacity: (n_hosts, n_years) MSm3/y.
        gas_capacity: (n_hosts, n_years) BSm3/y.
        water_capacity: (n_hosts, n_years) MSm3/y.
        names: Host identifiers.
        coords: (n_hosts, 2) x, y position in km.
    """

    oil_capacity: np.ndarray
    gas_capacity: np.ndarray
    water_capacity: np.ndarray
    names: list[str]
    coords: np.ndarray

    def __post_init__(self) -> None:
        if not (
            self.oil_capacity.shape
            == self.gas_capacity.shape
            == self.water_capacity.shape
        ):
            raise ValueError(
                f"Shape mismatch: oil={self.oil_capacity.shape}, "
                f"gas={self.gas_capacity.shape}, water={self.water_capacity.shape}"
            )
        if self.oil_capacity.ndim != 2:
            raise ValueError(
                f"Expected 2D arrays, got {self.oil_capacity.ndim}D"
            )
        if len(self.names) != self.oil_capacity.shape[0]:
            raise ValueError(
                f"names length ({len(self.names)}) != "
                f"n_hosts ({self.oil_capacity.shape[0]})"
            )
        if self.coords.shape != (self.oil_capacity.shape[0], 2):
            raise ValueError(
                f"coords shape {self.coords.shape} != "
                f"expected ({self.oil_capacity.shape[0]}, 2)"
            )

    @property
    def n_hosts(self) -> int:
        return self.oil_capacity.shape[0]

    @property
    def n_years(self) -> int:
        return self.oil_capacity.shape[1]


@dataclass
class CostData:
    """
    Development, operating, and abandonment costs.

    CAPEX is assembled during evaluation so the optimizer can choose n_wells:
      capex = well_cost_per_well[field] × n_wells + subsea_base + pipeline_cost[field, host]

    Attributes:
        pipeline_cost: (n_fields, n_hosts) distance-dependent cost, MNOK.
        subsea_base: Cost per tieback, MNOK.
        well_cost_per_well: (n_fields,) depth-dependent cost per well, MNOK.
        opex_rate: (n_hosts,) fixed operating cost, MNOK/y.
        variable_opex_rate: (n_hosts,) variable operating cost, USD/bbl.
        abex: (n_hosts,) abandonment cost, MNOK.
    """

    pipeline_cost: np.ndarray
    subsea_base: float
    well_cost_per_well: np.ndarray

    opex_rate: np.ndarray
    variable_opex_rate: np.ndarray

    abex: np.ndarray

    def __post_init__(self) -> None:
        n_hosts = len(self.opex_rate)
        for attr_name in ("opex_rate", "variable_opex_rate", "abex"):
            arr = getattr(self, attr_name)
            if arr.shape != (n_hosts,):
                raise ValueError(
                    f"{attr_name} shape {arr.shape} != expected ({n_hosts},)"
                )
        if self.pipeline_cost.ndim != 2:
            raise ValueError(
                f"Expected 2D pipeline_cost, got {self.pipeline_cost.ndim}D"
            )
        if self.pipeline_cost.shape[1] != n_hosts:
            raise ValueError(
                f"pipeline_cost n_hosts ({self.pipeline_cost.shape[1]}) != "
                f"opex n_hosts ({n_hosts})"
            )

    def capex(self, n_wells: np.ndarray) -> np.ndarray:
        """
        Total CAPEX for the given well counts.

        Args:
            n_wells: (n_fields,) wells per field.

        Returns:
            (n_fields, n_hosts) CAPEX per field-host pair, MNOK.
        """
        drilling = self.well_cost_per_well * n_wells
        return drilling[:, np.newaxis] + self.subsea_base + self.pipeline_cost

    @property
    def n_fields(self) -> int:
        return self.pipeline_cost.shape[0]

    @property
    def n_hosts(self) -> int:
        return self.pipeline_cost.shape[1]


@dataclass
class PriceScenarios:
    """
    Price paths shared by all portfolio evaluations (sample average
    approximation). Conversion to NOK happens in the NPV module.

    Attributes:
        oil: (n_scenarios, n_years) USD/bbl.
        gas: (n_scenarios, n_years) USD/MMBtu.
    """

    oil: np.ndarray
    gas: np.ndarray

    def __post_init__(self) -> None:
        if self.oil.shape != self.gas.shape:
            raise ValueError(
                f"Shape mismatch: oil={self.oil.shape}, gas={self.gas.shape}"
            )
        if self.oil.ndim != 2:
            raise ValueError(f"Expected 2D arrays, got {self.oil.ndim}D")

    @property
    def n_scenarios(self) -> int:
        return self.oil.shape[0]

    @property
    def n_years(self) -> int:
        return self.oil.shape[1]


@dataclass
class PortfolioAllocation:
    """
    Decoded candidate solution.

    Attributes:
        assignment: (n_fields,) host index, -1 if not developed.
        start_year: (n_fields,) earliest development year.
        n_wells: (n_fields,) wells per field, 1 to max_wells.
    """

    assignment: np.ndarray
    start_year: np.ndarray
    n_wells: np.ndarray

    def __post_init__(self) -> None:
        if not (
            self.assignment.shape
            == self.start_year.shape
            == self.n_wells.shape
        ):
            raise ValueError(
                f"Shape mismatch: assignment={self.assignment.shape}, "
                f"start_year={self.start_year.shape}, "
                f"n_wells={self.n_wells.shape}"
            )
        if self.assignment.ndim != 1:
            raise ValueError(
                f"Expected 1D arrays, got {self.assignment.ndim}D"
            )

    @property
    def n_fields(self) -> int:
        return len(self.assignment)


@dataclass
class ValuationResult:
    """
    Objectives and per-field and per-scenario detail for one allocation.

    Attributes:
        expected_rov: Mean portfolio ROV across scenarios, MNOK.
        cvar: CVaR of portfolio value, MNOK.
        field_values: (n_fields,) expected ROV per field.
        scenario_values: (n_scenarios,) portfolio value per scenario.
        violation: Capacity constraint violation, 0 when capping is on.
    """

    expected_rov: float
    cvar: float
    field_values: np.ndarray
    scenario_values: np.ndarray
    violation: float = 0.0
