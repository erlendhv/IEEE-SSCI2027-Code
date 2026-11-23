"""
Least-Squares Monte Carlo (LSMC) real options valuation.

Longstaff & Schwartz (2001) backward induction for the optimal exercise
year. Exercised paths record the realised payoff, not the regression
estimate.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Two regressors (oil, gas) plus intercept need at least 3 points
MIN_ITM_PATHS = 3


def _ols_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit OLS with intercept and return in-sample predictions.

    Args:
        X: (n_samples, n_features) feature matrix.
        y: (n_samples,) target vector.

    Returns:
        (n_samples,) predicted values.
    """
    X_aug = np.column_stack([X, np.ones(X.shape[0])])
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    return X_aug @ beta


def lsmc_field(
    npvs: np.ndarray,
    oil_prices: np.ndarray,
    gas_prices: np.ndarray,
    discount_rate: float,
    start_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    LSMC backward induction for one field's exercise decision.

    At each year, going backward, payoff and continuation value are
    regressed on oil and gas prices over in-the-money paths (NPV > 0).
    A path exercises if the expected payoff is positive and exceeds the
    expected continuation value.

    Args:
        npvs: (n_scenarios, n_years) NPV if exercised at year t, discounted
            to year t. Years before start_year are -inf.
        oil_prices: (n_scenarios, n_years) oil prices, regressor.
        gas_prices: (n_scenarios, n_years) gas prices, regressor.
        discount_rate: Annual discount rate.
        start_year: Earliest valid exercise year.

    Returns:
        rov: (n_scenarios,) real option value discounted to year 0,
            floored at zero (go/no-go option).
        exercise_year: (n_scenarios,) exercise year, -1 if never exercised.
    """
    n_scenarios, n_years = npvs.shape

    finite_mask = np.isfinite(npvs)
    if not finite_mask.any():
        return np.zeros(n_scenarios), np.full(n_scenarios, -1)

    last_valid_year = int(np.max(np.where(finite_mask)[1]))

    option_cf = np.zeros(n_scenarios)
    exercise_year = np.full(n_scenarios, -1)

    for t in range(last_valid_year, start_year - 1, -1):
        cont_value = option_cf * np.exp(-discount_rate)

        itm = npvs[:, t] > 0

        # Too few paths for a regression: every path continues
        if itm.sum() < MIN_ITM_PATHS:
            option_cf = cont_value
            continue

        X_itm = np.column_stack([
            oil_prices[itm, t],
            gas_prices[itm, t],
        ])

        payoff_itm = npvs[itm, t]
        cont_itm = cont_value[itm]

        E_payoff = _ols_predict(X_itm, payoff_itm)
        E_cont = _ols_predict(X_itm, cont_itm)

        exercise_mask = (E_payoff > E_cont) & (E_payoff > 0)

        option_cf = cont_value.copy()

        itm_indices = np.where(itm)[0]
        exercise_indices = itm_indices[exercise_mask]

        option_cf[exercise_indices] = npvs[exercise_indices, t]
        exercise_year[exercise_indices] = t

    rov = option_cf * np.exp(-discount_rate * start_year)

    rov = np.maximum(rov, 0.0)

    return rov, exercise_year
