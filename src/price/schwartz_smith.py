"""
Schwartz & Smith (2000) two-factor commodity price model.

The log spot price is ln S(t) = chi(t) + xi(t), where chi is a
mean-reverting short-term deviation (Ornstein-Uhlenbeck) and xi is a
long-term level following a random walk with drift. Gas short-term shocks
are correlated with oil short-term shocks through rho_oil_gas.

Units: oil in USD/bbl, gas in USD/MMBtu.
"""

import logging

import numpy as np

from src.models import PriceScenarios

logger = logging.getLogger(__name__)


def generate_price_scenarios(config: dict) -> PriceScenarios:
    """
    Generate correlated oil and gas price scenarios.

    Args:
        config: Full config dict with 'price' and 'seeds' sections.

    Returns:
        PriceScenarios with oil (USD/bbl) and gas (USD/MMBtu).
    """
    seed = config["seeds"]["price"]
    rng = np.random.default_rng(seed)

    n_scenarios = config["price"]["n_scenarios"]
    n_years = config["n_years"]
    oil_params = config["price"]["oil"]
    gas_params = config["price"]["gas"]
    rho_oil_gas = config["price"]["rho_oil_gas"]

    oil_xi, oil_chi, oil_eps_chi = _simulate_two_factor(
        oil_params, n_scenarios, n_years, rng,
    )

    gas_xi, gas_chi = _simulate_two_factor_correlated(
        gas_params, rho_oil_gas, oil_eps_chi, n_scenarios, n_years, rng,
    )

    oil_prices = _factors_to_prices(oil_xi, oil_chi, oil_params, n_years)
    gas_prices = _factors_to_prices(gas_xi, gas_chi, gas_params, n_years)

    # The gas model is calibrated in different units than USD/MMBtu
    gas_unit_conv = gas_params.get("unit_conversion", 1.0)
    gas_prices = gas_prices * gas_unit_conv

    logger.info(
        "Generated %d SS price scenarios (%d years). "
        "Oil: %.1f → %.1f USD/bbl (mean). Gas: %.2f → %.2f USD/MMBtu (mean).",
        n_scenarios, n_years,
        oil_prices[:, 0].mean(), oil_prices[:, -1].mean(),
        gas_prices[:, 0].mean(), gas_prices[:, -1].mean(),
    )

    return PriceScenarios(oil=oil_prices, gas=gas_prices)


def generate_price_scenarios_final(config: dict) -> PriceScenarios:
    """
    Generate a larger, independent scenario set for re-evaluating the front.

    Uses n_scenarios_final and a different seed than the optimization paths.
    """
    modified_config = {
        **config,
        "seeds": {**config["seeds"], "price": config["seeds"]["price"] + 1},
        "price": {
            **config["price"],
            "n_scenarios": config["price"]["n_scenarios_final"],
        },
    }
    return generate_price_scenarios(modified_config)


# ── Two-factor simulation ────────────────────────────────────────


def _simulate_two_factor(
    params: dict,
    n_scenarios: int,
    n_years: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the two Schwartz-Smith latent factors.

    Args:
        params: Schwartz-Smith parameters (E0, X0, sigma_E, sigma_X,
            lambda_X, mu_star, kappa, rho).
        n_scenarios: Number of Monte Carlo paths.
        n_years: Number of annual time steps.

    Returns:
        (xi, chi, eps_chi): long-term factor, short-term factor, and
        short-term shocks, each (n_scenarios, n_years).
    """
    kappa = params["kappa"]
    sigma_X = params["sigma_X"]
    sigma_E = params["sigma_E"]
    lambda_X = params["lambda_X"]
    mu_star = params["mu_star"]
    rho = params["rho"]

    xi = np.zeros((n_scenarios, n_years))
    chi = np.zeros((n_scenarios, n_years))
    # Short-term shocks are returned so gas can be correlated with them
    eps_chi = np.zeros((n_scenarios, n_years))

    xi[:, 0] = params["E0"]
    chi[:, 0] = params["X0"]

    # Exact discretisation constants for dt = 1
    exp_neg_kappa = np.exp(-kappa)
    chi_std = sigma_X * np.sqrt((1 - np.exp(-2 * kappa)) / (2 * kappa))
    drift_adj = (1 - exp_neg_kappa) * lambda_X / kappa

    for t in range(1, n_years):
        eps_chi[:, t] = rng.standard_normal(n_scenarios)
        chi[:, t] = (
            chi[:, t - 1] * exp_neg_kappa
            - drift_adj
            + chi_std * eps_chi[:, t]
        )

        eps_xi_indep = rng.standard_normal(n_scenarios)
        eps_xi = rho * eps_chi[:, t] + np.sqrt(1 - rho ** 2) * eps_xi_indep
        xi[:, t] = xi[:, t - 1] + mu_star + sigma_E * eps_xi

    return xi, chi, eps_chi


def _simulate_two_factor_correlated(
    params: dict,
    rho_cross: float,
    other_eps_chi: np.ndarray,
    n_scenarios: int,
    n_years: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the two factors with short-term shocks correlated to another
    commodity's short-term shocks.

    Args:
        params: Schwartz-Smith parameters for this commodity.
        rho_cross: Correlation with the other commodity's short-term shocks.
        other_eps_chi: (n_scenarios, n_years) other commodity's short-term shocks.

    Returns:
        (xi, chi): long-term and short-term factors, each (n_scenarios, n_years).
    """
    kappa = params["kappa"]
    sigma_X = params["sigma_X"]
    sigma_E = params["sigma_E"]
    lambda_X = params["lambda_X"]
    mu_star = params["mu_star"]
    rho = params["rho"]

    xi = np.zeros((n_scenarios, n_years))
    chi = np.zeros((n_scenarios, n_years))

    xi[:, 0] = params["E0"]
    chi[:, 0] = params["X0"]

    exp_neg_kappa = np.exp(-kappa)
    chi_std = sigma_X * np.sqrt((1 - np.exp(-2 * kappa)) / (2 * kappa))
    drift_adj = (1 - exp_neg_kappa) * lambda_X / kappa

    for t in range(1, n_years):
        eps_chi = (
            rho_cross * other_eps_chi[:, t]
            + np.sqrt(1 - rho_cross ** 2) * rng.standard_normal(n_scenarios)
        )
        chi[:, t] = chi[:, t - 1] * exp_neg_kappa - drift_adj + chi_std * eps_chi

        eps_xi = rho * eps_chi + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_scenarios)
        xi[:, t] = xi[:, t - 1] + mu_star + sigma_E * eps_xi

    return xi, chi


# ── Factor-to-price conversion ───────────────────────────────────


def _factors_to_prices(
    xi: np.ndarray,
    chi: np.ndarray,
    params: dict,
    n_years: int,
) -> np.ndarray:
    """
    Convert latent factors to prices with the Schwartz & Smith (2000)
    futures price formula, eq. 12:
    ln F(t,T) = chi(t) * exp(-kappa * (T - t)) + xi(t) + A(T - t).

    Args:
        xi: (n_scenarios, n_years) long-term factor.
        chi: (n_scenarios, n_years) short-term factor.
        params: Schwartz-Smith parameters.

    Returns:
        (n_scenarios, n_years) price paths in the model's calibration units.
    """
    kappa = params["kappa"]
    sigma_X = params["sigma_X"]
    sigma_E = params["sigma_E"]
    lambda_X = params["lambda_X"]
    mu_star = params["mu_star"]
    rho = params["rho"]

    t = np.arange(n_years, dtype=float)
    exp_neg_kappa_t = np.exp(-kappa * t)

    # Jensen's inequality correction for the lognormal mean
    variance_correction = 0.5 * (
        (1 - np.exp(-2 * kappa * t)) * sigma_X ** 2 / (2 * kappa)
        + sigma_E ** 2 * t
        + 2 * (1 - exp_neg_kappa_t) * rho * sigma_X * sigma_E / kappa
    )

    risk_adj = mu_star * t - (1 - exp_neg_kappa_t) * lambda_X / kappa

    ln_price = (
        chi * exp_neg_kappa_t[np.newaxis, :]
        + xi
        + risk_adj[np.newaxis, :]
        + variance_correction[np.newaxis, :]
    )

    return np.exp(ln_price)
