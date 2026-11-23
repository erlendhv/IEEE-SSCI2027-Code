"""
YAML config loading with dot-notation overrides, validation, and sub-seeds
derived from master_seed.
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml


REQUIRED_KEYS = [
    "experiment_name",
    "master_seed",
    "n_fields",
    "n_hosts",
    "n_years",
    "max_start_year",
    "price",
    "valuation",
    "data_generation",
    "pso",
]


def load_config(
    path: str = "config/default.yaml",
    overrides: dict[str, Any] | None = None,
) -> dict:
    """
    Load a YAML config file.

    Args:
        path: Path to the YAML file.
        overrides: Dot-notation overrides,
            e.g. {"price.n_scenarios": 500, "n_fields": 5}.

    Returns:
        Config dict with a 'seeds' sub-dict added.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required keys are missing or values are invalid.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            _set_nested(config, key, value)

    _validate(config)

    config["seeds"] = _derive_seeds(config["master_seed"])

    return config


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a nested value by dot notation: "price.n_scenarios" sets d["price"]["n_scenarios"]."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _validate(config: dict) -> None:
    """Check required top-level keys, the start-year range, and positive Schwartz-Smith parameters."""
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    max_sy = config.get("max_start_year", 0)
    n_years = config.get("n_years", 1)
    if max_sy < 0 or max_sy >= n_years:
        raise ValueError(
            f"max_start_year must be in [0, n_years-1], "
            f"got max_start_year={max_sy}, n_years={n_years}"
        )

    for commodity in ("oil", "gas"):
        params = config.get("price", {}).get(commodity, {})
        for key in ("kappa", "sigma_E", "sigma_X"):
            val = params.get(key)
            if val is not None and val <= 0:
                raise ValueError(
                    f"price.{commodity}.{key} must be positive, got {val}"
                )


def _derive_seeds(master_seed: int) -> dict[str, int]:
    """
    Derive independent per-component seeds from master_seed with SeedSequence.

    Returns:
        Dict mapping component name to integer seed.
    """
    seq = np.random.SeedSequence(master_seed)
    children = seq.spawn(4)
    component_names = ["data_generation", "price", "valuation", "pso"]

    return {
        name: int(child.generate_state(1)[0])
        for name, child in zip(component_names, children)
    }
