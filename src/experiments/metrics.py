"""Hypervolume Indicator (HVI) and Inverted Generational Distance (IGD), computed on saved results after optimization."""

import numpy as np

from src.optimizer.utils import get_non_dominated

# Distance of the HVI reference point below the normalised [0, 1] objective range
NORMALISED_REFERENCE_MARGIN = 0.1


def hypervolume_2d(
    objectives: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """Exact 2D hypervolume of the area dominated by the front and bounded by the reference point. Both objectives are maximized.

    Args:
        objectives: (n_solutions, 2) objective values.
        reference_point: (2,) point dominated by all solutions.

    Returns:
        Hypervolume indicator value.
    """
    if len(objectives) == 0:
        return 0.0

    nd_mask = get_non_dominated(objectives)
    front = objectives[nd_mask]

    # Only solutions at or above the reference point on both objectives contribute
    valid = np.all(front >= reference_point, axis=1)
    front = front[valid]

    if len(front) == 0:
        return 0.0

    sorted_idx = np.argsort(front[:, 0])
    front = front[sorted_idx]

    hv = 0.0
    prev_y = reference_point[1]
    for i in range(len(front)):
        x = front[i, 0] - reference_point[0]
        y = front[i, 1]
        if y > prev_y:
            if i == 0:
                hv += x * (y - reference_point[1])
            else:
                hv += (front[i, 0] - front[i - 1, 0]) * (y - reference_point[1])
            prev_y = y

    # Sweep by objective 0 descending, tracking the running max of objective 1
    front = front[np.argsort(-front[:, 0])]
    hv = 0.0
    y_max = reference_point[1]
    for i in range(len(front)):
        if front[i, 1] > y_max:
            hv += (front[i, 0] - reference_point[0]) * (front[i, 1] - y_max)
            y_max = front[i, 1]

    return float(hv)


def inverted_generational_distance(
    objectives: np.ndarray,
    reference_front: np.ndarray,
) -> float:
    """Inverted Generational Distance: mean over reference points of the Euclidean distance to the nearest solution.

    Args:
        objectives: (n_solutions, n_objectives) approximation set.
        reference_front: (n_ref, n_objectives) reference Pareto front.

    Returns:
        IGD value (lower is better).
    """
    if len(objectives) == 0:
        return float("inf")
    if len(reference_front) == 0:
        return 0.0

    min_distances = np.empty(len(reference_front))
    for i, ref in enumerate(reference_front):
        distances = np.sqrt(np.sum((objectives - ref) ** 2, axis=1))
        min_distances[i] = distances.min()

    return float(min_distances.mean())


def construct_reference_front(
    all_objectives: list[np.ndarray],
) -> np.ndarray:
    """Reference Pareto front as the non-dominated subset of all solutions from all runs, following Kaucic (2019).

    Args:
        all_objectives: List of (n_solutions, n_objectives) arrays,
            one per run across all configurations.

    Returns:
        (n_ref, n_objectives) reference front (non-dominated solutions).
    """
    combined = np.vstack(all_objectives)
    nd_mask = get_non_dominated(combined)
    return combined[nd_mask]


def per_instance_metrics(
    fronts: dict[str, dict[int, np.ndarray]],
) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]]]:
    """HVI and IGD per run, with normalised references built per seed.

    Each seed is its own problem instance, so fronts are only compared within a seed. The points of all configurations on a seed are pooled and each objective is rescaled so the pooled points span [0, 1]. The IGD reference front is the non-dominated subset of the unique pooled points. The HVI reference point lies NORMALISED_REFERENCE_MARGIN below zero on both objectives, so the largest possible HVI is (1 + margin)^2.

    Args:
        fronts: {config: {seed: (n_solutions, 2) objectives}}, both
            objectives maximized. Fronts may be empty.

    Returns:
        (hvi, igd), each {config: {seed: value}}. An empty front gets
        HVI 0 and IGD inf.
    """
    hvi = {config: {} for config in fronts}
    igd = {config: {} for config in fronts}
    reference_point = np.full(2, -NORMALISED_REFERENCE_MARGIN)
    seeds = sorted({seed for by_seed in fronts.values() for seed in by_seed})

    for seed in seeds:
        instance = {
            config: by_seed[seed]
            for config, by_seed in fronts.items() if seed in by_seed
        }
        non_empty = [front for front in instance.values() if len(front) > 0]
        if not non_empty:
            for config in instance:
                hvi[config][seed] = 0.0
                igd[config][seed] = float("inf")
            continue

        pooled = np.unique(np.vstack(non_empty), axis=0)
        low = pooled.min(axis=0)
        span = pooled.max(axis=0) - low
        # An objective with a single pooled value carries no information
        span[span == 0] = 1.0
        reference_front = construct_reference_front([(pooled - low) / span])

        for config, front in instance.items():
            scaled = (front - low) / span
            hvi[config][seed] = hypervolume_2d(scaled, reference_point)
            igd[config][seed] = inverted_generational_distance(
                scaled, reference_front
            )

    return hvi, igd
