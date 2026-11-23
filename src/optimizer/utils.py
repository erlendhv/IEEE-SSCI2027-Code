"""Optimizer utilities: CVaR, Pareto and constraint dominance, crowding distance, archive.

Both objectives are maximized: objective[0] is expected ROV and
objective[1] is CVaR (higher means less downside risk).
"""

import numpy as np


# ── CVaR ─────────────────────────────────────────────────────────


def calculate_cvar(values: np.ndarray, alpha: float = 0.05) -> float:
    """Conditional Value at Risk: mean of the worst alpha fraction of values.

    Args:
        values: Portfolio values per scenario, shape (n_scenarios,).
        alpha: Tail fraction (0.05 = worst 5%).
    """
    if len(values) == 0:
        return 0.0

    sorted_values = np.sort(values)
    cutoff = max(1, int(np.ceil(alpha * len(sorted_values))))
    return float(np.mean(sorted_values[:cutoff]))


# ── Pareto dominance ─────────────────────────────────────────────


def pareto_dominates(
    obj_a: np.ndarray,
    obj_b: np.ndarray,
    maximize: np.ndarray | None = None,
) -> bool:
    """Return True if A is at least as good in all objectives and strictly better in one.

    Args:
        obj_a: Shape (n_obj,).
        obj_b: Shape (n_obj,).
        maximize: Per-objective flag (True = maximize). Default: maximize all.
    """
    if maximize is None:
        maximize = np.ones(len(obj_a), dtype=bool)

    sign = np.where(maximize, 1.0, -1.0)
    a = obj_a * sign
    b = obj_b * sign

    at_least_as_good = np.all(a >= b)
    strictly_better = np.any(a > b)
    return bool(at_least_as_good and strictly_better)


# ── Non-dominated set ────────────────────────────────────────────


def get_non_dominated(
    objectives: np.ndarray,
    maximize: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask of the non-dominated solutions.

    Args:
        objectives: Shape (n_solutions, n_objectives).
        maximize: Per-objective flag. Default: maximize all.

    Returns:
        Shape (n_solutions,), True where non-dominated.
    """
    n = len(objectives)
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            if pareto_dominates(objectives[j], objectives[i], maximize):
                mask[i] = False
                break

    return mask


# ── Crowding distance ────────────────────────────────────────────


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """Crowding distance per solution, shape (n_solutions,).

    Boundary solutions (extreme on any objective) get infinite distance so
    they are never pruned from the archive.

    Args:
        objectives: Shape (n_solutions, n_objectives).
    """
    n = len(objectives)
    if n == 0:
        return np.array([])
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)

    for obj_idx in range(objectives.shape[1]):
        sorted_idx = np.argsort(objectives[:, obj_idx])

        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        obj_range = (
            objectives[sorted_idx[-1], obj_idx]
            - objectives[sorted_idx[0], obj_idx]
        )
        if obj_range == 0:
            continue

        # Interior points add the normalized gap between their neighbors
        for i in range(1, n - 1):
            idx = sorted_idx[i]
            if np.isinf(distances[idx]):
                continue

            prev_val = objectives[sorted_idx[i - 1], obj_idx]
            next_val = objectives[sorted_idx[i + 1], obj_idx]
            distances[idx] += (next_val - prev_val) / obj_range

    return distances


# ── Constraint dominance (CDP) ───────────────────────────────────


def constraint_dominates(
    obj_a: np.ndarray,
    feasible_a: bool,
    violation_a: float,
    obj_b: np.ndarray,
    feasible_b: bool,
    violation_b: float,
    maximize: np.ndarray | None = None,
) -> bool:
    """Return True if A dominates B under the Constraint Dominance Principle (Deb, 2002).

    Rules:
      1. Both feasible: Pareto dominance.
      2. One feasible: the feasible solution wins.
      3. Both infeasible: the strictly lower violation wins.

    Args:
        maximize: Per-objective flag. Default: maximize all.
    """
    if feasible_a and not feasible_b:
        return True
    if not feasible_a and feasible_b:
        return False

    if feasible_a and feasible_b:
        return pareto_dominates(obj_a, obj_b, maximize)

    return violation_a < violation_b


# ── Binary tournament selection ──────────────────────────────────


def binary_tournament_selection(
    objectives: np.ndarray,
    distances: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Return the index of a leader chosen by binary tournament.

    Of two random archive members, a dominating one wins. Otherwise the one
    with higher crowding distance wins, and ties are broken at random.

    Args:
        objectives: Shape (n_archive, n_objectives).
        distances: Crowding distances, shape (n_archive,).
    """
    n = len(objectives)
    if n == 1:
        return 0

    i, j = rng.choice(n, size=2, replace=False)

    if pareto_dominates(objectives[i], objectives[j]):
        return i
    if pareto_dominates(objectives[j], objectives[i]):
        return j

    if distances[i] > distances[j]:
        return i
    if distances[j] > distances[i]:
        return j

    return i if rng.random() < 0.5 else j


# ── Pareto archive ───────────────────────────────────────────────


def update_archive(
    archive: list[dict],
    position: np.ndarray,
    objectives: np.ndarray,
    feasible: bool,
    violation: float,
    max_size: int,
) -> list[dict]:
    """Add a solution to a Pareto archive if no member dominates it under CDP.

    Members the new solution dominates are removed. When the archive
    exceeds max_size, the most crowded members are pruned.

    Args:
        archive: Entries with keys position, objectives, feasible, violation.
        position: Shape (n_dim,).
        objectives: Shape (n_obj,).
        max_size: Archive capacity.

    Returns:
        The updated archive.
    """
    # Solutions with the same objectives as a member count as duplicates
    for entry in archive:
        if np.allclose(entry["objectives"], objectives, atol=1e-8):
            return archive

    for entry in archive:
        if constraint_dominates(
            entry["objectives"], entry["feasible"], entry["violation"],
            objectives, feasible, violation,
        ):
            return archive

    archive = [
        entry for entry in archive
        if not constraint_dominates(
            objectives, feasible, violation,
            entry["objectives"], entry["feasible"], entry["violation"],
        )
    ]

    archive.append({
        "position": position.copy(),
        "objectives": objectives.copy(),
        "feasible": feasible,
        "violation": violation,
    })

    while len(archive) > max_size:
        archive_objectives = np.array([s["objectives"] for s in archive])
        distances = crowding_distance(archive_objectives)
        archive.pop(int(np.argmin(distances)))

    return archive


# ── Non-dominated sorting (NSGA-II) ──────────────────────────────


def non_dominated_sort(
    objectives: np.ndarray,
    feasible: np.ndarray,
    violation: np.ndarray,
) -> np.ndarray:
    """Rank solutions into non-dominated fronts under CDP (Deb et al., 2002).

    Front 0 holds the solutions no other solution dominates, front 1 those
    dominated only by front 0, and so on. All objectives are maximized.

    Args:
        objectives: Shape (n_solutions, n_objectives).
        feasible: Shape (n_solutions,).
        violation: Shape (n_solutions,).

    Returns:
        Front index per solution, shape (n_solutions,), 0 = best.
    """
    n = len(objectives)

    # dominates[a, b] is True when solution a dominates solution b
    at_least_as_good = np.all(
        objectives[:, None, :] >= objectives[None, :, :], axis=2,
    )
    strictly_better = np.any(
        objectives[:, None, :] > objectives[None, :, :], axis=2,
    )
    pareto = at_least_as_good & strictly_better

    feasible_a = feasible[:, None]
    feasible_b = feasible[None, :]
    lower_violation = violation[:, None] < violation[None, :]
    dominates = (
        (feasible_a & ~feasible_b)
        | (feasible_a & feasible_b & pareto)
        | (~feasible_a & ~feasible_b & lower_violation)
    )

    n_dominators = dominates.sum(axis=0)
    ranks = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    rank = 0

    while remaining.any():
        front = remaining & (n_dominators == 0)
        ranks[front] = rank
        remaining &= ~front
        n_dominators = n_dominators - dominates[front].sum(axis=0)
        rank += 1

    return ranks
