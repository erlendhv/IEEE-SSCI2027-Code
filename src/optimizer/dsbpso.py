"""Dynamic Sticky Binary PSO (DSBPSO), adapted from Nguyen et al. (2021).

Each bit has a stickiness instead of a velocity. Recently flipped bits
resist further changes, while stale bits become easier to flip.

Flipping probability per bit:
  p_d = i_s * (1 - stk_d) + i_p * |pb_d - x_d| + i_g * |gb_d - x_d|

where:
  i_s = stickiness importance (decreases over time)
  i_p = cognitive weight (pull toward personal best)
  i_g = social weight (pull toward archive leader)
  stk_d = stickiness of bit d (1.0 = just flipped, decays toward 0)
  pb_d, gb_d = personal best and global best at bit d

Constraint: i_s + i_p + i_g = 1, with alpha = i_p / i_g fixed.
"""

import numpy as np

from src.models import (
    CostData,
    FieldData,
    HostData,
    PriceScenarios,
)
from src.optimizer.base_pso import BaseMOPSO, Particle


class DSBPSO(BaseMOPSO):
    """Multi-objective DSBPSO with stickiness-based position updates."""

    algorithm_name = "DSBPSO"

    def __init__(
        self,
        config: dict,
        field_data: FieldData,
        host_data: HostData,
        cost_data: CostData,
        prices: PriceScenarios,
    ) -> None:
        super().__init__(config, field_data, host_data, cost_data, prices)

        pso = config["pso"]
        self.alpha = pso["alpha"]

        # Dynamic parameter bounds (Nguyen et al., 2021): i_s starts high and decays to 0
        self.i_s_initial = pso["is_initial_factor"] / self.n_dim
        self.i_s_final = 0.0

        # Unsticking period in iterations: stickiness decays by 1/ustk per iteration
        self.ustk_initial = max(
            1, int(pso["ustk_initial_pct"] * self.max_iterations / 100),
        )
        self.ustk_final = max(
            1, int(pso["ustk_final_pct"] * self.max_iterations / 100),
        )

        self.i_s = self.i_s_initial
        self.i_p = 0.0
        self.i_g = 0.0
        self.ustk = self.ustk_initial

        self.particles = [self._make_particle() for _ in range(self.n_particles)]

    # ── Particle creation ────────────────────────────────────────

    def _make_particle(self) -> Particle:
        """Create a randomly initialized particle with zero stickiness."""
        position = self.rng.integers(0, 2, self.n_dim)
        return Particle(
            position=position,
            pbest_position=position.copy(),
            pbest_objectives=np.array([-np.inf, -np.inf]),
            pbest_feasible=False,
            pbest_violation=np.inf,
            objectives=np.array([-np.inf, -np.inf]),
            feasible=False,
            violation=np.inf,
            stickiness=np.zeros(self.n_dim),
        )

    # ── Dynamic parameters ───────────────────────────────────────

    def _update_dynamic_parameters(self, iteration: int) -> None:
        """Linearly interpolate the control parameters.

        i_s decreases from i_s_initial to 0 and ustk moves from ustk_initial
        to ustk_final. i_p and i_g follow from i_s + i_p + i_g = 1 and
        alpha = i_p / i_g.
        """
        t = iteration / max(1, self.max_iterations - 1)

        self.i_s = self.i_s_initial + t * (self.i_s_final - self.i_s_initial)
        self.ustk = max(
            1,
            int(self.ustk_initial + t * (self.ustk_final - self.ustk_initial)),
        )

        self.i_g = (1.0 - self.i_s) / (self.alpha + 1.0)
        self.i_p = self.alpha * self.i_g

    # ── Position update ──────────────────────────────────────────

    def _update_particle(self, particle: Particle) -> None:
        """Flip each bit with probability p_d, then update stickiness.

        Flipped bits get stk_d = 1; the rest decay by 1/ustk, floored at 0.
        """
        gbest_position = self._select_leader()
        old_position = particle.position.copy()

        flip_prob = (
            self.i_s * (1.0 - particle.stickiness)
            + self.i_p * np.abs(particle.pbest_position - particle.position)
            + self.i_g * np.abs(gbest_position - particle.position)
        )

        flips = self.rng.random(self.n_dim) < flip_prob
        particle.position[flips] = 1 - particle.position[flips]

        changed = particle.position != old_position
        particle.stickiness[changed] = 1.0
        particle.stickiness[~changed] = np.maximum(
            0.0, particle.stickiness[~changed] - 1.0 / self.ustk,
        )
