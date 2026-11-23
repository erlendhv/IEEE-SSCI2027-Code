"""Standard Binary PSO (BPSO) with velocity-sigmoid position update (Kennedy & Eberhart, 1997).

Position update per bit d:
  v_d = w * v_d + c1 * r1 * (pbest_d - x_d) + c2 * r2 * (gbest_d - x_d)
  v_d = clamp(v_d, -v_max, v_max)
  x_d = 1 if rand() < sigmoid(v_d) else 0

Inertia weight schedules (Liu et al., 2016):
  - "constant": w = w_initial
  - "increasing": w rises linearly from w_initial to w_final. In BPSO a
    small w explores (sigmoid near 0.5, random flipping) and a large w
    exploits (sigmoid saturates, near-deterministic updates).
"""

import numpy as np

from src.models import (
    CostData,
    FieldData,
    HostData,
    PriceScenarios,
)
from src.optimizer.base_pso import BaseMOPSO, Particle


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid transfer function, numerically stable."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class StandardBPSO(BaseMOPSO):
    """Multi-objective BPSO with a configurable inertia weight schedule."""

    algorithm_name = "BPSO"

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
        bpso = pso.get("bpso", {})
        self.c1 = bpso.get("c1", 2.0)
        self.c2 = bpso.get("c2", 2.0)
        self.v_max = bpso.get("v_max", 6.0)

        self.w_schedule = bpso.get("w_schedule", "constant")
        self.w_initial = bpso.get("w_initial", 0.9)
        self.w_final = bpso.get("w_final", 0.9)
        self.w = self.w_initial

        self.particles = [self._make_particle() for _ in range(self.n_particles)]

    # ── Particle creation ────────────────────────────────────────

    def _make_particle(self) -> Particle:
        """Create a randomly initialized particle with zero velocity."""
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
            velocity=np.zeros(self.n_dim),
        )

    # ── Dynamic parameters ───────────────────────────────────────

    def _update_dynamic_parameters(self, iteration: int) -> None:
        """Update the inertia weight from its schedule."""
        if self.w_schedule == "constant":
            self.w = self.w_initial
        else:
            t =iteration / max(1, self.max_iterations - 1)
            self.w = self.w_initial + t * (self.w_final - self.w_initial)

    # ── Position update ──────────────────────────────────────────

    def _update_particle(self, particle: Particle) -> None:
        """Velocity-sigmoid position update (see module docstring)."""
        gbest_position = self._select_leader()

        r1 = self.rng.random(self.n_dim)
        r2 = self.rng.random(self.n_dim)

        particle.velocity = (
            self.w * particle.velocity
            + self.c1 * r1 * (particle.pbest_position - particle.position)
            + self.c2 * r2 * (gbest_position - particle.position)
        )

        np.clip(particle.velocity, -self.v_max, self.v_max, out=particle.velocity)

        prob = _sigmoid(particle.velocity)
        particle.position = (self.rng.random(self.n_dim) < prob).astype(int)
