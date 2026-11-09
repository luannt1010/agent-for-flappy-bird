"""Linear difficulty curriculum, driven by training steps rather than test data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .benchmark_env import PhysicsConfig


@dataclass(frozen=True)
class Curriculum:
    """Interpolate easy episode physics toward baseline or an explicit hard end.

    ``alpha = clip(progress,0,1)`` and ``p(alpha) = (1-alpha)*p_easy + alpha*p_end``
    for every physics parameter. Callers supply ``progress=steps/budget`` at
    reset, so physics does not change during a transition. By default the final
    endpoint is the environment's base configuration, exactly baseline at 1.
    """

    easy: PhysicsConfig = field(default_factory=lambda: PhysicsConfig(
        pipe_gap=160.0, pipe_speed=-3.0, gravity=0.8,
        flap_velocity=-8.0, pipe_spacing=180.0))
    hard: PhysicsConfig | None = None

    def at(self, progress: float, base: PhysicsConfig) -> PhysicsConfig:
        """Return interpolated config; finite scalar progress is clamped to [0,1].

        Inputs: training completion fraction and the environment's base physics.
        Output: validated immutable episode PhysicsConfig. ``hard=None`` uses
        ``base`` as the end, allowing custom baseline physics without drift.
        """
        if not np.isfinite(progress):
            raise ValueError("progress must be finite.")
        alpha = float(np.clip(progress, 0.0, 1.0))
        start, end = asdict(self.easy), asdict(self.hard or base)
        return PhysicsConfig(**{key: (1 - alpha) * value + alpha * end[key]
                                for key, value in start.items()})

