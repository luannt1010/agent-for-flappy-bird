"""Per-episode domain randomization for physics-robust Flappy Bird policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .benchmark_env import PhysicsConfig


@dataclass(frozen=True)
class DomainRandomization:
    """Sample independent uniform physics parameters at each episode reset.

    For each supplied interval ``[a,b]``, ``parameter ~ Uniform(a,b)``; a
    ``None`` interval preserves the supplied base value. Defaults vary gap
    85-135px, speed -5 to -3px/step, gravity 0.7-1.3px/step², spacing
    130-180px, and vertical wind -0.15 to +0.15px/step². Flap strength is
    preserved unless explicitly randomized. Physics remains fixed within an
    episode, making the sampled transition process stationary per episode.
    """

    pipe_gap: tuple[float, float] | None = (85.0, 135.0)
    pipe_speed: tuple[float, float] | None = (-5.0, -3.0)
    gravity: tuple[float, float] | None = (0.7, 1.3)
    flap_velocity: tuple[float, float] | None = None
    wind: tuple[float, float] | None = (-0.15, 0.15)
    pipe_spacing: tuple[float, float] | None = (130.0, 180.0)

    def __post_init__(self) -> None:
        """Validate interval order and every possible physics-range endpoint."""
        base = PhysicsConfig()
        lower, upper = {}, {}
        for name, bounds in asdict(self).items():
            if bounds is None:
                continue
            if len(bounds) != 2 or not all(np.isfinite(item) for item in bounds):
                raise ValueError(f"{name} must be a pair of finite bounds or None.")
            if bounds[0] > bounds[1]:
                raise ValueError(f"{name} lower bound must not exceed upper bound.")
            lower[name], upper[name] = map(float, bounds)
        replace(base, **lower)
        replace(base, **upper)
        gravity_low = self.gravity[0] if self.gravity is not None else base.gravity
        wind_low = self.wind[0] if self.wind is not None else base.wind
        if gravity_low + wind_low <= 0:
            raise ValueError("All samples must have gravity + wind > 0.")

    def sample(self, rng: np.random.Generator, base: PhysicsConfig) -> PhysicsConfig:
        """Return sampled PhysicsConfig using only ``rng``; never mutate ``base``.

        Input is an explicit local NumPy generator and a validated base config.
        Output is a validated immutable config. Intervals use pixels and steps,
        with negative horizontal pipe speeds and negative upward flap velocity.
        """
        sampled = {name: float(rng.uniform(*bounds)) for name, bounds in asdict(self).items()
                   if bounds is not None}
        return replace(base, **sampled)

