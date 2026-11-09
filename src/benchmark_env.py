"""Deterministic, unthrottled Flappy Bird environment for RL experiments.

The legacy game is intentionally left unchanged. This environment retains its
sprites, binary 84x84 observations, rewards and default physics, while exposing
explicit episode boundaries and an independent ``numpy.random.Generator``.
It never opens a window unless ``render=True`` and never automatically resets.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import pygame


@dataclass(frozen=True)
class PhysicsConfig:
    """Episode physics; distances are pixels and time is one 1/30-second step.

    Without a flap, ``v' = min(max_fall_velocity, v + gravity + wind)``;
    a flap sets ``v' = flap_velocity``. Position is ``y' = max(0, y + v')``
    and pipe position is ``x' = x + pipe_speed``. Positive wind blows down.
    ``pipe_gap`` is vertical clearance and ``pipe_spacing`` is horizontal
    left-edge spacing. The default configuration reproduces baseline values.
    """

    pipe_gap: float = 100.0
    pipe_speed: float = -4.0
    gravity: float = 1.0
    flap_velocity: float = -9.0
    wind: float = 0.0
    pipe_spacing: float = 144.0
    max_fall_velocity: float = 10.0

    def __post_init__(self) -> None:
        """Reject nonfinite values and geometry that cannot fit a bird/course."""
        if not all(np.isfinite(value) for value in asdict(self).values()):
            raise ValueError("All physics parameters must be finite.")
        if not 40.0 <= self.pipe_gap <= 284.0:
            raise ValueError("pipe_gap must be between 40 and 284 pixels.")
        if self.pipe_speed >= 0:
            raise ValueError("pipe_speed must be negative (pipes move left).")
        if self.gravity <= 0 or self.gravity + self.wind <= 0:
            raise ValueError("gravity and gravity + wind must be positive.")
        if self.flap_velocity >= 0 or self.max_fall_velocity <= 0:
            raise ValueError("flap_velocity must be negative; max_fall_velocity positive.")
        if self.pipe_spacing < 90:
            raise ValueError("pipe_spacing must be at least 90 pixels.")


@dataclass(frozen=True)
class PipeLayout:
    """Placement rules for consecutive gaps, separate from game physics.

    Gap tops lie on a 10px grid. With constant gap size, a gap-top shift is
    also the shift of its center. Defaults preserve the legacy independent
    100..170px sampling; surprise layouts constrain each consecutive shift.
    """

    gap_top_min: float = 100.0
    gap_top_max: float = 170.0
    min_shift: float = 0.0
    max_shift: float | None = None

    def __post_init__(self) -> None:
        values = (self.gap_top_min, self.gap_top_max, self.min_shift)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Pipe layout bounds must be finite.")
        if self.gap_top_min < 20 or self.gap_top_max < self.gap_top_min or self.min_shift < 0:
            raise ValueError("Invalid pipe layout bounds or minimum shift.")
        if self.max_shift is not None and (not np.isfinite(self.max_shift) or self.max_shift < self.min_shift):
            raise ValueError("max_shift must be finite and >= min_shift, or None.")

    def allowed_after(self, candidates: np.ndarray, previous: float) -> np.ndarray:
        """Filter valid positions by absolute displacement from the previous gap."""
        shifts = np.abs(candidates - previous)
        allowed = shifts >= self.min_shift
        if self.max_shift is not None:
            allowed &= shifts <= self.max_shift
        return candidates[allowed]


class RandomizationProtocol(Protocol):
    """Structural interface for optional per-episode physics randomization."""

    def sample(self, rng: np.random.Generator, base: PhysicsConfig) -> PhysicsConfig:
        """Return sampled physics, consuming only the supplied local generator."""
        ...


class CurriculumProtocol(Protocol):
    """Structural interface for a training-progress-dependent physics schedule."""

    def at(self, progress: float, base: PhysicsConfig) -> PhysicsConfig:
        """Return physics for progress in the closed interval [0, 1]."""
        ...


@dataclass
class _Pipe:
    """One obstacle with its horizontal left edge and vertical gap top."""

    x: float
    gap_top: float


class _PipeCourse:
    """Shared obstacle world, driven solely by episode seed and elapsed steps."""

    def __init__(self, rng: np.random.Generator, physics: PhysicsConfig,
                 pipe_width: int, base_y: float, bird_center_x: float,
                 layout: PipeLayout | None = None) -> None:
        """Create two initial obstacles; input RNG belongs to this episode."""
        self.rng = rng
        self.physics = physics
        self.pipe_width = pipe_width
        self.base_y = base_y
        self.bird_center_x = bird_center_x
        self.layout = layout or PipeLayout()
        candidates = np.arange(self.layout.gap_top_min, self.layout.gap_top_max + 1e-6, 10.0)
        self._gap_candidates = candidates[candidates + physics.pipe_gap <= base_y - 20]
        if not len(self._gap_candidates):
            raise ValueError("pipe_gap leaves no valid obstacle placement.")
        # Reject dead ends before an episode starts; never silently relax rules.
        if any(not len(self.layout.allowed_after(self._gap_candidates, previous))
               for previous in self._gap_candidates):
            raise ValueError("Pipe layout has a gap position with no valid successor.")
        self._previous_gap_top = None
        self.pipes = [self._generate(288.0), self._generate(288.0 + physics.pipe_spacing)]
        self.passed_this_step = 0

    def _generate(self, x: float) -> _Pipe:
        """Sample a valid gap, applying the declared consecutive-shift rule.

        Baseline samples ``80 + 10*k``, ``k in {2,...,9}``, exactly as the
        original game. Large gaps restrict this set to leave 20px below it.
        """
        candidates = self._gap_candidates
        if self._previous_gap_top is not None:
            candidates = self.layout.allowed_after(candidates, self._previous_gap_top)
        self._previous_gap_top = float(self.rng.choice(candidates))
        return _Pipe(float(x), self._previous_gap_top)

    def advance(self) -> None:
        """Advance once and count center crossings without speed assumptions.

        A pipe scores iff ``old_center >= bird_center > new_center``. New
        pipes are placed ``pipe_spacing`` beyond the last pipe as its left
        edge enters the screen; a while loop handles arbitrarily large steps.
        """
        self.passed_this_step = 0
        for pipe in self.pipes:
            previous_center = pipe.x + self.pipe_width / 2
            pipe.x += self.physics.pipe_speed
            if previous_center >= self.bird_center_x > pipe.x + self.pipe_width / 2:
                self.passed_this_step += 1
        while self.pipes[-1].x <= 288.0:
            self.pipes.append(self._generate(self.pipes[-1].x + self.physics.pipe_spacing))
        self.pipes[:] = [pipe for pipe in self.pipes if pipe.x + self.pipe_width >= 0]


@lru_cache(maxsize=1)
def _load_assets() -> dict[str, Any]:
    """Load immutable sprite surfaces/masks by absolute path, without a display.

    ``pygame.image.load`` retains per-pixel alpha without ``convert_alpha``;
    consequently headless experiments need neither SDL display nor audio init.
    """
    sprites = Path(__file__).resolve().parents[1] / "assets" / "sprites"
    pipe = pygame.image.load(str(sprites / "pipe-green.png"))
    pipes = [pygame.transform.rotate(pipe, 180), pipe]
    birds = [pygame.image.load(str(sprites / f"redbird-{phase}.png"))
             for phase in ("upflap", "midflap", "downflap")]
    return {
        "background": pygame.image.load(str(sprites / "background-black.png")),
        "base": pygame.image.load(str(sprites / "base.png")),
        "pipes": pipes,
        "birds": birds,
        "pipe_masks": [pygame.surfarray.array_alpha(item).astype(bool) for item in pipes],
        "bird_masks": [pygame.surfarray.array_alpha(item).astype(bool) for item in birds],
    }


class BenchmarkEnv:
    """Pixel environment with Gym-style reset/step and explicit time limits.

    ``reset(seed, progress)`` returns uint8 ``(4,84,84)`` frames in oldest to
    newest order. ``step(action)`` accepts integer 0 (coast) or 1 (flap), and
    returns ``(observation, reward, terminated, truncated, info)``. Rewards are
    +0.1 per live step, +1 per pipe-crossing step and -1 on collision. A time
    limit is a truncation, so algorithms can bootstrap its final observation.
    """

    screen_width = 288
    screen_height = 512
    fps = 30
    base_y = screen_height * 0.79

    def __init__(self, seed: int = 0, max_episode_steps: int = 10000,
                 render: bool = False, physics: PhysicsConfig | None = None,
                 randomization: RandomizationProtocol | None = None,
                 curriculum: CurriculumProtocol | None = None,
                 layout: PipeLayout | None = None) -> None:
        """Configure the environment; call reset before taking the first action.

        ``seed`` initializes a local RNG and does not mutate NumPy's global
        state. Curriculum first adjusts base physics at each reset, then an
        optional randomizer samples episode physics from that configuration.
        Neither reset nor step throttles execution to real time.
        """
        if not isinstance(max_episode_steps, (int, np.integer)) or max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer.")
        self.max_episode_steps = int(max_episode_steps)
        self.render_enabled = bool(render)
        self.base_physics = physics or PhysicsConfig()
        self.physics = self.base_physics
        self.randomization = randomization
        self.curriculum = curriculum
        self.layout = layout or PipeLayout()
        self._seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._assets = _load_assets()
        self.bird_width, self.bird_height = self._assets["birds"][0].get_size()
        self.pipe_width, self.pipe_height = self._assets["pipes"][0].get_size()
        self.bird_x = int(self.screen_width / 5)
        self._surface = pygame.Surface((self.screen_width, self.screen_height))
        self._window: pygame.Surface | None = None
        self._frames: deque[np.ndarray] = deque(maxlen=4)
        self._needs_reset = True
        self._closed = False
        self._course: _PipeCourse | None = None

    @property
    def pipes(self) -> list[dict[str, float]]:
        """Return a snapshot of obstacle coordinates for diagnostics/rendering."""
        if self._course is None:
            return []
        return [{"x": pipe.x, "gap_top": pipe.gap_top,
                 "gap_bottom": pipe.gap_top + self.physics.pipe_gap}
                for pipe in self._course.pipes]

    def render_surface(self) -> pygame.Surface:
        """Return a copy of the latest game frame for a separate demo window.

        Available after reset/step, including the final frame of an episode.
        Drawing a scoreboard on this copy cannot alter the agent's observations.
        This method neither advances the world nor creates a display.
        """
        if self._closed or self._course is None:
            raise RuntimeError("Call reset() before requesting a game frame.")
        return self._surface.copy()

    def reset(self, seed: int | None = None, progress: float = 0.0) -> np.ndarray:
        """Start an episode and return four copies of its initial uint8 frame.

        A supplied seed restarts the local RNG; ``seed=None`` continues its
        stream. ``progress`` is elapsed training steps / maximum step budget;
        schedules clamp it to [0,1]. Evaluation passes baseline physics with no
        curriculum/randomizer and independent held-out seeds.
        """
        if self._closed:
            raise RuntimeError("Cannot reset a closed environment.")
        if not np.isfinite(progress):
            raise ValueError("progress must be finite.")
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(seed)
        self.physics = self.base_physics
        if self.curriculum is not None:
            self.physics = self.curriculum.at(progress, self.physics)
        if self.randomization is not None:
            self.physics = self.randomization.sample(self._rng, self.physics)
        self.bird_y = float(int((self.screen_height - self.bird_height) / 2))
        self.current_velocity_y = 0.0
        self.bird_index = 0
        self.episode_steps = 0
        self.score = 0
        self.base_x = 0.0
        self._needs_reset = False
        self._course = _PipeCourse(self._rng, self.physics, self.pipe_width,
                                   self.base_y, self.bird_x + self.bird_width / 2, self.layout)
        frame = self._draw_frame()
        self._frames.clear()
        self._frames.extend(frame.copy() for _ in range(4))
        return self._observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one action and return the final frame even on episode end.

        Calling step before reset or after collision/truncation raises an
        error. This prevents silently mixing transitions across episodes.
        """
        return self._step(action, advance_course=True)

    def _step(self, action: int, *, advance_course: bool) -> tuple[
            np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one bird; competition may advance its shared course once.

        ``advance_course=False`` requires a course already advanced for this
        simultaneous timestep. All birds then use the same crossing count.
        """
        if self._needs_reset or self._closed:
            raise RuntimeError("Call reset() before stepping or after an episode ends.")
        if not isinstance(action, (int, np.integer)) or action not in (0, 1):
            raise ValueError("action must be integer 0 (coast) or 1 (flap).")
        if action == 1:
            self.current_velocity_y = self.physics.flap_velocity
        else:
            self.current_velocity_y = min(self.physics.max_fall_velocity,
                                          self.current_velocity_y + self.physics.gravity + self.physics.wind)
        self.bird_y = max(0.0, self.bird_y + self.current_velocity_y)
        self.episode_steps += 1
        self.bird_index = (0, 1, 2, 1)[(self.episode_steps // 3) % 4]
        shift = self._assets["base"].get_width() - self._assets["background"].get_width()
        self.base_x = -((-self.base_x + 100) % shift) if shift else 0
        assert self._course is not None
        if advance_course:
            self._course.advance()
        terminated = self._is_collided()
        if not terminated:
            self.score += self._course.passed_this_step
        truncated = self.episode_steps >= self.max_episode_steps and not terminated
        reward = -1.0 if terminated else (float(self._course.passed_this_step)
                                        if self._course.passed_this_step else 0.1)
        self._needs_reset = terminated or truncated
        self._frames.append(self._draw_frame())
        return self._observation(), reward, terminated, truncated, self._info()

    def _is_collided(self) -> bool:
        """Check ground and alpha-mask overlap against every upper/lower pipe."""
        if self.bird_y + self.bird_height + 1 >= self.base_y:
            return True
        bird_box = pygame.Rect(self.bird_x, self.bird_y, self.bird_width, self.bird_height)
        bird_mask = self._assets["bird_masks"][self.bird_index]
        assert self._course is not None
        for pipe in self._course.pipes:
            for index, pipe_y in enumerate((pipe.gap_top - self.pipe_height,
                                            pipe.gap_top + self.physics.pipe_gap)):
                pipe_box = pygame.Rect(pipe.x, pipe_y, self.pipe_width, self.pipe_height)
                overlap = bird_box.clip(pipe_box)
                if overlap.width <= 0 or overlap.height <= 0:
                    continue
                bx, by = overlap.x - bird_box.x, overlap.y - bird_box.y
                px, py = overlap.x - pipe_box.x, overlap.y - pipe_box.y
                bird_pixels = bird_mask[bx:bx + overlap.width, by:by + overlap.height]
                pipe_pixels = self._assets["pipe_masks"][index][
                    px:px + overlap.width, py:py + overlap.height]
                if np.any(bird_pixels & pipe_pixels):
                    return True
        return False

    def _draw_frame(self) -> np.ndarray:
        """Render offscreen and return binary uint8 (84,84) legacy observation.

        The legacy train/test loops first crop pygame's ``(W,H,3)`` RGB array
        to ``pixels[:288, :int(base_y)]`` and pass it directly as OpenCV BGR
        input. This crop, axis orientation and threshold ``gray > 1`` are
        retained for existing checkpoint input compatibility.
        """
        self._surface.blit(self._assets["background"], (0, 0))
        self._surface.blit(self._assets["base"], (self.base_x, self.base_y))
        self._surface.blit(self._assets["birds"][self.bird_index], (self.bird_x, self.bird_y))
        assert self._course is not None
        for pipe in self._course.pipes:
            self._surface.blit(self._assets["pipes"][0], (pipe.x, pipe.gap_top - self.pipe_height))
            self._surface.blit(self._assets["pipes"][1], (pipe.x, pipe.gap_top + self.physics.pipe_gap))
        if self.render_enabled:
            if self._window is None:
                pygame.display.init()
                self._window = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Flappy Bird benchmark")
            pygame.event.pump()
            self._window.blit(self._surface, (0, 0))
            pygame.display.flip()
        pixels = pygame.surfarray.array3d(self._surface)
        pixels = pixels[:self.screen_width, :int(self.base_y)]
        gray = cv2.cvtColor(cv2.resize(pixels, (84, 84)), cv2.COLOR_BGR2GRAY)
        return np.where(gray > 1, 255, 0).astype(np.uint8)

    def _observation(self) -> np.ndarray:
        """Return a copied oldest-to-newest frame stack, shape (4,84,84)."""
        return np.stack(self._frames)

    def _info(self) -> dict[str, Any]:
        """Return serializable score, elapsed steps/30, seed and episode physics."""
        return {"score": self.score, "episode_steps": self.episode_steps,
                "survival_seconds": self.episode_steps / self.fps,
                "seed": self._seed, "physics": asdict(self.physics), "layout": asdict(self.layout)}

    def close(self) -> None:
        """Release this environment's optional display and reject further use."""
        if self._window is not None:
            pygame.display.quit()
            self._window = None
        self._closed = True

