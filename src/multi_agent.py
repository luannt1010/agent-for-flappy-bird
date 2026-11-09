"""Simultaneous survival competition with shared obstacles and separate birds.

Birds cannot collide with or obstruct one another. Each agent observes its own
frame stack, acts independently, and receives the baseline game reward. The
same actual pipe-course object advances once per joint step, preventing course
differences after an agent dies. Ranking rewards survival duration first, then
pipes passed; exact ties share the win. No opponent state is added to pixels.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .benchmark_env import BenchmarkEnv


class CompetitiveFlappyBird:
    """Shared-course environment for two or more independent RL policies.

    ``reset`` returns a list of uint8 ``(4,84,84)`` observations. ``step``
    accepts one integer action per agent and returns observation, reward,
    terminated and truncated lists plus a common info dictionary. A dead
    agent's final frame and done flag persist; its later rewards are zero.
    """

    def __init__(self, num_agents: int = 2, seed: int = 0,
                 max_episode_steps: int = 10000, render: bool = False) -> None:
        """Configure independent birds and one seeded shared obstacle course.

        Each bird uses default baseline physics. Only agent zero optionally
        renders a window. ``num_agents`` must be at least two; training and
        competition evaluation choose actions outside this environment.
        """
        if not isinstance(num_agents, (int, np.integer)) or num_agents < 2:
            raise ValueError("num_agents must be an integer >= 2.")
        self.num_agents = int(num_agents)
        self.envs = [BenchmarkEnv(seed=seed, max_episode_steps=max_episode_steps,
                                  render=render and index == 0)
                     for index in range(self.num_agents)]
        self._seed = int(seed)
        self._needs_reset = True
        self._closed = False

    @property
    def alive(self) -> list[bool]:
        """Return one active flag per agent; before reset no birds are active."""
        if not hasattr(self, "_terminated") or self._closed:
            return [False] * self.num_agents
        return [not (terminated or truncated)
                for terminated, truncated in zip(self._terminated, self._truncated)]

    def reset(self, seed: int | None = None, progress: float = 0.0) -> list[np.ndarray]:
        """Reset all birds onto one course and return their initial frame stacks.

        The first bird owns course RNG; all other birds reference that same
        course object. ``seed=None`` continues its private RNG sequence.
        ``progress`` is accepted for environment-interface compatibility.
        """
        if self._closed:
            raise RuntimeError("Cannot reset a closed competition.")
        if seed is not None:
            self._seed = int(seed)
        observations = [env.reset(seed=seed, progress=progress) for env in self.envs]
        self._course = self.envs[0]._course
        for index, env in enumerate(self.envs[1:], start=1):
            env._course = self._course
            # The owner's RNG consumes obstacles throughout a round, whereas
            # other RNGs do not. On an unseeded reset their temporary courses
            # may differ, so rebuild initial observations after course sharing.
            frame = env._draw_frame()
            env._frames.clear()
            env._frames.extend(frame.copy() for _ in range(4))
            observations[index] = env._observation()
        self._terminated = [False] * self.num_agents
        self._truncated = [False] * self.num_agents
        self._observations = observations
        self._needs_reset = False
        return [item.copy() for item in observations]

    def step(self, actions: Sequence[int]) -> tuple[
            list[np.ndarray], list[float], list[bool], list[bool], dict[str, Any]]:
        """Advance the common world once and all still-active birds simultaneously.

        Returned info includes ``agents`` (per-agent score/time), ``ranking``
        (agent IDs in survival/score order), ``winner_ids`` (empty while still
        running), and ``finished``. Ties share a win; benchmark aggregation may
        allocate each tied winner ``1 / len(winner_ids)`` of a tournament win.
        """
        if self._needs_reset or self._closed:
            raise RuntimeError("Call reset before stepping or after competition ends.")
        if len(actions) != self.num_agents:
            raise ValueError("Supply exactly one action per agent.")
        if any(not isinstance(action, (int, np.integer)) or action not in (0, 1)
               for action in actions):
            raise ValueError("All actions must be integer 0 (coast) or 1 (flap).")
        assert self._course is not None
        self._course.advance()
        rewards = [0.0] * self.num_agents
        for index, (env, action) in enumerate(zip(self.envs, actions)):
            if self._terminated[index] or self._truncated[index]:
                continue
            obs, reward, terminated, truncated, _ = env._step(action, advance_course=False)
            self._observations[index] = obs
            rewards[index] = reward
            self._terminated[index], self._truncated[index] = terminated, truncated
        self._needs_reset = all(t or u for t, u in zip(self._terminated, self._truncated))
        agents = [dict(env._info(), agent_id=index) for index, env in enumerate(self.envs)]
        ranking = sorted(range(self.num_agents),
                         key=lambda index: (agents[index]["episode_steps"], agents[index]["score"]),
                         reverse=True)
        best = (agents[ranking[0]]["episode_steps"], agents[ranking[0]]["score"])
        winner_ids = [index for index in ranking
                      if (agents[index]["episode_steps"], agents[index]["score"]) == best]
        info = {"agents": agents, "ranking": ranking,
                "winner_ids": winner_ids if self._needs_reset else [],
                "finished": self._needs_reset, "pipe_course_shared": True,
                "seed": self._seed}
        return ([item.copy() for item in self._observations], rewards,
                self._terminated.copy(), self._truncated.copy(), info)

    def close(self) -> None:
        """Release every bird's resources and disallow further reset/step calls."""
        for env in self.envs:
            env.close()
        self._closed = True
