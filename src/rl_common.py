"""Shared, serializable hyperparameters and reproducible random state utilities.

Observations passed to agents are CPU uint8 arrays shaped (4, 84, 84).
Actions are integers: 0 = coast, 1 = flap. Rewards are unscaled game rewards.
"""

from dataclasses import asdict, dataclass
import random
from typing import Protocol

import numpy as np
import torch


def ensure_finite(values: torch.Tensor, name: str) -> torch.Tensor:
    """Reject NaN/Inf outputs before argmax can silently turn them into actions."""
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError(f"{name} contains NaN/Inf; benchmark aborted")
    return values


@dataclass(frozen=True)
class AgentConfig:
    """Algorithm settings; probabilities/discounts are in [0, 1].

    ``gamma`` discounts one game frame: G_t = sum_k gamma**k * r_(t+k).
    ``learning_rate`` applies to each optimizer. PER beta anneals from
    ``per_beta_start`` to 1; epsilon anneals linearly over environment steps.
    Replay capacity counts transitions, not bytes. SAC uses fixed temperature.
    """

    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 32
    replay_capacity: int = 10000
    warmup_steps: int = 1000
    target_update_interval: int = 1000
    epsilon_start: float = 0.1
    epsilon_end: float = 1e-4
    epsilon_decay_steps: int = 100000
    train_frequency: int = 4
    gradient_clip: float = 10.0
    n_step: int = 3
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 100000
    per_epsilon: float = 1e-6
    atoms: int = 51
    v_min: float = -10.0
    v_max: float = 100.0
    hidden_size: int = 512
    rollout_steps: int = 128
    ppo_epochs: int = 4
    minibatch_size: int = 32
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    sac_tau: float = 0.005
    sac_alpha: float = 0.2

    def __post_init__(self):
        """Reject invalid integer budgets, nonfinite values and invalid bounds."""
        values = asdict(self)
        for name, value in values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        positive_ints = (
            "batch_size", "replay_capacity", "target_update_interval",
            "epsilon_decay_steps", "train_frequency", "n_step", "per_beta_steps",
            "atoms", "hidden_size", "rollout_steps", "ppo_epochs", "minibatch_size",
        )
        for name in positive_ints:
            if not isinstance(values[name], int) or values[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        for name in ("gamma", "epsilon_start", "epsilon_end", "per_beta_start", "gae_lambda"):
            if not 0 <= values[name] <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end must not exceed epsilon_start")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size must not exceed replay_capacity")
        if self.atoms < 2 or self.v_min >= self.v_max:
            raise ValueError("C51 needs atoms >= 2 and v_min < v_max")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.per_epsilon <= 0:
            raise ValueError("learning_rate, gradient_clip and per_epsilon must be positive")
        if not 0 <= self.per_alpha <= 1 or not 0 < self.sac_tau <= 1:
            raise ValueError("per_alpha must be in [0,1] and sac_tau in (0,1]")
        if not 0 < self.ppo_clip < 1:
            raise ValueError("ppo_clip must be in (0, 1)")
        if min(self.value_coefficient, self.entropy_coefficient, self.sac_alpha) < 0:
            raise ValueError("Loss coefficients and SAC temperature must be nonnegative")


class Agent(Protocol):
    """Minimal interface used by the benchmark; checkpoints exclude replay data."""

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """Map uint8 (4,84,84) pixels to action 0/1; explore=False is deterministic."""
        ...

    def observe(self, state, action, reward, next_state, terminated, truncated=False):
        """Store a transition; only termination masks Bellman bootstrapping."""
        ...

    def update(self) -> dict:
        """Perform scheduled gradient updates; return scalar diagnostics or {}."""
        ...

    def state_dict(self) -> dict:
        """Return model/optimizer state suitable for torch.save."""
        ...

    def load_state_dict(self, payload: dict):
        """Restore a selection checkpoint for evaluation, not exact replay resume."""
        ...


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and Torch; optionally request deterministic kernels.

    Input is a nonnegative 32-bit seed. No return value. Exact reproducibility
    still requires the same hardware, Torch version and thread configuration.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic


def seed_sequence(namespace: int, count: int) -> list[int]:
    """Return count unique episode seeds from a reserved numeric namespace.

    Seeds are namespace + episode index. Keeping train/validation/test namespaces
    disjoint prevents checkpoint selection from using test courses.
    """
    if count < 1 or namespace < 0 or namespace + count >= 2**32:
        raise ValueError("Invalid seed namespace/count")
    return list(range(namespace, namespace + count))
