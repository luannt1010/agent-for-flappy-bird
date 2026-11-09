"""Shared categorical-policy networks and on-policy rollout mechanics.

Inputs are uint8 frame stacks ``[4, 84, 84]`` (or batched stacks); encoders
convert pixels to float32 and divide by 255 exactly once. The convolutional
architecture matches the existing DQN: 32@8/4, 64@4/2, 64@3/1, then a dense
hidden layer. Formula references: https://arxiv.org/abs/1506.02438 (GAE),
https://arxiv.org/abs/1707.06347 (PPO).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from src.rl_common import ensure_finite

if TYPE_CHECKING:
    from src.rl_common import AgentConfig


class PixelEncoder(nn.Module):
    """Map raw ``[B,4,84,84]`` pixels to ``[B,hidden_size]`` features."""

    def __init__(self, hidden_size: int = 512) -> None:
        """Create the baseline-shaped CNN; input is the desired feature width."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 7 * 7, hidden_size), nn.ReLU(),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return features for raw pixels using ``x = float32(states) / 255``."""
        return self.layers(states.to(dtype=torch.float32) / 255.0)


class CategoricalActorCritic(nn.Module):
    """Shared CNN with policy logits ``[B,2]`` and state values ``[B]``."""

    def __init__(self, hidden_size: int = 512) -> None:
        """Create encoder and independent linear policy/value output heads."""
        super().__init__()
        self.encoder = PixelEncoder(hidden_size)
        self.policy_head = nn.Linear(hidden_size, 2)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, V(s))``; ``pi(a|s) = softmax(logits)[a]``."""
        features = self.encoder(states)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


class CategoricalPolicy(nn.Module):
    """Separate SAC actor mapping raw frame stacks to two action logits."""

    def __init__(self, hidden_size: int = 512) -> None:
        """Create the pixel encoder and two-action categorical policy head."""
        super().__init__()
        self.encoder = PixelEncoder(hidden_size)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return ``[B,2]`` logits; probabilities equal ``softmax(logits)``."""
        return self.head(self.encoder(states))


class ActionValueNetwork(nn.Module):
    """SAC critic mapping raw frame stacks to ``[Q(s,0), Q(s,1)]``."""

    def __init__(self, hidden_size: int = 512) -> None:
        """Create an independent pixel encoder and two-action critic head."""
        super().__init__()
        self.encoder = PixelEncoder(hidden_size)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return a ``[B,2]`` tensor of unbounded action-value estimates."""
        return self.head(self.encoder(states))


def generalized_advantage_estimation(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute advantages and value targets from aligned one-dimensional arrays.

    ``delta_t = r_t + gamma*(1-terminated_t)*V(next_s_t) - V(s_t)``;
    ``A_t = delta_t + gamma*lambda*(1-terminated_t)*(1-truncated_t)*A_(t+1)``;
    ``R_t = A_t + V(s_t)``. Time-limit truncation bootstraps from the final
    observation but never propagates advantages into a reset episode. The last
    stored transition also bootstraps, allowing partial rollout updates.

    Inputs: equally sized arrays ``[T]``, discount and GAE lambda in [0,1].
    Outputs: float32 ``(advantages[T], returns[T])``. Lambda=1 gives n-step
    bootstrapped returns suitable for synchronous advantage actor-critic.
    """
    arrays = [np.asarray(item).reshape(-1) for item in
              (rewards, values, next_values, terminated, truncated)]
    if len({len(item) for item in arrays}) != 1:
        raise ValueError("GAE inputs must have equal lengths")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    rewards_, values_, next_values_, terminated_, truncated_ = arrays
    advantages = np.zeros(len(rewards_), dtype=np.float32)
    carry = 0.0
    for index in range(len(rewards_) - 1, -1, -1):
        bootstrap = 0.0 if bool(terminated_[index]) else float(next_values_[index])
        delta = float(rewards_[index]) + gamma * bootstrap - float(values_[index])
        continuation = not (bool(terminated_[index]) or bool(truncated_[index]))
        carry = delta + gamma * gae_lambda * float(continuation) * carry
        advantages[index] = carry
    return advantages, advantages + values_.astype(np.float32)


@dataclass
class RolloutTransition:
    """One on-policy transition with raw pixels and frozen behavior estimates."""

    state: np.ndarray
    action: int
    reward: float
    value: float
    next_value: float
    log_probability: float
    terminated: bool
    truncated: bool


class OnPolicyAgent:
    """Common collection/checkpoint contract for PPO and synchronous A2C.

    Call ``act -> env.step -> observe -> update`` for each training step.
    ``observe`` freezes behavior log probabilities and values before any update;
    ``act(explore=False)`` is deterministic and never changes rollout or RNG.
    """

    algorithm = "on_policy"

    def __init__(self, config: AgentConfig, device: str = "cpu", seed: int = 0) -> None:
        """Create actor-critic, Adam optimizer, private RNG and empty rollout."""
        self.config = config
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.network = CategoricalActorCritic(config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.rollout: list[RolloutTransition] = []
        self.environment_steps = 0
        self.gradient_steps = 0

    @torch.no_grad()
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """Return sampled training action or deterministic argmax-policy test action.

        Input: raw ``uint8[4,84,84]`` stack. Output: integer 0 or 1.
        Sampling uses this agent's NumPy RNG; deterministic evaluation does not
        consume random numbers and does not modify the training rollout.
        """
        logits, _ = self.network(torch.as_tensor(state, device=self.device).unsqueeze(0))
        ensure_finite(logits, f"{self.algorithm} policy logits")
        if not explore:
            return int(logits.argmax(dim=-1).item())
        probabilities = logits.softmax(dim=-1).squeeze(0).cpu().numpy().astype(np.float64)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(2, p=probabilities))

    @torch.no_grad()
    def observe(
        self, state: np.ndarray, action: int, reward: float,
        next_state: np.ndarray, terminated: bool, truncated: bool = False,
    ) -> None:
        """Store transition and frozen ``V(s), V(next_s), log pi_old(a|s)``.

        Inputs: raw current/final-next uint8 stacks, action 0/1, scalar reward,
        true MDP terminal flag and optional time-limit flag. Output: None.
        Pass the final observation before reset for truncated episodes. Terminal
        transitions set the bootstrap value to zero in GAE.
        """
        if action not in (0, 1):
            raise ValueError("Flappy Bird action must be 0 or 1")
        states = torch.as_tensor(np.stack((state, next_state)), device=self.device)
        logits, values = self.network(states)
        log_probability = logits[0].log_softmax(dim=-1)[action]
        self.rollout.append(RolloutTransition(
            np.array(state, dtype=np.uint8, copy=True), int(action), float(reward),
            float(values[0].item()), float(values[1].item()),
            float(log_probability.item()), bool(terminated), bool(truncated),
        ))
        self.environment_steps += 1

    def _prepare_rollout(self, gae_lambda: float) -> dict[str, torch.Tensor]:
        """Convert pending data to device tensors and calculate GAE/return targets.

        Input: lambda; output: dict of states ``[T,4,84,84]`` and aligned actions,
        old log probabilities, advantages, returns ``[T]``. Does not clear data.
        """
        transitions = self.rollout
        advantages, returns = generalized_advantage_estimation(
            np.asarray([item.reward for item in transitions]),
            np.asarray([item.value for item in transitions]),
            np.asarray([item.next_value for item in transitions]),
            np.asarray([item.terminated for item in transitions]),
            np.asarray([item.truncated for item in transitions]),
            self.config.gamma, gae_lambda,
        )
        return {
            "states": torch.as_tensor(np.stack([item.state for item in transitions]), device=self.device),
            "actions": torch.as_tensor([item.action for item in transitions], device=self.device),
            "old_log_probabilities": torch.as_tensor(
                [item.log_probability for item in transitions], dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(advantages, device=self.device),
            "returns": torch.as_tensor(returns, device=self.device),
        }

    def update(self) -> dict[str, float]:
        """Train when ``rollout_steps`` samples exist; otherwise return ``{}``.

        PPO/A2C use rollout cadence, independent of off-policy train_frequency.
        Outputs: scalar training diagnostics after an update.
        """
        if len(self.rollout) < self.config.rollout_steps:
            return {}
        return self._train_rollout()

    def _train_rollout(self) -> dict[str, float]:
        """Optimize pending samples and return diagnostics; implemented by subclass."""
        raise NotImplementedError

    def finish(self) -> dict[str, float]:
        """Optimize a final partial rollout, bootstrapping its final observation."""
        return self._train_rollout() if self.rollout else {}

    def state_dict(self) -> dict[str, Any]:
        """Return independent model/optimizer/RNG/counter/rollout checkpoint state.

        The pending rollout is included so loading between rollout boundaries
        preserves the policy behavior and every observed training transition.
        Its images are stored as CPU tensors, enabling torch.load with
        ``weights_only=True`` without allowlisting NumPy pickle globals.
        """
        return deepcopy({
            "algorithm": self.algorithm,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self.rng.bit_generator.state,
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "rollout": [{**vars(transition), "state": torch.from_numpy(transition.state.copy())}
                        for transition in self.rollout],
        })

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore a same-algorithm checkpoint; inputs use ``state_dict`` format."""
        if payload.get("algorithm", self.algorithm) != self.algorithm:
            raise ValueError("Checkpoint algorithm differs from this policy agent")
        self.network.load_state_dict(payload["network"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if "rng" in payload:
            self.rng.bit_generator.state = deepcopy(payload["rng"])
        self.environment_steps = int(payload.get("environment_steps", 0))
        self.gradient_steps = int(payload.get("gradient_steps", 0))
        self.rollout = []
        for item in payload.get("rollout", []):
            restored = deepcopy(item)
            state = restored["state"]
            restored["state"] = state.detach().cpu().numpy().copy() if isinstance(state, torch.Tensor) else np.array(state, copy=True)
            self.rollout.append(RolloutTransition(**restored))


def categorical_statistics(logits: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(log pi(a|s), H(pi))`` for logits ``[B,2]`` and actions ``[B]``.

    Entropy is exact categorical ``H(pi) = -sum_a pi(a|s)*log pi(a|s)``.
    Both outputs have shape ``[B]`` and retain gradients.
    """
    distribution = Categorical(logits=logits)
    return distribution.log_prob(actions), distribution.entropy()
