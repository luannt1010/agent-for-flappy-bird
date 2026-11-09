"""Discrete Soft Actor-Critic (SAC) with exact categorical expectations.

Flappy Bird has two discrete actions, so continuous Gaussian SAC and its
reparameterization formula do not apply. This implementation follows
Christodoulou, https://arxiv.org/abs/1910.07207 , with separate actor, twin
critics, slowly updated target critics and a fixed configurable temperature.
Raw uint8 pixels are normalized by 255 inside every network.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.policy_networks import ActionValueNetwork, CategoricalPolicy
from src.rl_common import ensure_finite

if TYPE_CHECKING:
    from src.rl_common import AgentConfig


def discrete_soft_value(
    logits: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor, alpha: float,
) -> torch.Tensor:
    """Return exact soft values ``[B]`` from logits and twin Q arrays ``[B,A]``.

    ``V(s) = sum_a pi(a|s) * (min(Q1(s,a), Q2(s,a)) - alpha*log pi(a|s))``.
    ``alpha`` is the nonnegative fixed entropy temperature. This function
    retains gradients; use ``torch.no_grad`` for target computations.
    """
    log_probabilities = logits.log_softmax(dim=-1)
    probabilities = log_probabilities.exp()
    return (probabilities * (torch.minimum(q1, q2) - alpha * log_probabilities)).sum(dim=-1)


def soft_bellman_target(
    rewards: torch.Tensor, terminated: torch.Tensor,
    next_soft_values: torch.Tensor, gamma: float,
) -> torch.Tensor:
    """Return target ``y = r + gamma*(1-terminated)*V_soft(next_s)``.

    Inputs and output: aligned scalar tensors ``[B]``. Time-limit truncations
    are deliberately absent from the mask: they bootstrap from the final
    observation before reset. Call without gradients for critic targets.
    """
    return rewards + gamma * (1.0 - terminated.to(dtype=rewards.dtype)) * next_soft_values


class SACAgent:
    """Off-policy categorical SAC with uniform replay and fixed temperature.

    ``L_Qi = mean((Qi(s,a)-y)^2)``;
    ``L_pi = mean(sum_a pi(a|s)*(alpha*log pi(a|s)-min_i Qi(s,a)))``;
    ``target_i <- (1-tau)*target_i + tau*Qi`` after each optimizer update.
    Inputs to act/observe are raw ``uint8[4,84,84]`` stacks; actions are 0/1.
    During warmup training actions are uniform; test actions always maximize
    the learned policy. Replay stores both final-next and current frames.
    """

    algorithm = "sac"

    def __init__(self, config: AgentConfig, device: str = "cpu", seed: int = 0) -> None:
        """Create seeded actor, independent critics, target copies, optimizers and replay."""
        self.config = config
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.actor = CategoricalPolicy(config.hidden_size).to(self.device)
            self.critic1 = ActionValueNetwork(config.hidden_size).to(self.device)
            self.critic2 = ActionValueNetwork(config.hidden_size).to(self.device)
        self.target1 = deepcopy(self.critic1).requires_grad_(False)
        self.target2 = deepcopy(self.critic2).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=config.learning_rate,
        )
        self.replay: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._replay_position = 0
        self.environment_steps = 0
        self.gradient_steps = 0
        self._last_update_step = -1

    @torch.no_grad()
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """Return integer action for raw ``[4,84,84]`` pixels without storing data.

        Training uses uniform warmup then ``Categorical(pi)`` sampling.
        Evaluation uses ``argmax_a pi(a|s)`` with no RNG/counter/replay mutation.
        """
        if explore and self.environment_steps < self.config.warmup_steps:
            return int(self.rng.integers(2))
        logits = self.actor(torch.as_tensor(state, device=self.device).unsqueeze(0))
        ensure_finite(logits, "SAC policy logits")
        if not explore:
            return int(logits.argmax(dim=-1).item())
        probabilities = logits.softmax(dim=-1).squeeze(0).cpu().numpy().astype(np.float64)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(2, p=probabilities))

    def observe(
        self, state: np.ndarray, action: int, reward: float,
        next_state: np.ndarray, terminated: bool, truncated: bool = False,
    ) -> None:
        """Copy one transition into replay and increment the environment counter.

        Inputs: raw stacks, action 0/1, scalar reward, true terminal flag and
        optional time limit. Output: None. Truncation is not a Bellman terminal;
        ``next_state`` must be the final observation before reset.
        """
        if action not in (0, 1):
            raise ValueError("Flappy Bird action must be 0 or 1")
        transition = (np.array(state, dtype=np.uint8, copy=True), int(action), float(reward),
                      np.array(next_state, dtype=np.uint8, copy=True), bool(terminated))
        if len(self.replay) < self.config.replay_capacity:
            self.replay.append(transition)
        else:
            self.replay[self._replay_position] = transition
        self._replay_position = (self._replay_position + 1) % self.config.replay_capacity
        self.environment_steps += 1

    def _sample_batch(self) -> dict[str, torch.Tensor]:
        """Uniformly sample replay without replacement into aligned device tensors.

        Output keys: states/next_states ``[B,4,84,84]``, actions ``[B]``, and
        float32 rewards/terminated ``[B]``. Raw image dtype remains uint8.
        """
        indices = self.rng.choice(len(self.replay), self.config.batch_size, replace=False)
        transitions = [self.replay[int(index)] for index in indices]
        states, actions, rewards, next_states, terminated = zip(*transitions)
        return {
            "states": torch.as_tensor(np.stack(states), device=self.device),
            "actions": torch.as_tensor(actions, device=self.device),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            "next_states": torch.as_tensor(np.stack(next_states), device=self.device),
            "terminated": torch.as_tensor(terminated, dtype=torch.float32, device=self.device),
        }

    def update(self) -> dict[str, float]:
        """Perform one twin-critic/actor/target update when replay and cadence permit.

        Input: None; output: scalar critic_loss, actor_loss, entropy, alpha,
        mean target and updates=1, or ``{}`` while warming up. Exactly one
        update is permitted per observed step satisfying train_frequency.
        """
        if (len(self.replay) < self.config.batch_size
                or self.environment_steps < self.config.warmup_steps
                or self.environment_steps % self.config.train_frequency != 0
                or self._last_update_step == self.environment_steps):
            return {}
        batch = self._sample_batch()
        with torch.no_grad():
            next_soft_values = discrete_soft_value(
                self.actor(batch["next_states"]), self.target1(batch["next_states"]),
                self.target2(batch["next_states"]), self.config.sac_alpha,
            )
            targets = soft_bellman_target(
                batch["rewards"], batch["terminated"], next_soft_values, self.config.gamma,
            )
        actions = batch["actions"].unsqueeze(-1)
        q1 = self.critic1(batch["states"]).gather(1, actions).squeeze(-1)
        q2 = self.critic2(batch["states"]).gather(1, actions).squeeze(-1)
        critic_loss = F.mse_loss(q1, targets) + F.mse_loss(q2, targets)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())
        nn.utils.clip_grad_norm_(critic_parameters, self.config.gradient_clip, error_if_nonfinite=True)
        self.critic_optimizer.step()

        logits = self.actor(batch["states"])
        with torch.no_grad():
            q1_all = self.critic1(batch["states"])
            q2_all = self.critic2(batch["states"])
        actor_loss = -discrete_soft_value(logits, q1_all, q2_all, self.config.sac_alpha).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
        self.actor_optimizer.step()
        with torch.no_grad():
            for target, online in ((self.target1, self.critic1), (self.target2, self.critic2)):
                for target_parameter, online_parameter in zip(target.parameters(), online.parameters()):
                    target_parameter.lerp_(online_parameter, self.config.sac_tau)
            log_probabilities = logits.log_softmax(dim=-1)
            entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1).mean()
        self.gradient_steps += 1
        self._last_update_step = self.environment_steps
        return {"critic_loss": float(critic_loss.detach().item()),
                "actor_loss": float(actor_loss.detach().item()),
                "entropy": float(entropy.item()), "alpha": float(self.config.sac_alpha),
                "target_mean": float(targets.mean().item()), "updates": 1.0}

    def finish(self) -> dict[str, float]:
        """Return ``{}``; off-policy SAC has no pending on-policy rollout to flush."""
        return {}

    def state_dict(self) -> dict[str, Any]:
        """Return model, targets, optimizer, private RNG and counter checkpoint.

        Replay is intentionally excluded to keep inference/best-model snapshots
        compact. Loading starts a fresh replay buffer; this is a warm start,
        not exact continuation of an off-policy training trajectory.
        """
        return deepcopy({
            "algorithm": self.algorithm,
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(), "critic2": self.critic2.state_dict(),
            "target1": self.target1.state_dict(), "target2": self.target2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "rng": self.rng.bit_generator.state,
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
        })

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore matching SAC model/optimizer state and clear unsaved replay.

        Input: dict returned by state_dict. Output: None. Targets fall back to
        online critics if absent, allowing explicitly prepared actor/critic
        initialization checkpoints. Existing replay is discarded on load.
        """
        if payload.get("algorithm", self.algorithm) != self.algorithm:
            raise ValueError("Checkpoint algorithm differs from SAC")
        self.actor.load_state_dict(payload["actor"])
        self.critic1.load_state_dict(payload["critic1"])
        self.critic2.load_state_dict(payload["critic2"])
        self.target1.load_state_dict(payload.get("target1", payload["critic1"]))
        self.target2.load_state_dict(payload.get("target2", payload["critic2"]))
        if "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        if "critic_optimizer" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if "rng" in payload:
            self.rng.bit_generator.state = deepcopy(payload["rng"])
        self.environment_steps = int(payload.get("environment_steps", 0))
        self.gradient_steps = int(payload.get("gradient_steps", 0))
        self._last_update_step = -1
        self.replay.clear()
        self._replay_position = 0
