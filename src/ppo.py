"""Clipped PPO for two-action Flappy Bird with GAE and a shared CNN critic.

Reference: Schulman et al., https://arxiv.org/abs/1707.06347 .
Inputs and public agent protocol are documented by ``OnPolicyAgent``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.policy_networks import OnPolicyAgent, categorical_statistics


def clipped_policy_loss(
    log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    """Return scalar negative PPO surrogate for aligned tensors ``[B]``.

    ``r = exp(log pi_theta - log pi_old)``;
    ``L_policy = -mean(min(r*A, clip(r,1-epsilon,1+epsilon)*A))``.
    Advantages and behavior log probabilities must be frozen rollout targets.
    """
    ratios = (log_probabilities - old_log_probabilities).exp()
    return -torch.minimum(ratios * advantages, ratios.clamp(1 - clip, 1 + clip) * advantages).mean()


class PPOAgent(OnPolicyAgent):
    """PPO with fresh on-policy rollouts, minibatches, and configurable epochs.

    Minimizes ``L_policy + c_v*MSE(V,R) - c_H*mean(H(pi))``. Advantages
    are normalized once across a complete rollout; a one-sample rollout keeps
    its unnormalized advantage so a final partial update still learns.
    """

    algorithm = "ppo"

    def _train_rollout(self) -> dict[str, float]:
        """Optimize stored rollout using the clipped surrogate; return mean losses.

        Inputs: pending transitions in ``self.rollout``. Output diagnostics:
        policy_loss, value_loss, entropy, loss, approximate_kl and updates.
        Stored transitions are consumed after all PPO epochs.
        """
        batch = self._prepare_rollout(self.config.gae_lambda)
        advantages = batch["advantages"]
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        totals = {key: 0.0 for key in ("policy_loss", "value_loss", "entropy", "loss", "approximate_kl")}
        sample_count = 0
        updates = 0
        count = len(advantages)
        for _ in range(self.config.ppo_epochs):
            indices = self.rng.permutation(count)
            for start in range(0, count, self.config.minibatch_size):
                index = torch.as_tensor(indices[start:start + self.config.minibatch_size], device=self.device)
                logits, values = self.network(batch["states"][index])
                log_probabilities, entropy = categorical_statistics(logits, batch["actions"][index])
                old_log_probabilities = batch["old_log_probabilities"][index]
                policy_loss = clipped_policy_loss(
                    log_probabilities, old_log_probabilities, advantages[index], self.config.ppo_clip,
                )
                value_loss = F.mse_loss(values, batch["returns"][index])
                mean_entropy = entropy.mean()
                loss = policy_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * mean_entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip,
                                         error_if_nonfinite=True)
                self.optimizer.step()
                with torch.no_grad():
                    log_ratio = log_probabilities - old_log_probabilities
                    approximate_kl = ((log_ratio.exp() - 1) - log_ratio).mean()
                metrics = {"policy_loss": policy_loss, "value_loss": value_loss,
                           "entropy": mean_entropy, "loss": loss, "approximate_kl": approximate_kl}
                weight = len(index)
                for key, value in metrics.items():
                    totals[key] += float(value.detach().item()) * weight
                sample_count += weight
                updates += 1
        self.gradient_steps += updates
        self.rollout.clear()
        return {**{key: value / sample_count for key, value in totals.items()}, "updates": float(updates)}
