"""Synchronous advantage actor-critic for categorical Flappy Bird actions.

This is a single-environment synchronous A2C implementation with n-step
bootstrapped returns, not asynchronous A3C. The underlying actor-critic
objective follows Mnih et al.: https://proceedings.mlr.press/v48/mniha16.pdf .
"""

from __future__ import annotations

from torch import nn
from torch.nn import functional as F

from src.policy_networks import OnPolicyAgent, categorical_statistics


class A2CAgent(OnPolicyAgent):
    """One gradient update per rollout using n-step advantages and entropy.

    ``R_t = sum_(i=0)^(n-1) gamma^i*r_(t+i) + gamma^n*V(s_(t+n))``;
    ``A_t = R_t - V(s_t)``;
    ``L = -mean(log pi(a_t|s_t)*stop_gradient(A_t))
           + c_v*MSE(V(s_t), R_t) - c_H*mean(H(pi))``.
    n is the remaining rollout length, cut at episode reset. True terminal
    states have no bootstrap; time-limit truncations do bootstrap.
    """

    algorithm = "a2c"

    def _train_rollout(self) -> dict[str, float]:
        """Consume pending rollout in one Adam step and return scalar losses.

        Input: collected transitions. Outputs: policy_loss, value_loss, entropy,
        total loss and updates=1. GAE lambda=1 yields n-step returns.
        """
        batch = self._prepare_rollout(gae_lambda=1.0)
        logits, values = self.network(batch["states"])
        log_probabilities, entropy = categorical_statistics(logits, batch["actions"])
        policy_loss = -(log_probabilities * batch["advantages"]).mean()
        value_loss = F.mse_loss(values, batch["returns"])
        mean_entropy = entropy.mean()
        loss = policy_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * mean_entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip,
                                 error_if_nonfinite=True)
        self.optimizer.step()
        self.gradient_steps += 1
        self.rollout.clear()
        return {"policy_loss": float(policy_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "entropy": float(mean_entropy.detach().item()),
                "loss": float(loss.detach().item()), "updates": 1.0}
