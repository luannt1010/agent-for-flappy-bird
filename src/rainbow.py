"""Rainbow: Double + Dueling + PER + n-step + C51 + factorized Noisy Nets.

The C51 loss is ``-sum_j m_j log p_theta(s,a,z_j)``, where m is the projected
n-step distribution ``R^(n) + gamma^n*(1-terminal)*Z_target(s',a*)`` and
``a*=argmax_a sum_j z_j*p_online(s',a,z_j)``. PER uses KL(m || p), as in the
original Rainbow paper, rather than the cross entropy (which includes the
target distribution's entropy and would change sampling priorities).

Primary references: Rainbow https://arxiv.org/abs/1710.02298;
C51 https://arxiv.org/abs/1707.06887;
Noisy Networks https://arxiv.org/abs/1706.10295.
"""

import copy
import math
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.dqn_agent import DQNAgent, linear_schedule, observation_tensor
from src.replay import NStepAccumulator
from src.rl_common import AgentConfig, ensure_finite


class NoisyLinear(nn.Module):
    """Factorized Gaussian layer: y=(mu_W+sigma_W*epsilon_W)x+mu_b+sigma_b*epsilon_b.

    ``epsilon_W=f(epsilon_out) outer f(epsilon_in)``,
    ``epsilon_b=f(epsilon_out)``, ``f(x)=sign(x)*sqrt(abs(x))`` and each base
    epsilon is standard Gaussian. Noise is active only in training mode;
    eval mode uses the learned mean parameters for deterministic inference.
    Input [...,in_features] maps to [...,out_features].
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        """Initialize means uniformly and factorized noise scales as in Fortunato.

        Weight sigma is sigma_init/sqrt(in_features); bias sigma is
        sigma_init/sqrt(out_features). Noise is sampled once at construction.
        """
        super().__init__()
        if in_features < 1 or out_features < 1 or sigma_init < 0:
            raise ValueError("NoisyLinear dimensions must be positive and sigma_init nonnegative.")
        self.in_features, self.out_features = in_features, out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma_init / math.sqrt(in_features))
        nn.init.constant_(self.bias_sigma, sigma_init / math.sqrt(out_features))
        self.reset_noise()

    @staticmethod
    def _scale_noise(values: torch.Tensor) -> torch.Tensor:
        """Transform iid Gaussian values with f(x)=sign(x)*sqrt(abs(x))."""
        return values.sign() * values.abs().sqrt()

    def reset_noise(self) -> None:
        """Sample fresh independent Gaussian factors on the parameter device."""
        epsilon_in = self._scale_noise(torch.randn(self.in_features, device=self.weight_mu.device,
                                                   dtype=self.weight_mu.dtype))
        epsilon_out = self._scale_noise(torch.randn(self.out_features, device=self.weight_mu.device,
                                                    dtype=self.weight_mu.dtype))
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Apply learned mean plus fixed sampled noise in train mode, mean in eval."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(values, weight, bias)


class RainbowNetwork(nn.Module):
    """Dueling categorical network with noisy streams and the original conv body.

    Input: raw float32 pixels [B,4,84,84]. Logits have shape [B,2,atoms],
    ``l(s,a,j)=v(s,j)+A(s,a,j)-mean_b A(s,b,j)``. A softmax over the atom axis
    represents the categorical return distribution for each action.
    """

    def __init__(self, atoms: int = 51, hidden_size: int = 512):
        """Create two action distributions with atoms >= 2 and positive hidden size."""
        super().__init__()
        if atoms < 2 or hidden_size < 1:
            raise ValueError("Rainbow needs atoms >= 2 and hidden_size >= 1.")
        self.atoms = int(atoms)
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, 8, 4), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1), nn.ReLU(inplace=True))
        self.value = nn.Sequential(NoisyLinear(7 * 7 * 64, hidden_size), nn.ReLU(),
                                   NoisyLinear(hidden_size, atoms))
        self.advantage = nn.Sequential(NoisyLinear(7 * 7 * 64, hidden_size), nn.ReLU(),
                                       NoisyLinear(hidden_size, 2 * atoms))
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return dueling categorical logits [B,2,atoms] from [B,4,84,84] pixels."""
        features = self.conv3(self.conv2(self.conv1(pixels))).flatten(1)
        value = self.value(features).view(-1, 1, self.atoms)
        advantage = self.advantage(features).view(-1, 2, self.atoms)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def distribution(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return normalized probabilities softmax(logits, atom dimension) [B,2,atoms]."""
        return self(pixels).softmax(dim=-1)

    def reset_noise(self) -> None:
        """Resample each NoisyLinear layer independently; no effect in eval mode."""
        for layer in self.modules():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


@torch.no_grad()
def categorical_projection(probabilities: torch.Tensor, rewards: torch.Tensor,
                           discounts: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Project a Bellman distribution onto equally spaced C51 atoms, conserving mass.

    Inputs: probabilities [B,N], rewards [B], bootstrap discounts [B], support
    [N] with N >= 2 increasing, equally spaced values. All tensors share a
    device. Output is detached [B,N]. For each source atom, let
    ``b=(clip(r+d*z_j,v_min,v_max)-v_min)/delta_z``; split its probability
    between floor(b) and ceil(b). When both indices coincide, all mass is
    assigned to that one index, avoiding the common zero-mass exact-bin bug.
    True termination is represented by d=0, time-limit truncation by d>0.
    """
    if (probabilities.ndim != 2 or support.ndim != 1 or support.numel() < 2
            or probabilities.shape[1] != support.numel()
            or rewards.shape != probabilities.shape[:1]
            or discounts.shape != rewards.shape):
        raise ValueError("C51 expects probabilities [B,N], rewards/discounts [B], support [N].")
    spacing = support[1] - support[0]
    if not bool(spacing > 0) or not torch.allclose(support[1:] - support[:-1], spacing.expand_as(support[1:]),
                                                 rtol=1e-4, atol=1e-6):
        raise ValueError("C51 support must be increasing and equally spaced.")
    count = support.numel()
    transformed = rewards[:, None] + discounts[:, None] * support[None]
    positions = ((transformed.clamp(support[0], support[-1]) - support[0]) / spacing).clamp(0, count - 1)
    lower, upper = positions.floor().long(), positions.ceil().long()
    lower_weights = upper.to(positions.dtype) - positions + (lower == upper).to(positions.dtype)
    upper_weights = positions - lower.to(positions.dtype)
    offsets = (torch.arange(probabilities.shape[0], device=probabilities.device) * count)[:, None]
    projected = torch.zeros_like(probabilities)
    projected.view(-1).index_add_(0, (lower + offsets).reshape(-1),
                                 (probabilities * lower_weights).reshape(-1))
    projected.view(-1).index_add_(0, (upper + offsets).reshape(-1),
                                 (probabilities * upper_weights).reshape(-1))
    return projected


class RainbowAgent(DQNAgent):
    """Full six-component Rainbow agent with distributional Double n-step targets.

    Noisy layers provide exploration; epsilon-greedy is disabled. Training
    minimizes importance-weighted categorical cross entropy; PER priority is
    ``KL(projected_target || online_distribution) + epsilon``. Replay stores
    uint8 CPU arrays and bootstrap discounts including actual n-step horizons.
    """

    def __init__(self, config: AgentConfig, device: str = "cpu", seed: int = 0):
        """Create Rainbow network/support, optimizer and prioritized replay from config."""
        self.config, self.device, self.variant = config, torch.device(device), "rainbow"
        self.rng = np.random.default_rng(seed)
        self.online = RainbowNetwork(config.atoms, config.hidden_size).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.support = torch.linspace(config.v_min, config.v_max, config.atoms, device=self.device)
        self._return_horizon = config.n_step
        self.replay = self._make_replay()
        self.accumulator = NStepAccumulator(self._return_horizon, config.gamma)
        self.environment_steps = 0
        self.gradient_steps = 0
        self._last_update_step = -1

    @property
    def epsilon(self) -> float:
        """Return zero: Rainbow explores with parameter noise, not epsilon-greedy."""
        return 0.0

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """Return argmax expected return, with fresh noisy parameters only if exploring.

        Input is uint8 [4,84,84]. ``explore=False`` uses learned mean weights
        and deterministic argmax over ``E[Z(s,a)]=sum_j p_j*z_j``.
        """
        previous_mode = self.online.training
        self.online.train(explore)
        if explore:
            self.online.reset_noise()
        with torch.no_grad():
            distribution = self.online.distribution(observation_tensor(state, self.device))
            values = ensure_finite((distribution * self.support).sum(dim=-1), "Rainbow action values")
            action = int(values.argmax(dim=1).item())
        self.online.train(previous_mode)
        return action

    def update(self) -> Dict[str, float]:
        """Take a scheduled Rainbow gradient step, returning loss/KL/Q/PER diagnostics.

        The online expectation selects the Double action and the independently
        noisy target evaluates its distribution. Projection/targets have no
        gradient. Target synchronization counts optimizer steps, as for DQN.
        """
        if not self._ready():
            return {}
        self._last_update_step = self.environment_steps
        beta = linear_schedule(self.config.per_beta_start, 1.0, self.environment_steps,
                               self.config.per_beta_steps)
        batch = self.replay.sample(self.config.batch_size, beta)
        states = observation_tensor(batch.states, self.device)
        next_states = observation_tensor(batch.next_states, self.device)
        actions = torch.as_tensor(batch.actions, device=self.device)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        discounts = torch.as_tensor(batch.discounts, device=self.device)
        weights = torch.as_tensor(batch.weights, device=self.device)
        rows = torch.arange(self.config.batch_size, device=self.device)
        self.online.train()
        self.target.train()
        self.online.reset_noise()
        self.target.reset_noise()
        log_probabilities = self.online(states).log_softmax(dim=-1)[rows, actions]
        with torch.no_grad():
            online_next = self.online.distribution(next_states)
            next_actions = (online_next * self.support).sum(dim=-1).argmax(dim=1)
            next_distribution = self.target.distribution(next_states)[rows, next_actions]
            projected = categorical_projection(next_distribution, rewards, discounts, self.support)
        per_item_loss = -(projected * log_probabilities).sum(dim=1)
        loss = (weights * per_item_loss).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip,
                                                error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            target_log = projected.clamp_min(torch.finfo(projected.dtype).tiny).log()
            kl_divergence = (projected * (target_log - log_probabilities.detach())).sum(dim=1).clamp_min(0)
            q_mean = (log_probabilities.detach().exp() * self.support).sum(dim=1).mean()
            unbounded_atoms = rewards[:, None] + discounts[:, None] * self.support
            clipped_mass = (next_distribution * ((unbounded_atoms < self.support[0])
                            | (unbounded_atoms > self.support[-1]))).sum(dim=1).mean()
        self.replay.update_priorities(batch.indices, kl_divergence.cpu().numpy())
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        return {"loss": float(loss.detach()), "kl_divergence": float(kl_divergence.mean()),
                "q_mean": float(q_mean), "beta": beta, "epsilon": 0.0,
                "support_clip_mass": float(clipped_mass), "gradient_norm": float(gradient_norm)}

    def state_dict(self) -> dict:
        """Return a replay-free checkpoint with C51 support metadata for safe restore."""
        payload = super().state_dict()
        payload["support"] = {"atoms": self.config.atoms, "v_min": self.config.v_min,
                              "v_max": self.config.v_max, "n_step": self.config.n_step}
        return payload

    def load_state_dict(self, payload: dict) -> None:
        """Restore Rainbow state, rejecting a checkpoint with a different C51 support.

        Network shapes alone cannot detect different v_min/v_max. Support
        metadata prevents silently interpreting distributions on another scale.
        """
        expected = {"atoms": self.config.atoms, "v_min": self.config.v_min,
                    "v_max": self.config.v_max, "n_step": self.config.n_step}
        if "support" in payload and payload["support"] != expected:
            raise ValueError("Rainbow checkpoint support/n_step differs from AgentConfig.")
        super().load_state_dict(payload)
