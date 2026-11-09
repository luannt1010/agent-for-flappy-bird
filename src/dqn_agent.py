"""Original DQN baseline and isolated Double/Dueling/PER extensions.

The baseline preserves ``DeepQNetwork`` and raw 0..255 float pixel inputs.
It corrects the old training loop's undetached target and adds a frozen target
network: ``y = r + gamma*(1-terminal)*max_a Q_target(s',a)``. Baseline and all
isolated extensions minimize mean squared TD error, with one-step returns.

References: Mnih et al. https://www.nature.com/articles/nature14236;
Double DQN https://arxiv.org/abs/1509.06461;
Dueling https://arxiv.org/abs/1511.06581;
PER https://arxiv.org/abs/1511.05952.
"""

import copy
from typing import Dict

import numpy as np
import torch
from torch import nn

from src.deep_q_network import DeepQNetwork
from src.replay import NStepAccumulator, PrioritizedReplayBuffer, ReplayBuffer
from src.rl_common import AgentConfig, ensure_finite


def observation_tensor(observations: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert uint8 [4,84,84] or [B,4,84,84] to float32 batch on device.

    Output keeps the original network's pixel scale [0,255], without /255
    normalization, so existing baseline model weights remain compatible.
    """
    array = np.asarray(observations)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4 or array.shape[1:] != (4, 84, 84) or array.dtype != np.uint8:
        raise ValueError("Expected uint8 observations of shape (4,84,84) or (B,4,84,84).")
    return torch.as_tensor(array, device=device, dtype=torch.float32)


def linear_schedule(start: float, end: float, step: int, duration: int) -> float:
    """Return start + clip(step/duration,0,1)*(end-start), a scalar schedule."""
    if duration <= 0:
        raise ValueError("Schedule duration must be positive.")
    fraction = min(max(step / duration, 0.0), 1.0)
    return float(start + fraction * (end - start))


class DuelingDeepQNetwork(nn.Module):
    """Dueling convolutional Q network: Q(s,a)=V(s)+A(s,a)-mean_b A(s,b).

    Input is float32 pixels [B,4,84,84] on scale [0,255]; output is Q values
    [B,2]. The three convolutions match the original DeepQNetwork backbone.
    Each independent value/advantage stream has one hidden ReLU layer.
    """

    def __init__(self, hidden_size: int = 512):
        """Create two action outputs using a positive stream hidden_size."""
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive.")
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, 8, 4), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1), nn.ReLU(inplace=True))
        self.value = nn.Sequential(nn.Linear(7 * 7 * 64, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, 1))
        self.advantage = nn.Sequential(nn.Linear(7 * 7 * 64, hidden_size), nn.ReLU(),
                                       nn.Linear(hidden_size, 2))
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Map [B,4,84,84] pixels to centered-dueling Q values [B,2]."""
        features = self.conv3(self.conv2(self.conv1(pixels))).flatten(1)
        advantage = self.advantage(features)
        return self.value(features) + advantage - advantage.mean(dim=1, keepdim=True)


class DQNAgent:
    """One-step DQN agent, with selectable single-improvement ablations.

    ``variant`` is dqn, double_dqn, dueling_dqn or per_dqn. Double uses
    ``y=r+d*Q_target(s',argmax_a Q_online(s',a))``; others use target max.
    ``loss=mean_i w_i*(y_i-Q_online(s_i,a_i))**2``; w=1 without PER.
    Epsilon/PER beta use environment steps; target synchronization counts
    optimizer updates. State dicts omit replay and support model selection,
    evaluation and warm restart, not bit-exact interrupted-training resume.
    """

    VARIANTS = ("dqn", "double_dqn", "dueling_dqn", "per_dqn")

    def __init__(self, config: AgentConfig, device: str = "cpu", seed: int = 0,
                 variant: str = "dqn"):
        """Build networks/buffer from AgentConfig, device and local RNG seed.

        Torch initialization is controlled by the benchmark's shared seed
        utility. This constructor does not reset global random generators.
        """
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown DQN variant {variant!r}; choose {self.VARIANTS}.")
        self.config, self.device, self.variant = config, torch.device(device), variant
        self.rng = np.random.default_rng(seed)
        self.online = (DuelingDeepQNetwork(config.hidden_size) if variant == "dueling_dqn"
                       else DeepQNetwork()).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self._return_horizon = 1
        self.replay = self._make_replay()
        self.accumulator = NStepAccumulator(self._return_horizon, config.gamma)
        self.environment_steps = 0
        self.gradient_steps = 0
        self._last_update_step = -1

    def _make_replay(self) -> ReplayBuffer:
        """Create uniform/PER storage according to the agent's configured variant."""
        if self.variant in ("per_dqn", "rainbow"):
            return PrioritizedReplayBuffer(self.config.replay_capacity, self.config.per_alpha,
                                           self.config.per_epsilon, self.rng)
        return ReplayBuffer(self.config.replay_capacity, self.rng)

    @property
    def epsilon(self) -> float:
        """Current linearly annealed exploration probability in [0,1]."""
        return linear_schedule(self.config.epsilon_start, self.config.epsilon_end,
                               self.environment_steps, self.config.epsilon_decay_steps)

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """Return action 0/1 from uint8 [4,84,84]; explore=False is deterministic.

        Training is epsilon-greedy, ``a~Uniform({0,1})`` with probability epsilon;
        evaluation returns argmax Q, using the first action to break ties.
        """
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(2))
        with torch.no_grad():
            values = ensure_finite(self.online(observation_tensor(state, self.device)), "DQN action values")
            return int(values.argmax(dim=1).item())

    def observe(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, terminated: bool,
                truncated: bool = False) -> None:
        """Store one step; bootstrap on truncation, mask bootstrap on termination.

        State inputs are uint8 [4,84,84]; no tensor or GPU storage is retained.
        Increments the environment-step counter used by exploration/schedules.
        """
        for transition in self.accumulator.append(state, action, reward, next_state,
                                                  terminated, truncated):
            self.replay.add(transition)
        self.environment_steps += 1

    def _ready(self) -> bool:
        """Whether warmup, batch availability and one-update-per-step schedule hold."""
        return (self.environment_steps >= self.config.warmup_steps
                and len(self.replay) >= self.config.batch_size
                and self.environment_steps % self.config.train_frequency == 0
                and self._last_update_step != self.environment_steps)

    def update(self) -> Dict[str, float]:
        """Run a scheduled detached Bellman/MSE update; return diagnostics or {}.

        Returns loss, mean absolute TD error, Q mean, epsilon and beta. Target
        parameters never receive gradients. Gradient norm is clipped globally.
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
        self.online.train()
        predicted = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            target_values = self.target(next_states)
            if self.variant == "double_dqn":
                selected = self.online(next_states).argmax(dim=1, keepdim=True)
                successor_values = target_values.gather(1, selected).squeeze(1)
            else:
                successor_values = target_values.max(dim=1).values
            targets = rewards + discounts * successor_values
        td_error = targets - predicted
        loss = (weights * td_error.square()).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip,
                                                error_if_nonfinite=True)
        self.optimizer.step()
        self.replay.update_priorities(batch.indices, td_error.detach().abs().cpu().numpy())
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {"loss": float(loss.detach()), "td_error": float(td_error.detach().abs().mean()),
                "q_mean": float(predicted.detach().mean()), "epsilon": self.epsilon,
                "beta": beta, "gradient_norm": float(gradient_norm)}

    def finish(self) -> Dict[str, float]:
        """Flush any pending returns at the training budget; no optimizer update."""
        for transition in self.accumulator.flush():
            self.replay.add(transition)
        return {}

    def state_dict(self) -> dict:
        """Return selection checkpoint with networks/optimizer/RNG, excluding replay.

        ``online`` uses unchanged original model keys for dqn/double_dqn/per_dqn.
        Call torch.save immediately or deepcopy if keeping an in-memory snapshot.
        """
        return {"algorithm": self.variant, "online": self.online.state_dict(),
                "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                "environment_steps": self.environment_steps, "gradient_steps": self.gradient_steps,
                "rng_state": copy.deepcopy(self.rng.bit_generator.state),
                "replay_included": False}

    def load_state_dict(self, payload: dict) -> None:
        """Restore a selection checkpoint or a bare original model state dictionary.

        A checkpoint from another named variant raises ValueError rather than
        silently using the wrong target rule. Replay/pending returns are reset.
        """
        if "algorithm" in payload and payload["algorithm"] != self.variant:
            raise ValueError(f"Cannot load {payload['algorithm']!r} as {self.variant!r}.")
        if "online" not in payload:
            self.online.load_state_dict(payload)
            self.target.load_state_dict(payload)
        else:
            self.online.load_state_dict(payload["online"])
            self.target.load_state_dict(payload.get("target", payload["online"]))
            if "optimizer" in payload:
                self.optimizer.load_state_dict(payload["optimizer"])
            self.environment_steps = int(payload.get("environment_steps", 0))
            self.gradient_steps = int(payload.get("gradient_steps", 0))
            if "rng_state" in payload:
                self.rng.bit_generator.state = copy.deepcopy(payload["rng_state"])
        self.target.eval()
        self.target.requires_grad_(False)
        self.replay = self._make_replay()
        self.accumulator = NStepAccumulator(self._return_horizon, self.config.gamma)
        self._last_update_step = -1
