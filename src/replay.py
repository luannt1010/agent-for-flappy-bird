"""CPU replay buffers and episode-aware n-step returns.

States stay uint8 on the CPU until sampled. For a transition of length k,
``reward = sum(gamma**j * r[t+j], j=0..k-1)`` and ``discount = gamma**k``;
the discount is zero only for a true MDP terminal. Time-limit truncations
flush the pending trajectory but retain bootstrapping from the final state.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np


@dataclass(frozen=True)
class Transition:
    """One replay item: uint8 states, action, discounted return and bootstrap factor.

    ``target = reward + discount * V(next_state)``. ``discount`` already includes
    the terminal mask; ``steps`` records the actual horizon used for the return.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    discount: float
    steps: int = 1


@dataclass(frozen=True)
class ReplayBatch:
    """Sampled arrays: B states/actions/returns/successors/discounts and PER weights.

    ``states`` and ``next_states`` have shape [B,4,84,84], dtype uint8. Actions
    and indices are int64 [B]; rewards/discounts/weights are float32 [B].
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    discounts: np.ndarray
    weights: np.ndarray
    indices: np.ndarray


def _copy_state(state: np.ndarray) -> np.ndarray:
    """Validate/copy one [4,84,84] uint8 observation to immutable replay storage."""
    array = np.asarray(state)
    if array.shape != (4, 84, 84) or array.dtype != np.uint8:
        raise ValueError("Replay observations must be uint8 arrays of shape (4, 84, 84).")
    return np.array(array, copy=True, order="C")


class NStepAccumulator:
    """Build n-step Transition objects without crossing episode boundaries.

    Input to :meth:`append` is a single environment step. Output is zero or more
    transitions with ``R = sum_{j=0}^{k-1} gamma**j * r_j`` and bootstrap factor
    ``gamma**k * (1 - terminated)``. A truncated episode uses the supplied final
    observation and preserves the bootstrap term, as required for time limits.
    """

    def __init__(self, n_step: int, gamma: float):
        """Set positive return horizon n_step and discount gamma in [0,1]."""
        if n_step < 1 or not 0.0 <= gamma <= 1.0:
            raise ValueError("n_step must be positive and gamma must be in [0, 1].")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self._pending: Deque[tuple] = deque()

    def append(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, terminated: bool,
               truncated: bool = False) -> List[Transition]:
        """Consume a step and return completed n-step items, flushing at episode end.

        State inputs are uint8 [4,84,84]; action is 0 or 1. On termination or
        truncation every remaining prefix is emitted with its shorter horizon.
        """
        if int(action) not in (0, 1):
            raise ValueError("Flappy Bird action must be 0 or 1.")
        self._pending.append((_copy_state(state), int(action), float(reward),
                              _copy_state(next_state), bool(terminated)))
        if terminated or truncated:
            return self.flush()
        if len(self._pending) >= self.n_step:
            return [self._pop_transition()]
        return []

    def _pop_transition(self) -> Transition:
        """Emit the oldest prefix and advance it by one environment step."""
        reward, discount, count = 0.0, 1.0, 0
        next_state = self._pending[0][3]
        for _, _, step_reward, successor, terminated in self._pending:
            reward += discount * step_reward
            discount *= self.gamma
            count += 1
            next_state = successor
            if terminated:
                discount = 0.0
                break
            if count >= self.n_step:
                break
        state, action, _, _, _ = self._pending.popleft()
        return Transition(state, action, reward, next_state, discount, count)

    def flush(self) -> List[Transition]:
        """Return all pending prefixes at a budget cutoff, bootstrapping if nonterminal.

        Call this at the end of training to avoid silently losing fewer than n
        final steps. This does not invent a terminal state or a zero bootstrap.
        """
        output = []
        while self._pending:
            output.append(self._pop_transition())
        return output


class ReplayBuffer:
    """Bounded FIFO replay with uniform sampling and compact uint8 observations.

    Storage grows lazily to capacity; observations never occupy GPU memory.
    A capacity of N requires at most ``2*N*4*84*84`` observation bytes. Sampling
    is with replacement, giving each transition exactly probability ``1/N``.
    """

    def __init__(self, capacity: int, rng: Optional[np.random.Generator] = None):
        """Create a positive-capacity buffer with an optional seeded local RNG."""
        if capacity < 1:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = int(capacity)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._items: List[Transition] = []
        self._position = 0

    def __len__(self) -> int:
        """Return the number of valid transitions currently stored."""
        return len(self._items)

    def add(self, transition: Transition) -> int:
        """Copy a transition into FIFO storage and return its integer slot index."""
        copied = Transition(_copy_state(transition.state), int(transition.action),
                            float(transition.reward), _copy_state(transition.next_state),
                            float(transition.discount), int(transition.steps))
        index = self._position
        if len(self._items) < self.capacity:
            self._items.append(copied)
        else:
            self._items[index] = copied
        self._position = (self._position + 1) % self.capacity
        return index

    def _batch(self, indices: np.ndarray, weights: np.ndarray) -> ReplayBatch:
        """Gather CPU transitions at indices into a batch of stacked NumPy arrays."""
        items = [self._items[int(index)] for index in indices]
        return ReplayBatch(
            np.stack([item.state for item in items]),
            np.asarray([item.action for item in items], dtype=np.int64),
            np.asarray([item.reward for item in items], dtype=np.float32),
            np.stack([item.next_state for item in items]),
            np.asarray([item.discount for item in items], dtype=np.float32),
            np.asarray(weights, dtype=np.float32), np.asarray(indices, dtype=np.int64),
        )

    def sample(self, batch_size: int, beta: float = 1.0) -> ReplayBatch:
        """Draw batch_size independent uniform samples; importance weights are one.

        ``beta`` is accepted for compatibility with prioritized replay and is
        unused. Returns ReplayBatch; an empty buffer or invalid size raises.
        """
        if not self._items or batch_size < 1:
            raise ValueError("Sampling needs a nonempty replay and a positive batch size.")
        indices = self.rng.integers(0, len(self), size=batch_size)
        return self._batch(indices, np.ones(batch_size, dtype=np.float32))

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Uniform replay ignores priority updates; provided for a shared interface."""


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional PER with O(log N) sum/min trees (Schaul et al., 2015).

    ``P(i) = p_i**alpha / sum_j(p_j**alpha)`` and
    ``w_i = (N*P(i))**(-beta) / max_j (N*P(j))**(-beta)``. The maximum is over
    all valid items, not only the sampled minibatch. Callers supply nonnegative
    TD errors or KL losses; epsilon is added here exactly once.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-6,
                 rng: Optional[np.random.Generator] = None):
        """Set capacity, priority exponent alpha >= 0, epsilon > 0 and local RNG."""
        super().__init__(capacity, rng)
        if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("PER needs finite alpha >= 0 and epsilon > 0.")
        self.alpha, self.epsilon = float(alpha), float(epsilon)
        self._tree_capacity = 1 << (capacity - 1).bit_length()
        self._sum = np.zeros(2 * self._tree_capacity, dtype=np.float64)
        self._min = np.full(2 * self._tree_capacity, np.inf, dtype=np.float64)
        self._max_priority = 1.0

    def _set_priority(self, index: int, priority: float) -> None:
        """Set one positive raw priority and propagate its alpha power to the root."""
        node = self._tree_capacity + index
        self._sum[node] = priority ** self.alpha
        self._min[node] = self._sum[node]
        node //= 2
        while node:
            self._sum[node] = self._sum[2 * node] + self._sum[2 * node + 1]
            self._min[node] = min(self._min[2 * node], self._min[2 * node + 1])
            node //= 2

    def add(self, transition: Transition) -> int:
        """Insert an item with the largest priority observed so every new item is seen."""
        index = super().add(transition)
        self._set_priority(index, self._max_priority)
        return index

    def sample(self, batch_size: int, beta: float = 0.4) -> ReplayBatch:
        """Draw independent proportional samples and normalized importance weights.

        Input beta in [0,1] corrects sampling bias; beta=1 is full correction.
        Return ReplayBatch with possibly repeated indices (sampling replacement).
        """
        if not self._items or batch_size < 1 or not 0.0 <= beta <= 1.0:
            raise ValueError("PER sampling needs data, positive batch size and beta in [0,1].")
        total = self._sum[1]
        masses = self.rng.random(batch_size) * total
        nodes = np.ones(batch_size, dtype=np.int64)
        while np.any(nodes < self._tree_capacity):
            left = 2 * nodes
            go_right = masses >= self._sum[left]
            masses = masses - np.where(go_right, self._sum[left], 0.0)
            nodes = left + go_right.astype(np.int64)
        indices = nodes - self._tree_capacity
        probabilities = self._sum[nodes] / total
        min_probability = self._min[1] / total
        # Algebraically equal to (N*P)^-beta / (N*P_min)^-beta, more stable.
        weights = (probabilities / min_probability) ** (-beta)
        return self._batch(indices, weights)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Set priorities from abs(TD error) or KL divergence, adding epsilon once.

        Inputs are equal-length arrays. Duplicate slots receive their largest
        supplied error to make updates independent of minibatch ordering.
        """
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        errors = np.asarray(priorities, dtype=np.float64).reshape(-1)
        if len(indices) != len(errors) or np.any(indices < 0) or np.any(indices >= len(self)):
            raise ValueError("Priority indices must refer to valid items and match priorities.")
        if np.any(~np.isfinite(errors)) or np.any(errors < 0):
            raise ValueError("Priorities must be finite and nonnegative.")
        for index in np.unique(indices):
            priority = float(errors[indices == index].max()) + self.epsilon
            self._max_priority = max(self._max_priority, priority)
            self._set_priority(int(index), priority)
