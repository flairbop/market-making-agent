"""
replay_buffer.py
----------------
Experience replay buffer for DQN training.

The replay buffer stores (state, action, reward, next_state, done) tuples
collected during environment interaction. During training, minibatches are
sampled uniformly at random to break temporal correlation between updates.

This is a fixed-capacity circular buffer implemented as a deque. For a
production system one would use a prioritised experience replay (PER) buffer,
but uniform replay is standard for clean DQN baselines.

Memory consumption estimate:
  - State dim 8 floats each × 2 (state + next_state) = 16 floats × 4 bytes
  - Plus action (int4) + reward (float4) + done (bool1) ≈ ~70 bytes/transition
  - 100,000 transitions ≈ ~7 MB — easily fits in RAM.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class Transition:
    """One stored experience transition."""

    state: np.ndarray       # shape (state_dim,)
    action: int
    reward: float
    next_state: np.ndarray  # shape (state_dim,)
    done: bool              # True if episode ended


class ReplayBuffer:
    """
    Uniform experience replay buffer.

    Parameters
    ----------
    capacity:
        Maximum number of transitions stored. Oldest are evicted once full.
    """

    def __init__(self, capacity: int) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add one transition to the buffer."""
        self._buffer.append(
            Transition(
                state=state.astype(np.float32),
                action=action,
                reward=float(reward),
                next_state=next_state.astype(np.float32),
                done=done,
            )
        )

    def sample(
        self, batch_size: int, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random minibatch.

        Returns
        -------
        states, actions, rewards, next_states, dones — all as torch tensors.
        """
        batch: List[Transition] = random.sample(self._buffer, batch_size)

        states = torch.tensor(
            np.stack([t.state for t in batch]), dtype=torch.float32, device=device
        )
        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.long, device=device
        )
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=device
        )
        next_states = torch.tensor(
            np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device
        )
        dones = torch.tensor(
            [t.done for t in batch], dtype=torch.float32, device=device
        )
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """True once the buffer has at least ``batch_size`` transitions."""
        return len(self._buffer) >= batch_size
