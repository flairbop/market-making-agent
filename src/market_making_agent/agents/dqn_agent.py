"""
dqn_agent.py
------------
Deep Q-Network (DQN) agent with optional Double DQN support.

Algorithm summary (Double DQN):
  1. Maintain online network Q_θ and target network Q_θ⁻.
  2. At each step, act ε-greedily using Q_θ.
  3. Store (s, a, r, s', done) in replay buffer.
  4. Once buffer is ready, sample a minibatch and compute TD targets:
       Non-terminal: y = r + γ * Q_θ⁻(s', argmax_a Q_θ(s', a))
       Terminal:     y = r
  5. Minimise MSE loss: L = (Q_θ(s,a) - y)²
  6. Every target_update_freq steps, copy θ → θ⁻.

Double DQN (Van Hasselt et al., 2016) decouples action selection from
evaluation, reducing the maximisation bias present in vanilla DQN.

Epsilon-greedy exploration:
  ε linearly decays from epsilon_start → epsilon_end over epsilon_decay_steps.
  This balances exploration early in training with exploitation later.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from market_making_agent.agents.networks import build_network
from market_making_agent.agents.replay_buffer import ReplayBuffer
from market_making_agent.config import Config


class DQNAgent:
    """
    DQN agent with optional Double DQN and configurable exploration.

    Parameters
    ----------
    cfg:
        Full configuration object.
    state_dim:
        Observation space dimension.
    n_actions:
        Number of discrete actions.
    device:
        PyTorch device string, e.g. "cpu" or "cuda".
    """

    def __init__(
        self,
        cfg: Config,
        state_dim: int,
        n_actions: int,
        device: str = "cpu",
    ) -> None:
        self._cfg = cfg.agent
        self._n_actions = n_actions
        self._device = device
        self._double_dqn: bool = self._cfg.double_dqn

        # Online and target networks
        self._online_net = build_network(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_layers=list(self._cfg.hidden_layers),
            dueling=self._cfg.dueling,
        ).to(device)

        self._target_net = build_network(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_layers=list(self._cfg.hidden_layers),
            dueling=self._cfg.dueling,
        ).to(device)

        # Initialise target network as a copy of online network
        self._target_net.load_state_dict(self._online_net.state_dict())
        self._target_net.eval()  # Target net never trains

        self._optimizer = optim.Adam(
            self._online_net.parameters(), lr=self._cfg.learning_rate
        )
        self._loss_fn = nn.MSELoss()

        self._replay_buffer = ReplayBuffer(capacity=int(self._cfg.replay_buffer_capacity))

        # Exploration
        self._epsilon: float = self._cfg.epsilon_start
        self._epsilon_end: float = self._cfg.epsilon_end
        self._epsilon_decay_steps: int = int(self._cfg.epsilon_decay_steps)

        # Step counter
        self._steps_done: int = 0
        self._update_count: int = 0

        # Training stats
        self._last_loss: float = 0.0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        ε-greedy action selection.

        Parameters
        ----------
        state:
            Current observation vector.
        deterministic:
            If True, always pick the greedy action (no exploration).
            Used during evaluation.

        Returns
        -------
        int
            Selected action index.
        """
        if not deterministic and np.random.random() < self._epsilon:
            return int(np.random.randint(0, self._n_actions))

        state_t = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            q_values = self._online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def _update_epsilon(self) -> None:
        """Linearly decay epsilon from start to end over decay_steps."""
        fraction = min(1.0, self._steps_done / self._epsilon_decay_steps)
        self._epsilon = self._cfg.epsilon_start + fraction * (
            self._cfg.epsilon_end - self._cfg.epsilon_start
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push one transition to the replay buffer."""
        self._replay_buffer.push(state, action, reward, next_state, done)
        self._steps_done += 1
        self._update_epsilon()

    def train_step(self) -> Optional[float]:
        """
        Sample a minibatch and perform one gradient update.

        Returns
        -------
        float or None
            Loss value, or None if buffer is not yet ready.
        """
        batch_size = int(self._cfg.batch_size)
        if not self._replay_buffer.is_ready(batch_size):
            return None

        states, actions, rewards, next_states, dones = self._replay_buffer.sample(
            batch_size, device=self._device
        )

        # --- Compute Q(s, a) using online net ---
        q_values = self._online_net(states)  # (B, A)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # --- Compute target Q-values ---
        with torch.no_grad():
            if self._double_dqn:
                # Double DQN: online net selects action, target net evaluates
                next_actions = self._online_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self._target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Vanilla DQN: target net is used for both selection and evaluation
                next_q = self._target_net(next_states).max(dim=1).values

            # Terminal states have zero future value
            target_q = rewards + self._cfg.gamma * next_q * (1.0 - dones)

        # --- MSE loss ---
        loss = self._loss_fn(current_q, target_q)

        # --- Gradient update ---
        self._optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability (large |reward| can cause large gradients)
        nn.utils.clip_grad_norm_(self._online_net.parameters(), self._cfg.gradient_clip)
        self._optimizer.step()

        self._update_count += 1
        self._last_loss = float(loss.item())
        return self._last_loss

    def sync_target_network(self) -> None:
        """Hard-copy online network weights to target network."""
        self._target_net.load_state_dict(self._online_net.state_dict())

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def get_checkpoint(self, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Build a checkpoint dictionary for saving.

        Includes model state, optimizer state, and training metadata.
        """
        ckpt: dict[str, Any] = {
            "model_state_dict": self._online_net.state_dict(),
            "target_state_dict": self._target_net.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "steps_done": self._steps_done,
            "epsilon": self._epsilon,
            "update_count": self._update_count,
        }
        if extra:
            ckpt.update(extra)
        return ckpt

    def load_checkpoint(self, ckpt: dict[str, Any]) -> None:
        """Restore agent state from a saved checkpoint."""
        self._online_net.load_state_dict(ckpt["model_state_dict"])
        self._target_net.load_state_dict(ckpt.get("target_state_dict", ckpt["model_state_dict"]))
        self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._steps_done = ckpt.get("steps_done", 0)
        self._epsilon = ckpt.get("epsilon", self._cfg.epsilon_end)
        self._update_count = ckpt.get("update_count", 0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def steps_done(self) -> int:
        return self._steps_done

    @property
    def last_loss(self) -> float:
        return self._last_loss

    @property
    def buffer_size(self) -> int:
        return len(self._replay_buffer)
