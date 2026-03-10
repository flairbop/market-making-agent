"""
networks.py
-----------
Neural network architectures for the DQN agent.

We implement two variants:
  1. MLP Q-Network: standard feedforward net Q(s) → R^|A|
  2. Dueling Q-Network: separate value V(s) and advantage A(s,a) streams.
     Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)]
     This helps the network learn that state value is often decoupled from
     action advantage, especially when most actions are similarly valued.

Both share the same interface: forward(state) → Q-values for all actions.

Architecture choice:
  A modest 2-layer MLP with 128 units each is used by default.
  The market-making state space is low-dimensional (8 features), so deep
  or convolutional architectures add complexity without benefit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPQNetwork(nn.Module):
    """
    Standard multi-layer perceptron Q-network.

    Output shape: (batch_size, n_actions) — Q-values for each discrete action.

    Parameters
    ----------
    state_dim:
        Dimension of the input state vector.
    n_actions:
        Number of discrete actions.
    hidden_layers:
        List of hidden layer widths, e.g. [128, 128].
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [128, 128]
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture (Wang et al., 2016).

    Decomposes Q into two streams:
      - Value stream: V(s) → scalar
      - Advantage stream: A(s, a) → R^|A|
    Recombined: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)

    Parameters
    ----------
    state_dim, n_actions, hidden_layers:
        Same as MLPQNetwork.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [128, 128]

        # Shared trunk
        trunk_layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_layers[:-1]:
            trunk_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)
        final_h = hidden_layers[-1]

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, final_h),
            nn.ReLU(),
            nn.Linear(final_h, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, final_h),
            nn.ReLU(),
            nn.Linear(final_h, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        # Combine: subtract mean advantage → identifiability
        q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q


def build_network(
    state_dim: int,
    n_actions: int,
    hidden_layers: list[int],
    dueling: bool = True,
) -> nn.Module:
    """
    Factory function for Q-networks.

    Parameters
    ----------
    dueling:
        If True, build DuelingQNetwork; otherwise MLPQNetwork.
    """
    if dueling:
        return DuelingQNetwork(state_dim, n_actions, hidden_layers)
    return MLPQNetwork(state_dim, n_actions, hidden_layers)
