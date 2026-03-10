"""
baselines.py
------------
Baseline market-making strategies for benchmarking the DQN agent.

Three baselines are implemented:

1. FixedSpreadAgent
   Always quotes at a fixed symmetric offset. This represents the
   simplest possible market-making policy: pick a spread, stick with it.
   It serves as a floor — a learned policy should do at least this well.

2. RandomAgent
   Uniformly samples a random action at every step. This represents the
   performance of a completely uninformed agent and provides a lower bound
   on what any reasonable policy should achieve.

3. InventorySkewAgent
   Inventory-aware heuristic: when inventory is positive (long), the agent
   tightens the ask to encourage offloading; when inventory is negative
   (short), it tightens the bid. The idea is to lean against imbalanced
   inventory without complex optimization.

   This heuristic is inspired by classical market-making theory:
   market makers naturally skew quotes toward the direction that reduces
   their inventory risk. It is a reasonable approximation of what a
   rule-of-thumb manual trader might do.

All agents expose a common `select_action(state, env)` interface.
"""

from __future__ import annotations

import numpy as np


class FixedSpreadAgent:
    """
    Always quotes at a fixed offset tier.

    Parameters
    ----------
    action_index:
        Index into quote_offsets list. E.g. 1 = second-tightest spread.
    """

    name: str = "FixedSpread"

    def __init__(self, action_index: int = 1) -> None:
        self._action = action_index

    def select_action(self, state: np.ndarray, n_actions: int) -> int:
        """Return the fixed action regardless of state."""
        return min(self._action, n_actions - 1)

    def reset(self) -> None:
        pass


class RandomAgent:
    """
    Uniformly random action selection.

    This is the simplest imaginable baseline and should be beaten easily.
    """

    name: str = "Random"

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng()

    def select_action(self, state: np.ndarray, n_actions: int) -> int:
        """Sample a uniformly random action."""
        return int(self._rng.integers(0, n_actions))

    def reset(self) -> None:
        pass


class InventorySkewAgent:
    """
    Inventory-aware heuristic quoting agent.

    Logic:
      - Compute a base action index (middle of the action space by default).
      - Adjust the action based on current inventory:
          * High positive inventory → use a wider bid and tighter ask tier.
            This makes it more likely the ask fills (offloading inventory).
          * High negative inventory → use a tighter bid and wider ask tier.
            This makes it more likely the bid fills (building inventory back).
      - Clip to valid action range.

    The state vector encodes normalised inventory at index 0 (see state_builder.py).

    Parameters
    ----------
    skew_sensitivity:
        How aggressively to skew based on inventory. Higher = more reactive.
    base_action:
        Default action when inventory is zero.
    """

    name: str = "InventorySkew"

    def __init__(
        self,
        skew_sensitivity: float = 2.0,
        base_action: int = 1,
    ) -> None:
        self._sensitivity = skew_sensitivity
        self._base = base_action

    def select_action(self, state: np.ndarray, n_actions: int) -> int:
        """
        Compute skewed action from inventory state feature.

        state[0] is normalised inventory in [-1, 1].
        Positive inventory → shift toward higher action index (wider spreads on bid side).
        Actually, for a symmetric env, we can't independently skew bid/ask.
        Strategy: when long, use wider offset (earn more per fill, wait longer to offload).
        When flat or short, use tighter offset (fill faster, rebuild inventory).
        """
        norm_inv = float(state[0])  # ∈ [-1, 1]

        # Skew: long inventory → prefer wider quote (reduce fill rate on bid)
        # This reduces position accumulation on the long side.
        delta = int(round(self._sensitivity * norm_inv))
        action = np.clip(self._base + delta, 0, n_actions - 1)
        return int(action)

    def reset(self) -> None:
        pass


# Type alias for any baseline agent
BaselineAgent = FixedSpreadAgent | RandomAgent | InventorySkewAgent

BASELINE_REGISTRY: dict[str, type] = {
    "FixedSpread": FixedSpreadAgent,
    "Random": RandomAgent,
    "InventorySkew": InventorySkewAgent,
}


def get_all_baselines() -> list[BaselineAgent]:
    """Return one instance of each baseline agent."""
    return [
        FixedSpreadAgent(action_index=1),
        RandomAgent(),
        InventorySkewAgent(skew_sensitivity=2.0, base_action=1),
    ]
