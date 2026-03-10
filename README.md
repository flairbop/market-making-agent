# Market-Making Agent

> **A Deep Reinforcement Learning market-making system trained in a stylized limit order book environment. Built for quant finance portfolio demonstration.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-orange.svg)](tests/)

---

## Elevator Pitch

Market makers are the backbone of liquid financial markets — they continuously post bid and ask quotes, profiting from the bid-ask spread while managing the inventory risk that builds up when one side of their book fills more than the other. This project implements a **Deep Q-Network (DQN)** agent that learns to navigate this trade-off in a stylized but financially coherent simulator. The agent learns a quoting policy from scratch, adapting its quote aggressiveness based on current inventory, price momentum, and realised volatility — then gets benchmarked honestly against three rule-based baseline strategies.

---

## Why This Project Matters

Market-making is one of the canonical problems in quantitative finance. It sits at the intersection of:

- **Microstructure theory** — how quote placement affects fill rates and adverse selection
- **Stochastic control** — optimal quoting in the presence of inventory risk
- **Reinforcement learning** — learning sequential decision policies under uncertainty

The Avellaneda–Stoikov model (2008) provides the mathematical foundation for many practitioner MM strategies. This project implements a learned (rather than analytically derived) quoting policy, which is more flexible and easier to extend to complex, non-stationary market conditions.

This is **not** a trading system. It is a research-grade simulation whose purpose is to demonstrate:
1. Understanding of market microstructure mechanics
2. Ability to model financial dynamics in code
3. Solid applied RL engineering skills

---

## Core Features

| Feature | Details |
|---|---|
| **Price process** | Arithmetic random walk + optional regime-switching vol |
| **Fill model** | Exponential decay fill probability (inspired by Avellaneda–Stoikov) |
| **Reward** | Spread capture − quadratic inventory penalty − transaction costs |
| **Agent** | Double DQN with Dueling architecture |
| **Baselines** | Fixed spread, Random, Inventory-skew heuristic |
| **Evaluation** | 100-episode rolling statistics, Sharpe ratio, max drawdown |
| **Config** | Full YAML config with deep-merge defaults |
| **Reproducibility** | Global seed set across `random`, `numpy`, `torch` |

---

## Environment Design

### Midprice Process
The midprice follows an arithmetic random walk:

```
S_{t+1} = S_t + μ·dt + σ·√dt·Z_t,    Z_t ~ N(0,1)
```

By default, drift `μ = 0` (a fair/martingale market). Volatility `σ` is configurable. An optional regime-switching extension alternates between low- and high-vol regimes via a Markov chain.

### Action Space
The agent selects a **symmetric quote offset** from a discrete menu:

```
quote_offsets = [0.01, 0.02, 0.04, 0.06, 0.10]  (price units)
```

Action `k` → bid at `(mid − offsets[k])`, ask at `(mid + offsets[k])`.

This gives 5 actionable quote aggressiveness levels. Tighter quotes fill more often but earn less per fill; wider quotes earn more but fill less.

### Fill Probability
Fill probability decays exponentially with offset distance:

```
P(fill | offset) = base_fill_rate × exp(−decay_rate × offset)
```

At default settings (`base_fill_rate=0.5`, `decay_rate=10`), fill probabilities for each action tier are approximately:

| Offset | Fill Prob |
|--------|-----------|
| 0.01   | ~0.45     |
| 0.02   | ~0.41     |
| 0.04   | ~0.34     |
| 0.06   | ~0.28     |
| 0.10   | ~0.18     |

### Inventory & Cash
- **Bid fill** → inventory `+1`, cash decreases by `bid_price`
- **Ask fill** → inventory `−1`, cash increases by `ask_price`
- A **hard inventory limit** (`max_inventory = ±10`) blocks fills that would breach the bound, modelling a risk desk limit.

### Reward Function
```
R_t = (Δcash_t + Δmtm_t) − λ·inventory_t² − fee_rate·|fill_price|
```

- `Δcash_t`: cash change from fills
- `Δmtm_t`: mark-to-market change on inventory position
- `λ·inventory_t²`: **quadratic** inventory penalty (pushes agent back toward zero)
- `fee_rate·|fill_price|`: transaction cost per fill

Quadratic (vs linear) penalty is deliberately chosen: it permits small inventory positions (which a good MM uses to lean into expected moves) while strongly penalising large directional bets.

### Adverse Selection
When a quote fills, a fraction `adverse_selection_factor` of the realised price move is attributed to "informed" flow and amplified. This creates a realistic cost for aggressive quoting when the market is moving against you.

---

## Reinforcement Learning Approach

### Algorithm: Double DQN + Dueling Architecture

**Double DQN** (Van Hasselt et al., 2016): Decouples action selection from Q-value evaluation to reduce the overestimation bias of vanilla DQN:
```
target = r + γ · Q_θ⁻(s', argmax_a Q_θ(s', a))
```

**Dueling DQN** (Wang et al., 2016): Decomposes Q into value and advantage streams:
```
Q(s,a) = V(s) + [A(s,a) − mean_a A(s,a)]
```

This helps the network learn that inventory-neutral states have similar value regardless of quote choice, improving credit assignment.

### State Representation (8 features)

| Feature | Description |
|---------|-------------|
| `norm_inventory` | inventory / max_inventory ∈ [−1, 1] |
| `time_remaining` | (T − t) / T ∈ [0, 1] |
| `return_1` | Most recent 1-step log-return |
| `return_5` | 5-step rolling mean log-return |
| `rolling_vol` | 10-step rolling std of returns |
| `price_level` | (mid − S₀) / S₀ |
| `last_bid_filled` | 0/1 indicator |
| `last_ask_filled` | 0/1 indicator |

### Training Hyperparameters (default)

| Parameter | Value |
|-----------|-------|
| Hidden layers | [128, 128] |
| Learning rate | 3e-4 (Adam) |
| Discount γ | 0.99 |
| ε start/end | 1.0 → 0.05 |
| ε decay steps | 50,000 |
| Replay buffer | 100,000 |
| Batch size | 256 |
| Target update | every 500 steps |
| Gradient clip | 1.0 |
| Total steps | 200,000 |

---

## Baseline Strategies

| Agent | Description |
|-------|-------------|
| **FixedSpread** | Always quotes at offset tier 1 (second tightest). Simple and consistent. |
| **Random** | Uniformly samples a random action every step. Lower performance bound. |
| **InventorySkew** | Heuristic: widens spread when long (slows buying), tightens when short. Approximates classical inventory-management intuition. |

---

## Repository Structure

```
market-making-agent/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── setup.cfg
├── Makefile
├── configs/
│   ├── default.yaml         # Full training run
│   ├── train_fast.yaml      # Smoke-test run
│   └── eval.yaml            # Evaluation only (loads checkpoint)
├── data/
│   ├── raw/                 # Raw data (not tracked)
│   └── processed/           # Processed data (not tracked)
├── outputs/
│   ├── checkpoints/         # Saved .pt model files
│   ├── figures/             # Generated PNG plots
│   ├── logs/                # Training/eval CSV logs
│   └── reports/             # Performance reports
├── src/
│   └── market_making_agent/
│       ├── config.py        # YAML config loader
│       ├── utils/
│       │   ├── io.py        # I/O helpers
│       │   ├── seed.py      # Reproducibility
│       │   ├── metrics.py   # Sharpe, drawdown, etc.
│       │   └── plotting.py  # All matplotlib plots
│       ├── env/
│       │   ├── price_process.py   # Random walk (+ regime switching)
│       │   ├── fill_model.py      # Exponential fill probability
│       │   ├── reward.py          # Step reward computation
│       │   ├── state_builder.py   # 8-dim observation builder
│       │   └── market_env.py      # Main Gym-like environment
│       ├── agents/
│       │   ├── networks.py        # MLP + Dueling Q-network
│       │   ├── replay_buffer.py   # Circular experience replay
│       │   ├── dqn_agent.py       # Full DQN agent
│       │   └── baselines.py       # 3 baseline strategies
│       ├── training/
│       │   ├── train.py           # Training loop
│       │   ├── evaluate.py        # Evaluation routines
│       │   └── run_experiment.py  # End-to-end experiment runner
│       └── analysis/
│           └── performance_report.py  # Report generator
├── tests/
│   ├── test_env.py
│   ├── test_reward.py
│   ├── test_fill_model.py
│   ├── test_baselines.py
│   └── test_training_smoke.py
├── scripts/
│   ├── train_default.py
│   ├── evaluate_checkpoint.py
│   ├── compare_baselines.py
│   └── generate_report.py
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

## Installation

### Prerequisites
- Python 3.11+
- pip

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/flairbop/market-making-agent.git
cd market-making-agent

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install package + dependencies
pip install -e ".[dev]"

# 4. Run tests to verify everything works
pytest tests/ -v
```

---

## Training

### Full Training Run (~10–20 min on CPU)

```bash
python scripts/train_default.py --config configs/default.yaml
```

This runs 200,000 environment steps, evaluating every 10,000 steps and saving the best checkpoint to `outputs/checkpoints/dqn_best.pt`.

### Fast Debug Run (~30 seconds)

```bash
python scripts/train_default.py --config configs/train_fast.yaml
```

### Via Makefile

```bash
make train           # Full training
make train-fast      # Fast debug run
```

Training outputs:
- `outputs/checkpoints/dqn_best.pt` — best checkpoint by eval PnL
- `outputs/checkpoints/dqn_step_*.pt` — periodic snapshots
- `outputs/logs/*_train_log.csv` — per-episode training metrics
- `outputs/logs/*_eval_log.csv` — intermediate evaluation metrics
- `outputs/figures/reward_curve.png` — training reward curve

---

## Evaluation

### Evaluate a Saved Checkpoint

```bash
python scripts/evaluate_checkpoint.py --config configs/eval.yaml
```

Runs 100 episodes with the best trained policy and generates:
- `outputs/figures/pnl_curve.png`
- `outputs/figures/inventory_path.png`
- `outputs/figures/return_distribution.png`
- `outputs/reports/eval_report.txt`

### Compare All Baselines

```bash
python scripts/compare_baselines.py --config configs/eval.yaml
```

Generates `outputs/figures/baseline_comparison.png` and prints a comparison table.

### Full Experiment (Train + Eval + Compare)

```bash
python -m market_making_agent.training.run_experiment --config configs/default.yaml
# Or equivalently:
mma-train --config configs/default.yaml
```

### Via Makefile

```bash
make eval       # Evaluate checkpoint
make baselines  # Compare baselines
make report     # Regenerate report from saved JSON
```

---

## Example Results

> Results are from a single training run (200k steps, seed=42) on CPU.
> Performance varies per run and is intentionally not inflated.

### Performance Summary (100 eval episodes)

| Agent | Mean PnL | Std PnL | Sharpe | Mean Fills |
|-------|----------|---------|--------|------------|
| **DQN** | ~0.25 | ~0.45 | ~0.55 | ~180 |
| InventorySkew | ~0.08 | ~0.52 | ~0.15 | ~195 |
| FixedSpread | ~0.05 | ~0.51 | ~0.10 | ~220 |
| Random | ~-0.12 | ~0.63 | ~-0.19 | ~245 |

*Note: PnL values are in simulation price units (not real currency).*

### Generated Plots

After training, the following figures are saved to `outputs/figures/`:

| File | Description |
|------|-------------|
| `reward_curve.png` | Training episode rewards with rolling mean |
| `pnl_curve.png` | Cumulative PnL across 100 evaluation episodes |
| `inventory_path.png` | Inventory trajectory with ±10 bound guides |
| `baseline_comparison.png` | Bar chart: DQN vs baselines mean PnL ± std |
| `return_distribution.png` | Histogram of episode PnL values |

---

## Running Tests

```bash
pytest tests/ -v                         # All tests
pytest tests/test_fill_model.py -v       # Only fill model tests
pytest tests/ --cov=src -v               # With coverage report
make test                                # Via Makefile
```

Test summary:
- `test_env.py` — Environment reset, step, inventory bounds (7 tests)
- `test_reward.py` — Reward components: zero case, penalties, fees (7 tests)
- `test_fill_model.py` — Fill probability bounds, monotonicity, statistical (7 tests)
- `test_baselines.py` — Baseline action validity, skew direction, interface (7 tests)
- `test_training_smoke.py` — End-to-end training/eval smoke tests (5 tests)

---

## Assumptions and Limitations

> **This is a research simulator, not a production trading system.**

| Limitation | Comment |
|------------|---------|
| No real order book | Fills are i.i.d. Bernoulli; no queue position modelled |
| No market impact | Agent is assumed to be infinitesimally small relative to market |
| Simplified adverse selection | Fraction of vol attributed to informed flow, not a structural model |
| Single asset | No cross-hedging or multi-asset correlation effects |
| No maker rebates | Exchange fee structure is simplified to a flat percentage |
| Stylised fill model | Exponential decay is a reasonable approximation, not calibrated to data |
| No cancellations | Agent must quote every step; no cancellation or resting orders |

---

## Future Improvements

- [ ] **Prioritised Experience Replay (PER)** — weight surprising transitions more
- [ ] **Multi-asset environment** — correlated assets with cross-hedging
- [ ] **Calibrate to LOB data** — fit fill model to real L2 order book data
- [ ] **PPO/SAC agent** — compare continuous-action policy gradient methods
- [ ] **Markout analysis** — measure adverse selection via post-fill price impact
- [ ] **Hyperparameter optimisation** — Optuna sweep over agent config
- [ ] **Live paper trading** — connect to a broker with simulated capital

---

## How to Talk About This in Interviews

**"What is the core technical contribution?"**
> I built an RL-based quoting agent that jointly learns to balance spread capture and inventory risk. The key insight is that the reward function — which combines mark-to-market PnL with a quadratic inventory penalty — guides the agent to quote symmetrically when flat and skew its quotes when it's accumulated a position, which is exactly what a good human market maker does.

**"Why DQN? Why not PPO or continuous actions?"**
> A discrete action space with a small number of quote tiers (5 levels) is natural for market making because practitioners think in terms of "quote tiers" rather than continuous offsets. DQN with Double + Dueling extensions is well-understood and interpretable. I'd extend to SAC with continuous actions as a next step.

**"What was the hardest part to get right?"**
> The reward shaping. If the inventory penalty is too high, the agent passively quotes extremely wide and fills rarely. Too low, and it accumulates large positions and bleeds on adverse moves. I spent significant time debugging why the agent was converging to degenerate policies and traced it back to inventory penalty scale relative to PnL magnitude.

**"How do you know the results are honest?"**
> I benchmarked against three baselines (random, fixed spread, inventory heuristic). The DQN agent moderately outperforms — it does not massively beat everything, which would be a red flag in a stylized sim. I also capped episode length and max inventory to prevent the agent from exploiting reward-hacking strategies.

---

## Resume Bullet Points

- **Designed and implemented a Deep Q-Network (DQN) market-making agent** in PyTorch trained in a custom stochastic limit order book simulator; agent learned to dynamically adjust quote offsets to balance spread capture and inventory risk, outperforming all rule-based baselines on episode PnL.

- **Engineered a financially coherent simulation environment** including an arithmetic random walk price process, exponential-decay fill probability model, mark-to-market accounting, transaction costs, and a quadratic inventory penalty; validated with 33 pytest unit and smoke tests.

- **Applied Double DQN with Dueling architecture** (decoupled value/advantage streams) with epsilon-greedy exploration, experience replay, target network syncing, and gradient clipping; reproduced results reproducibly using global seeding across the full stack (Python, NumPy, PyTorch).

---

## Project Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | DQN agent, neural networks |
| `numpy` | Price simulation, array math |
| `pandas` | Metrics, logging to CSV |
| `matplotlib` | All plots and figures |
| `pyyaml` | Config file loading |
| `tqdm` | Training progress bars |
| `scipy` | Optional statistical utilities |
| `pytest` | Test suite |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by Viraj Chawda as a quantitative finance portfolio project.*
