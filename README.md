# AI Controller for Kessler (XFC 2026 Entry)

## Overview

This project develops AI controllers for [Kessler](https://github.com/ThalesGroup/kessler-game), a 2D asteroid simulation, as an entry for the 2026 XFC (eXplainable Fuzzy Competition). The goal is a high-performance agent that remains interpretable through fuzzy logic.

The staged pipeline is:

1. **Imitation learning** — train a Sugeno neuro-fuzzy network on expert demonstrations
2. **DAgger refinement** — iteratively correct the student on states it actually visits
3. **PPO fine-tuning** — wrap the Sugeno net as a stochastic policy and optimize with reinforcement learning against curriculum scenarios

A separate fuzzy actor-critic track and several hand-authored fuzzy baselines are also included for comparison.

---

## Repository Layout

```
kessler-game/
├── src/kesslergame/            # Upstream Kessler simulation engine (Thales Group)
├── kessler_graphics/           # Optional UE5 graphics project
├── docs/                       # Kessler API & settings documentation
├── examples/                   # Hand-authored fuzzy controllers + scenarios
│   ├── ActorCriticFuzz/        # Fuzzy TSK actor-critic (TD-based online updates)
│   ├── hybrid_fuzzy.py         # Hybrid offensive/defensive fuzzy controller
│   ├── defensive_fuzzy.py      # Survival-oriented fuzzy controller
│   ├── fuzzy_aggressive_controller.py
│   ├── scenarios.py
│   ├── scenario_gauntlet.py
│   └── human_xbox_controller.py
├── neural_fuzzy/               # Stage 1–2: imitation learning + DAgger
│   ├── sugeno_nn.py            # Gaussian MFs + Rule layer + SugenoNet (PyTorch)
│   ├── nf_train.py             # Train Sugeno net on maneuver/combat data
│   ├── nf_controller.py        # Kessler controller wrapping the trained net
│   ├── dagger_collect.py       # Mixed expert/student rollouts (β schedule)
│   ├── dagger_train.py         # Full DAgger loop (collect → merge → retrain)
│   └── merge_datasets.py
└── PPO/                        # Stage 3: PPO fine-tuning of the fuzzy policy
    ├── rl_policy.py            # StochasticManeuverPolicy, StochasticCombatPolicy, ValueNet
    ├── rl_controller.py        # Rollout controller; builds features, samples actions, logs trajectories
    ├── rl_train.py             # PPO trainer (GAE, KL early stop, curriculum groups, cooldown)
    ├── rl_policy_exp.py        # Experimental shared-trunk MLP policy (no fuzzy warm start)
    ├── rl_train_exp.py         # Trainer for the experimental variant
    ├── visualize_training.py   # Parse run logs + CSVs into training plots
    └── models/                 # Checkpoints, warm-start bundles, per-run CSVs
```

---

## Approaches

### 1. Neuro-Fuzzy Sugeno Controller (`neural_fuzzy/`)

A Takagi–Sugeno network implemented in PyTorch (`sugeno_nn.py`):

- **Gaussian membership functions** with learnable centers and log-sigmas
- **Rule layer** enumerating all MF combinations across input features
- Separate heads per output: `thrust` + `turn_rate` for maneuver, `fire` + `drop_mine` for combat

Input features (shared across both tasks):

`dist`, `ttc` (time-to-collision), `heading_err`, `approach_speed`, `ammo`, `mines`, `threat_density`, `threat_angle`

Training (`nf_train.py`) uses MSE for the maneuver head and BCE-with-logits for the combat head, with early stopping on a validation split. Trained weights are saved as a bundle alongside feature normalization stats.

### 2. DAgger Refinement

`dagger_train.py` runs the full Dataset Aggregation loop:

1. Roll out episodes with a mixture of the expert and the current student, interpolated by a linearly decaying `β` (1.0 → 0.0)
2. Log `(state, expert_action)` pairs to CSV
3. Merge base demonstrations with newly collected DAgger data
4. Retrain the Sugeno net on the aggregated dataset
5. Repeat

The oracle used for corrections is the hand-authored `hybrid_fuzzy.py` controller.

### 3. PPO Fine-Tuning (`PPO/`)

The core RL stage. The pre-trained Sugeno net is wrapped as a stochastic policy and fine-tuned with PPO.

**Policy architecture** (`rl_policy.py`):

- `StochasticManeuverPolicy` — two SugenoNets (thrust, turn) produce Gaussian means with a learnable `log_std`; actions are tanh-squashed into [-1, 1] with the corresponding log-prob correction for PPO
- `StochasticCombatPolicy` — two SugenoNets produce Bernoulli logits for `fire` and `drop_mine` (clamped to ±4 to bound the log-ratio)
- `ValueNet` — plain MLP critic with LayerNorm + Tanh
- **Scenario conditioning** — optional one-hot scenario ID feeds a small linear layer that adds a learned bias to each head's output, so the policy can specialize per scenario while sharing a trunk
- `warm_start_maneuver` loads the `nf_train.py` bundle into the thrust/turn SugenoNets and returns the saved normalization stats

**Trainer** (`rl_train.py`):

- Generalized Advantage Estimation (λ=0.95) with per-episode advantage normalization
- Clipped PPO objective with separate optimizers for actor and critic
- **KL early stop** per update, configurable `ppo_epochs`
- **Cooldown** — after large KL spikes, temporarily scale down the LR
- `log_std` excluded from the optimizer (annealed on a schedule)
- Minimum pool size and **minimum scenario diversity** before an update
- Scenario sampling weights bias harder cases (`asteroid_rain`, `vertical_wall_left`) upward

**Curriculum groups** (`CURRICULUM_GROUPS`) for staged training via `--scenario_group`:

| Group | Scenarios |
|---|---|
| `foundation` | stock, donut_ring, vertical_wall_left |
| `motion` | asteroid_rain, crossing_lanes, spiral_arms |
| `pressure` | vertical_wall_left, asteroid_rain, four_corner |
| `full` | all seven training scenarios |

**Experimental variant** (`rl_policy_exp.py`, `rl_train_exp.py`): a shared-trunk MLP actor with four heads (thrust, turn, fire, mine) trained from scratch — no fuzzy warm start. Used as a baseline to measure what the fuzzy prior buys us.

### 4. Fuzzy Actor-Critic (`examples/ActorCriticFuzz/`)

An earlier experiment where both the policy and value function are Sugeno fuzzy systems built from JSON rule bases (`actor_rules.json`, `critic_rules.json`). The critic is updated online via TD error, with per-rule contributions weighted by firing strength. Kept for comparison; not the primary competition entry.

### 5. Hand-Authored Fuzzy Baselines (`examples/`)

- `hybrid_fuzzy.py` — threat prioritization, intercept-point targeting, rear-clearance checks
- `defensive_fuzzy.py` — survival-first evasive behavior
- `fuzzy_aggressive_controller.py` — aggressive target engagement

These serve both as competition-candidate controllers and as expert policies for imitation learning.

---

## Installation

The base Kessler simulator is a packaged Python module. From the repo root:

```bash
pip install -e .\kessler-game
```

For the neuro-fuzzy and PPO pipelines, additionally install:

```bash
pip install torch pandas numpy matplotlib
```

If you plan to use the Xbox controller for human demonstrations, also install your platform's gamepad library (e.g. `inputs` or `pygame`).

---

## Usage

### Run a scenario with a controller

```bash
cd kessler-game/examples
python scenario_test.py          # single scenario
python scenario_gauntlet.py      # sequence of scenarios
```

Edit the scenario / controller selection at the top of those files to swap between `hybrid_fuzzy`, `defensive_fuzzy`, the actor-critic, or a trained neuro-fuzzy controller.

### Stage 1 — Train the neuro-fuzzy controller

```bash
cd neural_fuzzy

# Maneuver head (thrust + turn_rate)
python nf_train.py --task maneuver --epochs 200 --num_mfs 3

# Combat head (fire + drop_mine)
python nf_train.py --task combat --epochs 200 --num_mfs 3
```

Models are saved under `neural_fuzzy/models/`.

### Stage 2 — DAgger refinement

```bash
cd neural_fuzzy
python dagger_train.py --iters 5 --episodes_per_iter 10 --scenario all
```

Collects rollouts across all training scenarios, aggregates datasets, and retrains both heads every iteration.

### Stage 3 — PPO fine-tuning

```bash
cd PPO

# Fine-tune the warm-started maneuver policy on all scenarios
python rl_train.py --episodes 300 --scenario all

# Curriculum training example (Windows PowerShell syntax)
python -u rl_train.py --episodes 1000 --scenario all \
  --scenario_group foundation \
  --init_bundle models/maneuver_best.pt \
  --csv_log models/foundation_run.csv 2>&1 | tee run_log_foundation.txt

# Evaluate deterministically, optionally with graphics
python rl_train.py --eval --scenario stock
python rl_train.py --eval --graphics --scenario stock --episodes 10
```

Common flags: `--lr`, `--critic_lr`, `--clip_eps`, `--entropy_coef`, `--ppo_epochs`, `--mini_batch_size`, `--max_steps_per_episode`, `--min_pool_steps`, `--min_pool_scenarios`, `--cooldown_episodes`, `--cooldown_lr_scale`, `--early_stop_patience`, `--resume`.

### Visualize a training run

```bash
cd PPO
python visualize_training.py \
  --log run_log_foundation.txt \
  --csv models/foundation_run.csv \
  --run_name foundation \
  --bundle models/maneuver_best.pt \
  --compare_run foundation=run_log_foundation.txt
```

Plots are written under `PPO/artifacts/plots/`.

---

## Tech Stack

- **Python 3.10+**
- **PyTorch** — neuro-fuzzy training, PPO policies & critic
- **NumPy / pandas** — data pipeline and logging
- **matplotlib** — training visualizations
- **Kessler** (Thales Group) — simulation environment
- Custom Sugeno/TSK fuzzy inference implementations

---

## Acknowledgments

Built on top of [Kessler](https://github.com/ThalesGroup/kessler-game) by Thales Group, used as the simulation environment for the [Explainable Fuzzy Competition](https://xfuzzycomp.github.io/XFC/).