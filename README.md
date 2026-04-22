# AI Controller for Kessler (XFC 2026 Entry)

## Overview

This project develops AI controllers for [Kessler](https://github.com/ThalesGroup/kessler-game), a 2D asteroid simulation, as an entry for the 2026 XFC (eXplainable Fuzzy Competition). The goal is a high-performance agent that remains interpretable through fuzzy logic.

The project explores several controller designs along the performance vs. explainability trade-off:

- **Neuro-fuzzy (Sugeno) controllers** trained by imitation learning with DAgger refinement
- **Fuzzy actor-critic** with TD-based online updates over TSK rules
- **Hand-authored hybrid fuzzy controllers** (aggressive, defensive, hybrid) as strong baselines

---

## Repository Layout

```
kessler-game/
├── src/kesslergame/            # Upstream Kessler simulation engine (Thales Group)
├── kessler_graphics/           # Optional UE5 graphics project
├── docs/                       # Kessler API & settings documentation
├── examples/                   # Hand-authored fuzzy controllers + scenarios
│   ├── ActorCriticFuzz/        # Fuzzy TSK actor-critic implementation
│   ├── hybrid_fuzzy.py         # Hybrid offensive/defensive fuzzy controller
│   ├── defensive_fuzzy.py      # Survival-oriented fuzzy controller
│   ├── fuzzy_aggressive_controller.py
│   ├── scenarios.py            # Custom scenario definitions
│   ├── scenario_gauntlet.py    # Run a sequence of scenarios back-to-back
│   └── human_xbox_controller.py
└── neural_fuzzy/               # Neuro-fuzzy imitation-learning pipeline
    ├── sugeno_nn.py            # Gaussian MFs + Rule layer + SugenoNet (PyTorch)
    ├── nf_train.py             # Train Sugeno net on maneuver/combat data
    ├── nf_infer.py             # Load bundle and run inference
    ├── nf_controller.py        # Kessler controller wrapping the trained net
    ├── dagger_collect.py       # Mixed expert/student rollouts (beta schedule)
    ├── dagger_train.py         # Full DAgger loop (collect → merge → retrain)
    ├── dagger_controller.py    # Controller used during DAgger rollouts
    ├── hybrid_fuzzy.py         # Expert controller used as DAgger oracle
    ├── merge_datasets.py       # Aggregate base + DAgger CSVs
    └── scenarios.py            # Training/eval scenarios
```

---

## Approaches

### 1. Neuro-Fuzzy Sugeno Controller (`neural_fuzzy/`)

A Takagi–Sugeno network implemented in PyTorch (`sugeno_nn.py`):

- **Gaussian membership functions** with learnable centers and log-sigmas
- **Rule layer** enumerating all MF combinations across input features
- Separate heads trained per output (`thrust` + `turn_rate` for maneuver; `fire` + `drop_mine` for combat)

Input features used by both tasks:

`dist`, `ttc` (time-to-collision), `heading_err`, `approach_speed`, `ammo`, `mines`, `threat_density`, `threat_angle`

Training (`nf_train.py`) uses MSE for the maneuver head and BCE-with-logits for the combat head, with early stopping on a validation split. Trained weights are saved as a bundle alongside feature normalization stats.

### 2. DAgger Refinement

`dagger_train.py` runs the full Dataset Aggregation loop:

1. Roll out episodes with a mixture of the expert and the current student, interpolated by a linearly decaying `beta` (1.0 → 0.0)
2. Log `(state, expert_action)` pairs to CSV
3. Merge base demonstrations with newly collected DAgger data
4. Retrain the Sugeno net on the aggregated dataset
5. Repeat

This reduces distribution shift from pure behavior cloning and improves robustness on states the student actually visits. The oracle used for corrections is the hand-authored `hybrid_fuzzy.py` controller.

### 3. Fuzzy Actor-Critic (`examples/ActorCriticFuzz/`)

An experimental actor-critic where both the policy and value function are Sugeno fuzzy systems built from JSON rule bases (`actor_rules.json`, `critic_rules.json`). The critic is updated online via TD error, with per-rule contributions weighted by firing strength.

### 4. Hand-Authored Fuzzy Baselines (`examples/`)

- `hybrid_fuzzy.py` — combines threat prioritization, intercept-point targeting, and rear-clearance checks
- `defensive_fuzzy.py` — survival-first evasive behavior
- `fuzzy_aggressive_controller.py` — aggressive target engagement

These serve both as competition-candidate controllers and as expert policies for imitation learning.

---

## Installation

The base Kessler simulator is a packaged Python module. From the repo root:

```bash
pip install -e .
```

For the neuro-fuzzy pipeline, additionally install:

```bash
pip install torch pandas numpy
```

If you plan to use the Xbox controller for human demonstrations, also install your platform's gamepad library (e.g. `inputs` or `pygame`).

---

## Usage

### Run a scenario with a controller

```bash
cd examples
python scenario_test.py          # single scenario
python scenario_gauntlet.py      # sequence of scenarios
```

Edit the scenario/controller selection at the top of those files to swap between `hybrid_fuzzy`, `defensive_fuzzy`, the actor-critic, or the neuro-fuzzy controller.

### Train the neuro-fuzzy controller

```bash
cd neural_fuzzy

# Train maneuver head (thrust + turn_rate)
python nf_train.py --task maneuver --epochs 200 --num_mfs 3

# Train combat head (fire + drop_mine)
python nf_train.py --task combat --epochs 200 --num_mfs 3
```

Models are saved under `neural_fuzzy/models/`.

### Run the full DAgger loop

```bash
cd neural_fuzzy
python dagger_train.py --iters 5 --episodes_per_iter 10 --scenario all
```

This will collect rollouts across all training scenarios, aggregate datasets, and retrain both heads every iteration.

### Evaluate

```bash
cd neural_fuzzy
python eval_controller.py
```

---

## Tech Stack

- **Python 3.10+**
- **PyTorch** — neuro-fuzzy model training
- **NumPy / pandas** — data pipeline
- **Kessler** (Thales Group) — simulation environment
- Custom Sugeno/TSK fuzzy inference implementations

---

## Acknowledgments

Built on top of [Kessler](https://github.com/ThalesGroup/kessler-game) by Thales Group, used as the simulation environment for the [Explainable Fuzzy Competition](https://xfuzzycomp.github.io/XFC/).