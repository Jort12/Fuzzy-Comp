# Resume Update

## Project Title
Hybrid Neuro-Fuzzy and Reinforcement Learning Controller for Kessler Game

## One-Line Summary
Built a Python/PyTorch autonomous game agent that combined fuzzy logic, neuro-fuzzy imitation learning, and PPO fine-tuning to navigate and fight across diverse asteroid-combat scenarios.

## Resume-Ready Bullets
- Built an autonomous spacecraft controller in Python and PyTorch that combined a hand-engineered fuzzy expert, neuro-fuzzy imitation learning, and PPO fine-tuning for real-time decision-making in the Kessler Game environment.
- Collected and trained on 456,180 expert state-action samples for both maneuvering and combat, using 8 normalized threat and ship-state features to learn reusable Sugeno neuro-fuzzy policies for inference and warm starts.
- Designed a scenario-conditioned PPO training pipeline with generalized advantage estimation, KL-based early stopping, critic warmup, checkpoint/resume support, exploration annealing, and pooled updates across 8 custom combat scenarios.
- Improved training stability with reward shaping, target-locking for consistent threat tracking, and deterministic multi-scenario evaluation; logged a best experimental score of 295.5 average reward with 356 asteroid hits and 29 deaths across 8 evaluation scenarios.

## Shorter Version
- Built a hybrid fuzzy and reinforcement learning controller in Python/PyTorch for the Kessler Game asteroid-combat simulator.
- Trained neuro-fuzzy policies on 456K+ expert examples and fine-tuned them with PPO across 8 custom scenarios using scenario-aware policy conditioning and deterministic evaluation.

## Skills and Keywords
Python, PyTorch, Reinforcement Learning, PPO, Neuro-Fuzzy Systems, Imitation Learning, Simulation, Reward Shaping, Policy Optimization, Model Evaluation

## Notes
- The 456,180-sample counts come from both `data/maneuver.csv` and `data/combat.csv`.
- The best logged 295.5 average reward metric comes from the experimental shared-actor PPO run in `run_log.txt`.
