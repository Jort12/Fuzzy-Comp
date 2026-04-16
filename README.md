# AI Controller for 2D Asteroid Game (XFC 2026 Entry)

## Overview

This project develops an AI controller for a 2D asteroid game designed for the 2026 XFC (eXplainable Fuzzy Control) competition. The goal is to build a high-performance agent that remains interpretable through fuzzy logic systems.

The system combines:

- **Imitation Learning** (DAgger and ANFIS pretraining)
- **Reinforcement Learning** (PPO fine-tuning / standalone PPO)
- A custom **Takagi–Sugeno–Kang (TSK)** fuzzy inference system
- Hybrid neuro-fuzzy experimentation for explainable control policies

The project explores the trade-off between performance, stability, and explainability in real-time decision-making agents.

---

## Key Features

- AI controller for real-time 2D asteroid gameplay  
- Hybrid learning pipeline:
  - **ANFIS (Adaptive Neuro-Fuzzy Inference System)** for imitation learning pretraining
  - **DAgger (Dataset Aggregation)** for iterative expert policy refinement
  - **PPO (Proximal Policy Optimization)** for final policy optimization
- Custom **TSK fuzzy inference system**
- Experimental actor-critic + fuzzy embedding architectures
- Explainable AI focus for XFC 2026 compliance
- Modular architecture for swapping learning strategies and controllers

---

## System Architecture

### Game Environment Interface

- State extraction from asteroid game simulation
- Action execution (thrust, rotation, fire)
- Reward computation and trajectory logging

---

## Learning Pipeline

The training pipeline follows a staged approach:

### (A) ANFIS Pretraining (Imitation Learning)

- Learns initial control policy from expert demonstrations
- Uses adaptive fuzzy membership tuning
- Produces a smooth, interpretable baseline controller

### (B) DAgger Refinement

- Iteratively collects expert-corrected trajectories
- Reduces distribution shift from imitation learning
- Improves robustness in unseen game states

### (C) PPO Optimization

- Final policy optimization via reinforcement learning
- Either:
  - Fine-tunes pretrained policy, or
  - Runs as a standalone high-performance controller

---

## Fuzzy Logic System (TSK)

The system integrates a Takagi–Sugeno–Kang fuzzy model for interpretability:

- Inputs: game state features (positions, velocities, angles)
- Fuzzy membership functions define state regions
- TSK rules map conditions → continuous control outputs
- Provides interpretable reasoning behind actions

---

## Hybrid Experiments (Actor-Critic + Fuzzy)

We also experimented with:

- Embedding fuzzy logic into policy networks
- Modifying actor-critic architectures with fuzzy feature layers
- Comparing performance vs pure PPO and neuro-fuzzy baselines

---

## Tech Stack

- Python 3.x
- PyTorch
- NumPy
- Custom ANFIS implementation
- Custom TSK fuzzy system
- PPO (custom or library-based)
