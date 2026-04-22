"""
rl_train.py: finetuning trained neuro-fuzzy policy with PPO.

Merged trainer (v2):
  Based on episode-cap variant (separate optimizers, KL early stop,
  per-episode advantage normalization, log_std excluded from optimizer).
  Cooldown mechanism restored from original trainer.
  Anneal schedule fixed: tracks actual PPO update count, not episode number.
  Scenario-diverse pool requirement restored (MIN_POOL_SCENARIOS).
  Actor now receives scenario one-hot context (highest-impact arch change).

Usage:
  # Step 1: Train base model from expert data
  python nf_train.py --task maneuver --epochs 200

  # Step 2: RL fine-tuning on top of the warm start
  python rl_train.py --episodes 300 --scenario all

  # Step 3: Evaluate
  python rl_train.py --eval --scenario stock
  python rl_train.py --eval --graphic --scenario stock --episodes 10
  
  
# Full curriculum training
python -u rl_train.py --episodes 1000 --scenario all --scenario_group foundation --init_bundle models\maneuver_best.pt --csv_log models\foundation_run.csv 2>&1 | Tee-Object -FilePath run_log_foundation.txt
python visualize_training.py --log run_log_foundation.txt --csv models\foundation_run.csv --run_name foundation --bundle models\maneuver_best.pt --compare_run foundation=run_log_foundation.txt

python -u rl_train.py --episodes 1000 --scenario all --scenario_group motion --init_bundle models\maneuver_best.pt --csv_log models\motion_run.csv 2>&1 | Tee-Object -FilePath run_log_motion.txt
python visualize_training.py --log run_log_motion.txt --csv models\motion_run.csv --run_name motion --bundle models\maneuver_best.pt --compare_run foundation=run_log_foundation.txt --compare_run motion=run_log_motion.txt

python -u rl_train.py --episodes 1000 --scenario all --scenario_group pressure --init_bundle models\maneuver_best.pt --csv_log models\pressure_run.csv 2>&1 | Tee-Object -FilePath run_log_pressure.txt
python visualize_training.py --log run_log_pressure.txt --csv models\pressure_run.csv --run_name pressure --bundle models\maneuver_best.pt --compare_run foundation=run_log_foundation.txt --compare_run motion=run_log_motion.txt --compare_run pressure=run_log_pressure.txt

python -u rl_train.py --episodes 1000 --scenario all --scenario_group full --init_bundle models\maneuver_best.pt --csv_log models\full_run.csv 2>&1 | Tee-Object -FilePath run_log_full.txt
python visualize_training.py --log run_log_full.txt --csv models\full_run.csv --run_name full --bundle models\maneuver_best.pt --compare_run foundation=run_log_foundation.txt --compare_run motion=run_log_motion.txt --compare_run pressure=run_log_pressure.txt --compare_run full=run_log_full.txt

"""
import argparse
import csv
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from kesslergame import KesslerGame, GraphicsType

import scenarios as sc
from rl_policy import (
    StochasticManeuverPolicy,
    ValueNet,
    warm_start_maneuver,
)
from rl_controller import RLController

# Keep this fixed so saved scenario IDs always line up with actor/critic inputs.
NUM_SCENARIOS = 8

# Default multi-scenario training set used by --scenario all.
DEFAULT_TRAIN_SCENARIOS = [
    "stock",
    "donut_ring",
    "vertical_wall_left",
    "spiral_arms",
    "crossing_lanes",
    "asteroid_rain",
    "four_corner",
]

# Hand-tuned sampling weights so harder cases show up more often.
DEFAULT_TRAIN_SCENARIO_WEIGHT_MAP = {
    "asteroid_rain": 0.24,
    "vertical_wall_left": 0.18,
    "spiral_arms": 0.15,
    "crossing_lanes": 0.16,
    "stock": 0.10,
    "four_corner": 0.10,
    "donut_ring": 0.07,
}

# Curriculum slices let us train on themed subsets without changing the scenario map.
CURRICULUM_GROUPS = {
    "foundation": [
        "stock",
        "donut_ring",
        "vertical_wall_left",
    ],
    "motion": [
        "asteroid_rain",
        "crossing_lanes",
        "spiral_arms",
    ],
    "pressure": [
        "vertical_wall_left",
        "asteroid_rain",
        "four_corner",
    ],
    "full": list(DEFAULT_TRAIN_SCENARIOS),
}

# Matching default sampling weights for each curriculum slice.
CURRICULUM_GROUP_WEIGHT_MAPS = {
    "foundation": {
        "stock": 0.40,
        "donut_ring": 0.32,
        "vertical_wall_left": 0.28,
    },
    "motion": {
        "asteroid_rain": 0.38,
        "crossing_lanes": 0.34,
        "spiral_arms": 0.28,
    },
    "pressure": {
        "asteroid_rain": 0.38,
        "four_corner": 0.34,
        "vertical_wall_left": 0.28,
    },
    "full": dict(DEFAULT_TRAIN_SCENARIO_WEIGHT_MAP),
}

# Compute PPO advantages and returns from one finished trajectory.
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        # Bootstrap from the next critic value, or 0 at the terminal step.
        next_val = values[t + 1] if t + 1 < T else 0.0
        # Temporal-difference error for this step before the recursive GAE smoothing.
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    return advantages, returns

# Run PPO on a pooled set of finished episodes and return aggregate training stats.
def ppo_update_pooled(
    maneuver_policy, value_net, opt_policy, opt_critic,
    episode_pool, clip_eps=0.2, entropy_coef=0.01, value_coef=1.0,
    epochs=1, mini_batch_size=512, gamma=0.99, lam=0.95,
    max_steps_per_episode=None, num_scenarios=NUM_SCENARIOS):

    device = next(maneuver_policy.parameters()).device
    # Gather each episode separately first, then flatten after GAE is computed.
    all_features, all_raw_m, all_old_logp = [], [], []
    all_adv, all_ret = [], []
    all_scenario_ctx = []
    raw_adv_stds = []

    for traj in episode_pool:
        if len(traj) == 0:
            continue
        # The controller already stored features, sampled actions, and log-probs per step.
        features = torch.cat([t["features"] for t in traj], dim=0).to(device)
        raw_m = torch.cat([t["raw_sample_m"] for t in traj], dim=0).to(device)
        old_logp = torch.stack([t["log_prob"] for t in traj]).to(device).squeeze(-1)
        rewards = [t["reward"] for t in traj]

        sc_id = traj[0].get("scenario_id", 0)
        sc_onehot = torch.zeros(features.shape[0], num_scenarios, device=device)
        sc_onehot[:, sc_id] = 1.0

        with torch.no_grad():
            # The critic also sees scenario context so its baseline matches the policy input.
            values_np = value_net(torch.cat([features, sc_onehot], dim=1)).cpu().numpy()

        advantages, returns = compute_gae(rewards, values_np, gamma, lam)

        adv_ep = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_ep = torch.tensor(returns, dtype=torch.float32, device=device)
        raw_adv_stds.append(float(adv_ep.std()) if adv_ep.numel() > 1 else 0.0)
        # Normalize per episode so one long or extreme rollout does not dominate the pool.
        if adv_ep.numel() > 1 and adv_ep.std() > 1e-8:
            adv_ep = (adv_ep - adv_ep.mean()) / (adv_ep.std() + 1e-8)
        adv_ep = torch.clamp(adv_ep, -5.0, 5.0)

        T = features.shape[0]
        cap = max_steps_per_episode or 0
        if cap > 0 and T > cap:
            # Random cropping keeps update cost bounded for very long episodes.
            start = int(torch.randint(0, T - cap + 1, (1,)).item())
            sl = slice(start, start + cap)
            features  = features[sl]
            raw_m     = raw_m[sl]
            old_logp  = old_logp[sl]
            adv_ep    = adv_ep[sl]
            ret_ep    = ret_ep[sl]
            sc_onehot = sc_onehot[sl]

        all_features.append(features)
        all_raw_m.append(raw_m)
        all_old_logp.append(old_logp)
        all_adv.append(adv_ep)
        all_ret.append(ret_ep)
        all_scenario_ctx.append(sc_onehot)

    if not all_features:
        return {"total_loss": 0, "policy_loss": 0, "value_loss": 0,
                "entropy": 0, "ratio": 1.0, "skipped": 0,
                "n_episodes": 0, "adv_std_spread": 0.0}

    features = torch.cat(all_features, dim=0)
    raw_m = torch.cat(all_raw_m, dim=0)
    old_logp = torch.cat(all_old_logp, dim=0)
    adv_t = torch.cat(all_adv, dim=0)
    ret_t = torch.cat(all_ret, dim=0)
    scenario_ctx = torch.cat(all_scenario_ctx, dim=0)

    N = features.shape[0]
    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    entropy_sum = 0.0
    ratio_sum = 0.0
    ratio_count = 0
    skipped_batches = 0
    target_kl = 0.05

    for _ in range(epochs):
        # Shuffle pooled samples each epoch so minibatches are less correlated.
        idxs = torch.randperm(N, device=device)
        stop_early = False

        for start in range(0, N, mini_batch_size):
            end = min(start + mini_batch_size, N)
            mb = idxs[start:end]

            # Slice one minibatch from the pooled rollout tensors.
            mb_features = features[mb]
            mb_raw_m = raw_m[mb]
            mb_old_logp = old_logp[mb]
            mb_adv = adv_t[mb]
            mb_ret = ret_t[mb]
            mb_sc = scenario_ctx[mb]

            new_logp_m, entropy_m = maneuver_policy.evaluate_action(
                mb_features, mb_raw_m, scenario_onehot=mb_sc)

            new_logp_m = new_logp_m.squeeze(-1)
            entropy_m = entropy_m.squeeze(-1) if entropy_m.dim() > 1 else entropy_m

            new_logp = new_logp_m
            entropy = entropy_m

            # PPO compares the new policy probability to the probability used to sample the action.
            log_ratio = new_logp - mb_old_logp
            log_ratio = log_ratio.clamp(-4.0, 4.0)

            ratio = torch.exp(log_ratio)

            mean_ratio = float(ratio.mean().item())
            approx_kl = float((mb_old_logp - new_logp).mean().abs().item())

            if mean_ratio > 3.0 or mean_ratio < 0.33:
                # Skip obviously unstable batches instead of letting PPO explode.
                skipped_batches += 1
                continue

            # Clipped surrogate objective is the main PPO policy loss.
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Critic learns to predict return targets for the same state/scenario input.
            v_pred = value_net(torch.cat([mb_features, mb_sc], dim=1))
            value_loss = nn.SmoothL1Loss()(v_pred, mb_ret)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            opt_critic.zero_grad(set_to_none=True)
            critic_obj = value_coef * value_loss

            policy_params = list(maneuver_policy.parameters())
            policy_trainable = any(p.requires_grad for p in policy_params)

            if policy_trainable:
                # Policy and critic use separate optimizers so warmup/freezing stays simple.
                opt_policy.zero_grad(set_to_none=True)
                policy_obj = policy_loss - entropy_coef * entropy.mean()
                policy_obj.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(policy_params, max_norm=0.5)
                opt_policy.step()

            # Critic update still runs during warmup, even if the actor is frozen.
            critic_obj.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
            opt_critic.step()

            total_loss_sum += float(loss.item())
            policy_loss_sum += float(policy_loss.item())
            value_loss_sum += float(value_loss.item())
            entropy_sum += float(entropy.mean().item())
            ratio_sum += mean_ratio
            ratio_count += 1

            # Stop early if this PPO pass is already drifting too far from the old policy.
            if approx_kl > target_kl:
                stop_early = True
                break

        if stop_early:
            break

    denom = max(ratio_count, 1)
    return {
        "total_loss": total_loss_sum / denom,
        "policy_loss": policy_loss_sum / denom,
        "value_loss": value_loss_sum / denom,
        "entropy": entropy_sum / denom,
        "ratio": ratio_sum / denom,
        "skipped": skipped_batches,
        "n_episodes": len(raw_adv_stds),
        "adv_std_spread": max(raw_adv_stds) / max(min(raw_adv_stds), 1e-8) if raw_adv_stds else 0.0,
    }



"""
    Actively shrink exploration after each real PPO update.
    Fixed: tracks ppo_update_count instead of episode number, so exploration
    decays evenly regardless of how many episodes it takes to fill the pool.
    Schedule: std ~0.33 (log_std -1.10) at start -> std ~0.10 (log_std -2.30) at end
"""
# Gradually reduce policy exploration noise across PPO updates.
def anneal_log_std_(maneuver_policy, ppo_update_count, expected_updates):

    progress = min(1.0, ppo_update_count / max(expected_updates, 1))

    target_log_std = -1.10 * (1.0 - progress) + -2.30 * progress

    with torch.no_grad():
        target = torch.full_like(maneuver_policy.log_std, target_log_std)
        maneuver_policy.log_std.lerp_(target, 0.25)
        maneuver_policy.log_std.clamp_(-2.30, -0.90)

# Save the compact inference bundle used by eval and deployment scripts.
def save_bundle(maneuver_policy, mu, sd, feature_cols, num_mfs, path):
    bundle = {"task": "rl", "heads": {}}

    # Save both continuous action heads in the same structure used by the NF bundle loader.
    for name, net in [("thrust", maneuver_policy.thrust_net),
                      ("turn_rate", maneuver_policy.turn_net)]:
        bundle["heads"][name] = {
            "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
            "feature_cols": feature_cols,
            "mu": mu.tolist() if mu is not None else None,
            "sd": sd.tolist() if sd is not None else None,
            "num_inputs": len(feature_cols),
            "num_mfs": num_mfs,
        }

    # Persist the learned exploration scale so future runs reload the same behavior.
    bundle["log_std"] = maneuver_policy.log_std.detach().cpu().tolist()

    if maneuver_policy.num_scenarios > 0:
        # Scenario bias parameters are pulled out so older bundle loaders can still find them.
        bundle["scenario_maneuver"] = {
            k: v.cpu() for k, v in maneuver_policy.state_dict().items()
            if "scenario_bias" in k
        }

    torch.save(bundle, path)
    print(f"Saved maneuver bundle -> {path}")



# Save full training state so PPO can resume from the same point later.
def save_rl_checkpoint(
    maneuver_policy, value_net, opt_policy, opt_critic,
    episode, best_reward, total_episodes, ppo_update_count, path,
):
    torch.save({
        "maneuver_policy": maneuver_policy.state_dict(),
        "value_net": value_net.state_dict(),
        "opt_policy": opt_policy.state_dict(),
        "opt_critic": opt_critic.state_dict(),
        "episode": episode,
        "best_reward": best_reward,
        "total_episodes": total_episodes,
        "ppo_update_count": ppo_update_count,
    }, path)
    print(f"Saved RL checkpoint (ep {episode}) -> {path}")


# Restore model weights and optimizer state from an RL checkpoint.
def load_rl_checkpoint(
    maneuver_policy, value_net, opt_policy, opt_critic, path, device,
):
    # strict=False keeps checkpoint loading compatible if new keys were added later.
    ckpt = torch.load(path, map_location=device)
    maneuver_policy.load_state_dict(ckpt["maneuver_policy"], strict=False)
    try:
        value_net.load_state_dict(ckpt["value_net"])
        critic_reloaded = True
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print("  (value_net/optimizer shape changed, reinitializing critic)")
            critic_reloaded = False
        else:
            raise
    if "opt_policy" in ckpt:
        opt_policy.load_state_dict(ckpt["opt_policy"])
        if critic_reloaded:
            opt_critic.load_state_dict(ckpt["opt_critic"])
    elif "optimizer" in ckpt:
        print("  (old single-optimizer checkpoint — skipping optimizer state)")
    ep = ckpt.get("episode", 0)
    best = ckpt.get("best_reward", -float("inf"))
    total_ep = ckpt.get("total_episodes", None)
    ppo_count = ckpt.get("ppo_update_count", 0)
    print(f"Resumed RL checkpoint from ep {ep}, best_reward={best:.2f}, ppo_updates={ppo_count}")
    return ep, best, total_ep, ppo_count


# Create the CSV logger that records one training row per episode.
def create_episode_csv_logger(csv_path: str):
    fieldnames = [
        "episode",
        "total_episodes",
        "scenario",
        "reward",
        "hits",
        "deaths",
        "steps",
        "std",
        "log_std_0",
        "log_std_1",
        "entropy",
        "ratio",
        "total_loss",
        "policy_loss",
        "value_loss",
        "skipped",
        "ppo_update_count",
        "pool_episodes",
        "adv_std_spread",
        "learning_rate",
        "policy_frozen",
        "det_train_avg",
        "det_all_avg",
        "det_total_hits",
        "det_total_deaths",
        "best_reward_so_far",
    ]

    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    handle = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handle.flush()
    return handle, writer


# Parse a CLI weight string into normalized sampling probabilities.
def parse_scenario_weights(weight_spec: str | None, scenario_names):
    if not scenario_names:
        return None

    if weight_spec is None or not weight_spec.strip():
        return None

    weights = {}
    for chunk in weight_spec.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(
                "Scenario weights must look like 'asteroid_rain=0.24,crossing_lanes=0.16'."
            )
        name, raw_value = piece.split("=", 1)
        name = name.strip()
        value = float(raw_value.strip())
        if name not in scenario_names:
            raise ValueError(f"Unknown scenario in weights: {name}")
        if value < 0.0:
            raise ValueError(f"Scenario weight must be non-negative: {name}={value}")
        weights[name] = value

    if not weights:
        return None

    # Order matters here because numpy.choice expects probabilities aligned with scenario_names.
    ordered = np.array([weights.get(name, 0.0) for name in scenario_names], dtype=np.float64)
    total = float(ordered.sum())
    if total <= 0.0:
        raise ValueError("Scenario weights must sum to a positive value.")
    ordered /= total
    return ordered


# Convert a scenario->weight dict into a probability array in scenario order.
def normalize_weight_map(weight_map, scenario_names):
    ordered = np.array([weight_map.get(name, 0.0) for name in scenario_names], dtype=np.float64)
    total = float(ordered.sum())
    if total <= 0.0:
        return None
    ordered /= total
    return ordered


# Run deterministic evaluation over multiple scenarios and collect summary stats.
def run_eval_sweep(game, scenario_names, scenario_map, scenario_to_idx,
                   maneuver_policy, mu, sd, num_scenarios):
    results = []

    for eval_name in scenario_names:
        # Deterministic eval removes action noise so checkpoint comparisons stay stable.
        eval_scenario = scenario_map[eval_name]()
        eval_sc_idx = scenario_to_idx[eval_name]
        eval_ctrl = RLController(
            maneuver_policy,
            mu=mu,
            sd=sd,
            deterministic=True,
            scenario_id=eval_sc_idx,
            num_scenarios=num_scenarios,
        )
        eval_ctrl.reset()
        eval_score, _ = game.run(scenario=eval_scenario, controllers=[eval_ctrl])
        eval_ctrl.finalize_episode(eval_score)

        det_reward = sum(t["reward"] for t in eval_ctrl.trajectory)
        det_hits = sum(t.asteroids_hit for t in eval_score.teams)
        det_deaths = sum(t.deaths for t in eval_score.teams)
        results.append({
            "name": eval_name,
            "reward": det_reward,
            "hits": det_hits,
            "deaths": det_deaths,
        })

    return results



# Parse args, build the trainer, and run training or evaluation.
def main():
    #Absolute behemoth of a cli
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--scenario", type=str, default="all")
    p.add_argument("--num_mfs", type=int, default=2)
    p.add_argument("--init_bundle", type=str, default=None, help="Optional warm-start bundle path. if none, training uses models/maneuver.pt and eval uses models/maneuver_best.pt.")
    p.add_argument("--csv_log", type=str, default=None, help="Optional per ep CSV output path. Defaults to a timestamped file in models/.")
    p.add_argument("--scenario_weights", type=str, default=None,
                   help="Optional weights 'asteroid_rain=0.24,vertical_wall_left=0.18'")
    p.add_argument("--scenario_group", type=str, default=None,
                   help="optional: foundation, motion, pressure, full")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--critic_lr", type=float, default=3e-4, help="Learning rate for value network (higher than policy LR)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--graphics", action="store_true")
    p.add_argument("--init_log_std", type=float, default=-1.0,
                   help="Initial exploration noise (log scale). "
                        "-0.5 ~ std=0.6, -1.0 ~ std=0.37, -2.0 ~ std=0.14")
    p.add_argument("--mini_batch_size", type=int, default=512)
    p.add_argument("--max_steps_per_episode", type=int, default=512)
    p.add_argument("--warmup_episodes", type=int, default=0)
    p.add_argument("--min_pool_steps", type=int, default=2048)
    p.add_argument("--min_pool_scenarios", type=int, default=4)
    p.add_argument("--cooldown_episodes", type=int, default=15)
    p.add_argument("--cooldown_lr_scale", type=float, default=0.5)
    p.add_argument("--early_stop_patience", type=int, default=100, help="Stop after this many eval sweeps without a new best deterministic average. Set to 0 to disable.")
    
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    
    p.add_argument("--resume", action="store_true", help="Resume training from rl_checkpoint.pt")
    args = p.parse_args()



    here = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(here, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Actor gets 8 handcrafted features; critic gets those plus scenario context.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_inputs = 8

    maneuver_policy = StochasticManeuverPolicy(
        num_inputs, args.num_mfs,
        num_scenarios=NUM_SCENARIOS,
        init_log_std=args.init_log_std,
    ).to(device)
    value_net = ValueNet(num_inputs + NUM_SCENARIOS).to(device)
    
    # Disable dropout so PPO compares stable log-probs for the same stored actions.
    for net in [maneuver_policy.thrust_net, maneuver_policy.turn_net]:
        net.dropout = nn.Identity()

    # Choose the warm-start bundle path based on train/eval mode.
    if args.init_bundle is not None:
        maneuver_path = args.init_bundle
        if not os.path.isabs(maneuver_path):
            maneuver_path = os.path.join(here, maneuver_path)
    elif args.eval:
        maneuver_path = os.path.join(model_dir, "maneuver_best.pt")
    else:
        maneuver_path = os.path.join(model_dir, "maneuver.pt")

    mu, sd, feature_cols = None, None, None
    if os.path.exists(maneuver_path): # load the warm start bundle if it exists, otherwise start with untrained policy (but still use the same feature set)
        # Warm start gives PPO a useful prior instead of random maneuvering.
        mu, sd, feature_cols = warm_start_maneuver(maneuver_policy, maneuver_path)
        _bundle = torch.load(maneuver_path, map_location="cpu")
        if "log_std" in _bundle:
            # Restore the saved exploration scale if the bundle already learned one.
            with torch.no_grad():
                maneuver_policy.log_std.copy_(torch.tensor(_bundle["log_std"]))
            print(f"Restored log_std: {maneuver_policy.log_std.data.tolist()}")
        if "scenario_maneuver" in _bundle and maneuver_policy.num_scenarios > 0:
            # Older bundles may not have scenario bias weights, so this stays optional.
            maneuver_policy.load_state_dict(_bundle["scenario_maneuver"], strict=False)
            print("Restored maneuver scenario bias from bundle")
        print("Warm started maneuver policy from expert.")
    else:
        print("No maneuver.pt found -> training from scratch")
        
        feature_cols = [
            "dist", "ttc", "heading_err", "approach_speed",
            "ammo", "mines", "threat_density", "threat_angle",
        ]


    # log_std is controlled by anneal_log_std_(), so it stays out of Adam.
    policy_params = (
        [p for n, p in maneuver_policy.named_parameters() if "log_std" not in n])
    opt_policy = optim.Adam(policy_params, lr=args.lr)
    opt_critic = optim.Adam(value_net.parameters(), lr=args.critic_lr)

    # Resume restores optimizer state and PPO schedule counters from disk.
    start_ep = 1
    best_reward = -float("inf")
    total_episodes = args.episodes
    ppo_update_count = 0
    rl_ckpt_path = os.path.join(model_dir, "rl_checkpoint.pt")
    if args.resume and os.path.exists(rl_ckpt_path):
        start_ep, best_reward, saved_total, ppo_update_count = load_rl_checkpoint(
            maneuver_policy, value_net, opt_policy, opt_critic,
            rl_ckpt_path, device,
        )
        start_ep += 1
        if saved_total is not None:
            total_episodes = saved_total

    # Scenario order must stay fixed because IDs are written into trajectories.
    scenario_map = {
        "stock": sc.stock_scenario,
        "donut_ring": sc.donut_ring,
        "vertical_wall_left": sc.vertical_wall_left,
        "spiral_arms": sc.spiral_arms,
        "crossing_lanes": sc.crossing_lanes,
        "asteroid_rain": sc.asteroid_rain,
        "four_corner": sc.four_corner,
        "sniper_practice": sc.sniper_practice,
    }
    scenario_to_idx = {name: i for i, name in enumerate(scenario_map.keys())}
    num_scenarios = len(scenario_map)
    assert num_scenarios == NUM_SCENARIOS, (
        f"scenario_map has {num_scenarios} entries but NUM_SCENARIOS={NUM_SCENARIOS}"
    )

    selected_group = None
    if args.scenario_group is not None:
        selected_group = args.scenario_group.lower().strip()
        if args.scenario.lower() != "all":
            raise ValueError("--scenario_group can only be used with --scenario all.")
        if selected_group not in CURRICULUM_GROUPS:
            valid_groups = ", ".join(sorted(CURRICULUM_GROUPS.keys()))
            raise ValueError(f"Unknown --scenario_group '{args.scenario_group}'. Valid groups: {valid_groups}")
        train_scenario_names = list(CURRICULUM_GROUPS[selected_group])
        eval_scenario_names = train_scenario_names if args.eval else list(scenario_map.keys())
        print(f"Curriculum group -> {selected_group}: {', '.join(train_scenario_names)}")
    elif args.scenario.lower() == "all":
        train_scenario_names = list(DEFAULT_TRAIN_SCENARIOS)
        eval_scenario_names = list(scenario_map.keys())
    else:
        if args.scenario not in scenario_map:
            valid_names = ", ".join(scenario_map.keys())
            raise ValueError(f"Unknown scenario '{args.scenario}'. Valid scenarios: {valid_names}")
        train_scenario_names = [args.scenario]
        eval_scenario_names = [args.scenario]

    train_scenario_probs = None
    if not args.eval and len(train_scenario_names) > 1:
        weight_spec = args.scenario_weights
        if weight_spec is not None:
            train_scenario_probs = parse_scenario_weights(weight_spec, train_scenario_names)
        else:
            default_weight_map = None
            if selected_group is not None:
                default_weight_map = CURRICULUM_GROUP_WEIGHT_MAPS.get(selected_group)
            elif args.scenario.lower() == "all":
                default_weight_map = DEFAULT_TRAIN_SCENARIO_WEIGHT_MAP
            if default_weight_map is not None:
                train_scenario_probs = normalize_weight_map(default_weight_map, train_scenario_names)
        if train_scenario_probs is not None:
            weights_str = ", ".join(
                f"{name}={prob:.2f}" for name, prob in zip(train_scenario_names, train_scenario_probs)
            )
            print(f"Weighted scenario sampling -> {weights_str}")

    game_settings = {
        "perf_tracker": True,
        "graphics_type": GraphicsType.Tkinter if args.graphics else GraphicsType.NoGraphics,
        "realtime_multiplier": 1.0 if args.graphics else 0.0,
        "graphics_obj": None,
        "frequency": 30,
    }
    game = KesslerGame(settings=game_settings)  # type:ignore

    csv_handle = None
    csv_writer = None
    csv_log_path = None
    if not args.eval:
        # Training writes one CSV row per episode for later plotting and debugging.
        if args.csv_log is not None:
            csv_log_path = args.csv_log
            if not os.path.isabs(csv_log_path):
                csv_log_path = os.path.join(here, csv_log_path)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            csv_log_path = os.path.join(model_dir, f"training_log_{timestamp}.csv")
        csv_handle, csv_writer = create_episode_csv_logger(csv_log_path)
        print(f"CSV logging -> {csv_log_path}")

    rng = np.random.default_rng()
    episode_pool = []
    pool_steps = 0
    pool_scenarios = set()
    base_lr = args.lr
    cooldown_until = 0
    train_start = time.perf_counter()

    # Rough estimate used only for the exploration annealing schedule.
    expected_updates = max(1, (args.episodes - args.warmup_episodes) // 3)
    if not (0.0 < args.cooldown_lr_scale <= 1.0):
        raise ValueError("--cooldown_lr_scale must be in (0, 1].")

    policy_frozen = False
    no_improve_evals = 0
    final_episode = start_ep - 1
    if args.warmup_episodes > 0 and not args.eval:
        # Optional critic warmup lets value targets stabilize before actor updates begin.
        for p in maneuver_policy.parameters():
            p.requires_grad = False

        policy_frozen = True
        print(f"Policy frozen for first {args.warmup_episodes} episodes (critic warmup)")

    if args.init_bundle is not None and not args.eval and not args.resume:
        # Use the warm-start policy itself as the first baseline best score.
        baseline_results = run_eval_sweep(
            game,
            train_scenario_names,
            scenario_map,
            scenario_to_idx,
            maneuver_policy,
            mu,
            sd,
            num_scenarios,
        )
        best_reward = sum(r["reward"] for r in baseline_results) / max(len(baseline_results), 1)
        print(f"Initial train-scenario deterministic avg from init bundle: {best_reward:.2f}")

    for ep in range(start_ep, args.episodes + 1):
        final_episode = ep
        if policy_frozen and ep > args.warmup_episodes:
            # Unfreeze once the requested warmup window is done.
            for p in maneuver_policy.parameters():
                p.requires_grad = True
            policy_frozen = False
            print(f"Policy unfrozen at episode {ep} (critic warmup done)")

        if args.eval:
            scenario_name = eval_scenario_names[(ep - 1) % len(eval_scenario_names)]
        else:
            # Training samples scenarios randomly, optionally using custom weights.
            scenario_name = rng.choice(train_scenario_names, p=train_scenario_probs)
        
        scenario = scenario_map[scenario_name]()
        sc_idx = scenario_to_idx[scenario_name]

        controller = RLController(
            maneuver_policy,
            mu=mu, sd=sd,
            deterministic=args.eval,
            scenario_id=sc_idx,
            num_scenarios=num_scenarios,
        )
        # Reset clears any controller state carried over from a prior episode.
        controller.reset()

        t0 = time.perf_counter()
        score, perf = game.run(scenario=scenario, controllers=[controller])
        controller.finalize_episode(score)
        dt = time.perf_counter() - t0

        # Episode-level metrics come from the stored trajectory plus the final score object.
        traj = controller.trajectory
        ep_reward = sum(t["reward"] for t in traj)
        ep_steps = len(traj)
        hits = sum(t.asteroids_hit for t in score.teams)
        deaths = sum(t.deaths for t in score.teams)

        if args.eval:
            print(f"[EVAL {ep}] {scenario_name}: reward={ep_reward:.1f} "
                  f"hits={hits} deaths={deaths} steps={ep_steps} time={dt:.1f}s")
            continue

        if ep_steps > 0:
            # Every step needs its scenario ID so pooled PPO can rebuild one-hot context.
            for step in traj:
                step["scenario_id"] = sc_idx
            episode_pool.append(traj)
            cap = args.max_steps_per_episode or len(traj)
            # Pool size is measured after per-episode capping so PPO update cost stays predictable.
            pool_steps += min(len(traj), cap)
            pool_scenarios.add(scenario_name)

        stats = None
        if pool_steps >= args.min_pool_steps and len(pool_scenarios) >= args.min_pool_scenarios:
            # Wait for enough data and enough scenario variety before each PPO update.
            
            stats = ppo_update_pooled(
                maneuver_policy,
                
                value_net,
                opt_policy,
                opt_critic,
                episode_pool,
                clip_eps=args.clip_eps,
                entropy_coef=args.entropy_coef,
                value_coef=1.0,
                epochs=args.ppo_epochs,
                mini_batch_size=args.mini_batch_size,
                gamma=args.gamma,
                max_steps_per_episode=args.max_steps_per_episode,
                num_scenarios=num_scenarios,
            )

            ppo_update_count += 1

            if not policy_frozen:
                # Exploration annealing only matters once the policy is actually trainable.
                anneal_log_std_(maneuver_policy, ppo_update_count, expected_updates)

            # Clear pooled rollouts after each PPO update so the next update uses fresh data.
            episode_pool = []
            pool_steps = 0
            pool_scenarios = set()

            if ep >= cooldown_until and cooldown_until > 0:
                # Restore the base LR after the temporary cooldown window ends.
                for g in opt_policy.param_groups:
                    g["lr"] = base_lr
                cooldown_until = 0

        if stats is None:
            # Fill in zero stats for episodes where no PPO update happened yet.
            stats = {
                "total_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "ratio": 1.0,
                "skipped": 0,
                "n_episodes": 0,
                "adv_std_spread": 0.0,
            }

        det_train_avg = None
        det_all_avg = None
        det_total_hits = None
        det_total_deaths = None

        # log_std is stored in log-space, so exp() turns it into an actual std for logging.
        log_std_val = maneuver_policy.log_std.exp().mean().item()
        log_std_raw = maneuver_policy.log_std.detach().cpu().tolist()
        skip_str = f" skip={stats['skipped']}" if stats["skipped"] > 0 else ""
        warmup_str = " [WARMUP]" if policy_frozen else ""
        diag_str = ""
        if stats["n_episodes"] > 0: # only log pool size and advantage spread if PPO update actually ran on some data
            diag_str = (f" pool={stats['n_episodes']}ep"
                        f" adv_spread={stats['adv_std_spread']:.1f}x"
                        f" ppo#{ppo_update_count}")
        
        
        print(# Episode summary
            f"[{ep:04d}/{args.episodes}] {scenario_name:20s} | "
            f"R={ep_reward:7.2f} hits={hits} deaths={deaths} steps={ep_steps:4d} "
            f"std={log_std_val:.3f} "
            f"loss={stats['total_loss']:.4f} "
            f"pi={stats['policy_loss']:.4f} "
            f"v={stats['value_loss']:.4f} "
            f"H={stats['entropy']:.4f} "
            f"ratio={stats['ratio']:.3f}{skip_str}{diag_str}{warmup_str} "
            f"log_std_raw={log_std_raw} "
            f"({dt:.1f}s, elapsed={((time.perf_counter()-train_start)/60):.1f}m)"
        )

        if ep % args.save_every == 0:
            # Rolling saves make long runs safer to interrupt and resume.
            save_bundle(
                maneuver_policy,
                mu,
                sd,
                feature_cols,
                args.num_mfs,
                os.path.join(model_dir, "maneuver_rl.pt"),
            )
            save_rl_checkpoint(
                maneuver_policy, value_net, opt_policy, opt_critic,
                ep, best_reward, total_episodes, ppo_update_count, rl_ckpt_path,
            )

        if ep % args.save_every == 0:
            # Deterministic eval decides whether this checkpoint becomes the new best model.
            eval_results = run_eval_sweep(
                game,
                eval_scenario_names,
                scenario_map,
                scenario_to_idx,
                maneuver_policy,
                mu,
                sd,
                num_scenarios,
            )
            train_eval_results = [r for r in eval_results if r["name"] in train_scenario_names]
            extra_eval_results = [
                r for r in eval_results
                if r["name"] not in train_scenario_names and r["name"] != "sniper_practice"
            ]
            sniper_result = next((r for r in eval_results if r["name"] == "sniper_practice"), None)

            total_det_reward = sum(r["reward"] for r in eval_results)
            total_det_hits = sum(r["hits"] for r in eval_results)
            total_det_deaths = sum(r["deaths"] for r in eval_results)
            avg_det_reward = total_det_reward / len(eval_results)
            avg_train_reward = (
                sum(r["reward"] for r in train_eval_results) / len(train_eval_results)
                if train_eval_results else 0.0
            )
            det_train_avg = avg_train_reward
            det_all_avg = avg_det_reward
            det_total_hits = total_det_hits
            det_total_deaths = total_det_deaths

            print(
                f" [DET-EVAL] train_avg={avg_train_reward:.1f} all_avg={avg_det_reward:.1f} "
                f"total_hits={total_det_hits} total_deaths={total_det_deaths}"
            )
            print("    " + ", ".join(
                f"{r['name']}={r['reward']:.0f}(h{r['hits']}/d{r['deaths']})"
                for r in train_eval_results
            ))
            if extra_eval_results:
                print("    extra: " + ", ".join(
                    f"{r['name']}={r['reward']:.0f}(h{r['hits']}/d{r['deaths']})"
                    for r in extra_eval_results
                ))
            if sniper_result is not None:
                print(
                    f"    combat-check: sniper_practice={sniper_result['reward']:.0f} "
                    f"(h{sniper_result['hits']}/d{sniper_result['deaths']})"
                )

            improved = avg_train_reward > (best_reward + args.early_stop_min_delta)
            if improved:
                best_reward = avg_train_reward
                no_improve_evals = 0
                save_bundle(
                    maneuver_policy,
                    mu,
                    sd,
                    feature_cols,
                    args.num_mfs,
                    os.path.join(model_dir, "maneuver_best.pt"),
                )
                print(f" New best (train-scenario deterministic avg): {best_reward:.2f}")
                # Temporary LR cooldown reduces the chance of immediate regression.
                for g in opt_policy.param_groups:
                    g["lr"] = base_lr * args.cooldown_lr_scale
                cooldown_until = ep + args.cooldown_episodes
                print(
                    f" LR cooldown: {base_lr:.1e} -> {base_lr*args.cooldown_lr_scale:.1e} "
                    f"until ep {cooldown_until}"
                )
            else:
                no_improve_evals += 1
                if args.early_stop_patience > 0:
                    print(f" No new best for {no_improve_evals}/{args.early_stop_patience} eval sweeps")
                    if no_improve_evals >= args.early_stop_patience:
                        print(" Early stopping on deterministic eval plateau")
                        break

        if csv_writer is not None:
            # Write one CSV row per episode, even when PPO did not update this round.
            current_lr = opt_policy.param_groups[0]["lr"] if opt_policy.param_groups else args.lr
            csv_writer.writerow({
                "episode": ep,
                "total_episodes": args.episodes,
                "scenario": scenario_name,
                "reward": f"{ep_reward:.6f}",
                "hits": hits,
                "deaths": deaths,
                "steps": ep_steps,
                "std": f"{log_std_val:.6f}",
                "log_std_0": f"{log_std_raw[0]:.6f}" if len(log_std_raw) > 0 else "",
                "log_std_1": f"{log_std_raw[1]:.6f}" if len(log_std_raw) > 1 else "",
                "entropy": f"{stats['entropy']:.6f}",
                "ratio": f"{stats['ratio']:.6f}",
                "total_loss": f"{stats['total_loss']:.6f}",
                "policy_loss": f"{stats['policy_loss']:.6f}",
                "value_loss": f"{stats['value_loss']:.6f}",
                "skipped": stats["skipped"],
                "ppo_update_count": ppo_update_count,
                "pool_episodes": stats["n_episodes"],
                "adv_std_spread": f"{stats['adv_std_spread']:.6f}",
                "learning_rate": f"{current_lr:.10f}",
                "policy_frozen": int(policy_frozen),
                "det_train_avg": f"{det_train_avg:.6f}" if det_train_avg is not None else "",
                "det_all_avg": f"{det_all_avg:.6f}" if det_all_avg is not None else "",
                "det_total_hits": det_total_hits if det_total_hits is not None else "",
                "det_total_deaths": det_total_deaths if det_total_deaths is not None else "",
                "best_reward_so_far": f"{best_reward:.6f}",
            })
            csv_handle.flush()

    if not args.eval:
        # Final save preserves the latest state even if training stopped early.
        save_bundle(
            maneuver_policy,
            mu, sd, feature_cols, args.num_mfs,
            os.path.join(model_dir, "maneuver_rl.pt"),
        )
        save_rl_checkpoint(
            maneuver_policy, value_net, opt_policy, opt_critic,
            final_episode, best_reward, total_episodes, ppo_update_count, rl_ckpt_path,
        )
        print("\nDone. Models saved to models/")
        if csv_log_path is not None:
            print(f"CSV log saved to {csv_log_path}")

    if csv_handle is not None:
        # Close the CSV cleanly so the last buffered row is not lost.
        csv_handle.close()

    total_time = time.perf_counter() - train_start
    minutes = total_time / 60
    print(f"\nTotal training time: {minutes:.1f} min ({total_time:.0f}s)")


if __name__ == "__main__":
    main()






# ------------------------------------------------------------
# INTRO: BASIC TARGETING
# Teaches the AI how to aim, shoot, and move in simple situations
# No pressure, very readable patterns
# Delete the first 2 or 3 if you want
# ------------------------------------------------------------
# INTRO_TARGETING_GROUP = 
# [
#     sc.single_target_practice(),
#     sc.dual_static_targets(),
#     sc.donut_ring(),
#     sc.slow_crossing_paths(),
#     sc.lane_switcher(),
#     sc.stock_scenario(),
# ]


# ------------------------------------------------------------
# LINEAR MOVEMENT: EASY DODGING
# Teaches the AI how to dodge simple straight moving threats
# Everything moves in clear directions
# ------------------------------------------------------------
# LINEAR_FLOW_GROUP = 
# [
#     sc.staggered_fall(),
#     sc.vertical_wall_left(),
#     sc.asteroid_rain(),
#     sc.horizontal_gate_runner(),
# ]


# ------------------------------------------------------------
# RING: CENTER PRESSURE
# Teaches the AI how to deal with pressure around itself
# Focus on staying alive when surrounded or collapsing inward
# ------------------------------------------------------------
# RING_AND_COLLAPSE_GROUP = 
# [
#     sc.donut_ring_closing(),
#     sc.inner_outer_rings(),
# ]


# ------------------------------------------------------------
# LANES: GRID TRAFFIC
# Teaches the AI how to read patterns and move through traffic
# Multiple directions at once
# ------------------------------------------------------------
# LANE_AND_GRID_GROUP = 
# [
#     sc.crossing_lanes(),
#     sc.phase_shift_grid(),
#     sc.diagonal_grid_fast(),
# ]


# ------------------------------------------------------------
# MAZE: PATH FINDING
# Teaches the AI how to find safe paths and navigate tight spaces
# Focus on positioning instead of just reacting
# ------------------------------------------------------------
# MAZE_AND_PATHING_GROUP = 
# [
#     sc.moving_maze_right(),
#     sc.s_curve_chokepoint(),
# ]


# ------------------------------------------------------------
# CURVED: ROTATING MOTION
# Teaches the AI to predict movement and not just in straight lines
# Important for aiming and avoiding harder patterns
# ------------------------------------------------------------
# CURVE_AND_ORBIT_GROUP = 
# [
#     sc.spiral_arms(),
#     sc.double_orbit_with_darts(),
#     sc.rotating_cross(),
# ]


# ------------------------------------------------------------
# EDGE: WRAP PRESSURE
# Teaches the AI to watch edges and react to side attacks
# Danger comes from off screen or both sides
# ------------------------------------------------------------
# EDGE_AND_WRAP_PRESSURE_GROUP = 
# [
#     sc.wrap_wall_light(),
#     sc.wrap_pincer(),
# ]


# ------------------------------------------------------------
# WAVE: SURROUND ATTACKS
# Teaches the AI to survive multiple attacks at once
# Pressure comes from different angles repeatedly
# ------------------------------------------------------------
# WAVE_AND_SURROUND_GROUP = 
# [
#     sc.corner_wave_pairs(),
#     sc.corner_shockwaves(),
# ]


# ------------------------------------------------------------
# PRIORITY / TARGET CHOICE
# Teaches the AI what to focus on first, big or small threats
# Important for decision making
# ------------------------------------------------------------
# MIXED_PRIORITY_GROUP = 
# [
#     sc.giants_with_kamikaze(),
# ]


# ------------------------------------------------------------
# COMPRESSION: HARD SURVIVAL
# Teaches the AI to survive when space gets very tight High Pressure
# ------------------------------------------------------------
# SPACE_COMPRESSION_GROUP = 
# [
#     sc.pinch_chamber(),
# ]
