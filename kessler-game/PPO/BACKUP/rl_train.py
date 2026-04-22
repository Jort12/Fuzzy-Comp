"""
rl_train.py: finetuning trained neuro-fuzzy policy with PPO.

Usage:
  # Step 1: Train base model from expert data
  python nf_train.py --task maneuver --epochs 200
  python nf_train.py --task combat   --epochs 200

  # Step 2: RL fine-tuning on top of the warm start
  python rl_train.py --episodes 300 --scenario all

  # Step 3: Evaluate
  python rl_train.py --eval --scenario stock
"""

import argparse
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from kesslergame import KesslerGame, GraphicsType

import scenarios as sc
from sugeno_nn import SugenoNet
from rl_policy import StochasticManeuverPolicy,StochasticCombatPolicy,ValueNet,warm_start_maneuver,warm_start_combat
from rl_controller import RLController

# Number of scenarios in the fixed scenario map. Used to size the
# critic's one-hot context input so it can distinguish scenarios.
NUM_SCENARIOS = 8


#  PPO update 
#Generalized Advantage Estimation, computes advantages and returns from rewards and value estimates for a single episode/trajectory. No bootstrapping across episode boundaries.
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        # Terminal step bootstraps with 0 (episode is over, no future reward), so next_val is 0 when t+1 is out of bounds.
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    return advantages, returns



# PPO update from multiple episodes. Computes GAE per-episode to avoid bootstrapping across episode boundaries, then concatenates for minibatch SGD.
def ppo_update_pooled(
    maneuver_policy, combat_policy, value_net, optimizer,
    episode_pool, clip_eps=0.2, entropy_coef=0.01, value_coef=0.25,
    epochs=1, mini_batch_size=512, gamma=0.99, lam=0.95,
    num_scenarios=NUM_SCENARIOS):

    device = next(maneuver_policy.parameters()).device
    # Pool data across episodes, compute advantages and returns per episode to avoid bootstrapping across episode boundaries.
    all_features, all_raw_m, all_fire, all_mine, all_old_logp = [], [], [], [], []
    all_adv, all_ret = [], []
    all_scenario_ctx = []

    for traj in episode_pool:
        if len(traj) == 0:
            continue
        features = torch.cat([t["features"] for t in traj], dim=0).to(device)
        raw_m = torch.cat([t["raw_sample_m"] for t in traj], dim=0).to(device)
        fire_acts = torch.stack([t["fire_action"] for t in traj]).to(device)
        mine_acts = torch.stack([t["mine_action"] for t in traj]).to(device)
        old_logp = torch.stack([t["log_prob"] for t in traj]).to(device).squeeze(-1)
        rewards = [t["reward"] for t in traj]

        #Build scenario one-hot for critic context (policy doesn't see this), must match NUM_SCENARIOS and scenario_to_idx in main()
        sc_id = traj[0].get("scenario_id", 0)#default to 0 if not present, but it should always be there
        sc_onehot = torch.zeros(features.shape[0], num_scenarios, device=device)#shape (T, num_scenarios)
        sc_onehot[:, sc_id] = 1.0#one-hot encoding of the scenario for the critic

        with torch.no_grad():
            values_np = value_net(torch.cat([features, sc_onehot], dim=1)).cpu().numpy()

        advantages, returns = compute_gae(rewards, values_np, gamma, lam)

        all_features.append(features)
        all_raw_m.append(raw_m)
        all_fire.append(fire_acts)
        all_mine.append(mine_acts)
        all_old_logp.append(old_logp)
        all_adv.append(torch.tensor(advantages, dtype=torch.float32, device=device))
        all_ret.append(torch.tensor(returns, dtype=torch.float32, device=device))
        all_scenario_ctx.append(sc_onehot)

    if not all_features:
        return {"total_loss": 0, "policy_loss": 0, "value_loss": 0,
                "entropy": 0, "ratio": 1.0, "skipped": 0}

    features = torch.cat(all_features, dim=0)
    raw_m = torch.cat(all_raw_m, dim=0)
    fire_acts = torch.cat(all_fire, dim=0)
    mine_acts = torch.cat(all_mine, dim=0)
    old_logp = torch.cat(all_old_logp, dim=0)
    adv_t = torch.cat(all_adv, dim=0)
    ret_t = torch.cat(all_ret, dim=0)
    scenario_ctx = torch.cat(all_scenario_ctx, dim=0)

    # Normalize advantages across all pooled data
    if adv_t.std() > 1e-8:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    N = features.shape[0]
    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    entropy_sum = 0.0
    ratio_sum = 0.0
    ratio_count = 0
    skipped_batches = 0

    for _ in range(epochs):
        idxs = torch.randperm(N, device=device)
        for start in range(0, N, mini_batch_size):
            end = min(start + mini_batch_size, N)
            mb = idxs[start:end]

            new_logp_m, entropy_m = maneuver_policy.evaluate_action(features[mb], raw_m[mb])
            new_logp_c, entropy_c = combat_policy.evaluate_action(features[mb], fire_acts[mb], mine_acts[mb])

            new_logp_m = new_logp_m.squeeze(-1)
            new_logp_c = new_logp_c.squeeze(-1)
            entropy_m = entropy_m.squeeze(-1) if entropy_m.dim() > 1 else entropy_m
            entropy_c = entropy_c.squeeze(-1) if entropy_c.dim() > 1 else entropy_c

            new_logp = new_logp_m + new_logp_c
            # Only apply entropy bonus to discrete combat policy.
            # For the continuous maneuver policy, log_std already controls exploration
            entropy = entropy_c

            log_ratio = new_logp - old_logp[mb]
            log_ratio = log_ratio.clamp(-4.0, 4.0)


            ratio = torch.exp(log_ratio)

            # Loose skip guard: block only truly destructive minibatches.
            # Good updates live around 1.8-2.3; this only catches the extremes.
            mean_ratio = float(ratio.mean().item())
            if mean_ratio > 3.0 or mean_ratio < 0.33:
                skipped_batches += 1
                continue

            surr1 = ratio * adv_t[mb]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            v_pred = value_net(torch.cat([features[mb], scenario_ctx[mb]], dim=1))
            value_loss = nn.SmoothL1Loss()(v_pred, ret_t[mb])

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(maneuver_policy.parameters()) +
                list(combat_policy.parameters()) +
                list(value_net.parameters()),
                max_norm=0.5,
            )
            optimizer.step()

            total_loss_sum += float(loss.item())
            policy_loss_sum += float(policy_loss.item())
            value_loss_sum += float(value_loss.item())
            entropy_sum += float(entropy.mean().item())
            ratio_sum += mean_ratio
            ratio_count += 1

        if ratio_count > 0 and abs(ratio_sum / ratio_count - 1.0) > 0.15:
            break

    denom = max(ratio_count, 1)
    return {
        "total_loss": total_loss_sum / denom,
        "policy_loss": policy_loss_sum / denom,
        "value_loss": value_loss_sum / denom,
        "entropy": entropy_sum / denom,
        "ratio": ratio_sum / denom,
        "skipped": skipped_batches,
    }



"""
    Actively shrink exploration after warmup.
    The old code only clamped an upper bound on log_std. If gradients did not
    move log_std downward, std could sit near the same value for hundreds of
    episodes. This helper moves log_std toward a scheduled target each PPO update.
"""
def anneal_log_std_(maneuver_policy, ep, total_episodes, warmup_episodes):

    if ep <= warmup_episodes:
        return

    denom = max(total_episodes - warmup_episodes, 1)
    progress = (ep - warmup_episodes) / denom
    progress = max(0.0, min(1.0, progress))

    target_log_std = -1.10 * (1.0 - progress) + -2.30 * progress

    with torch.no_grad():
        target = torch.full_like(maneuver_policy.log_std, target_log_std)
        maneuver_policy.log_std.lerp_(target, 0.25)
        maneuver_policy.log_std.clamp_(-2.30, -0.90)

#Save in the same format as nf_train.py so nf_infer.py can load it.
def save_bundle(maneuver_policy, combat_policy, mu, sd, feature_cols, num_mfs, path):
    bundle = {"task": "rl", "heads": {}}

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

    # Preserve the trained log_std so eval can reload it
    bundle["log_std"] = maneuver_policy.log_std.detach().cpu().tolist()

    torch.save(bundle, path)
    print(f"Saved maneuver bundle -> {path}")

    # Combat bundle
    combat_path = path.replace("maneuver", "combat")
    combat_bundle = {"task": "rl_combat", "heads": {}}
    for name, net in [("fire", combat_policy.fire_net),
                      ("drop_mine", combat_policy.mine_net)]:
        combat_bundle["heads"][name] = {
            "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
            "feature_cols": feature_cols,
            "mu": mu.tolist() if mu is not None else None,
            "sd": sd.tolist() if sd is not None else None,
            "num_inputs": len(feature_cols),
            "num_mfs": num_mfs,
        }
    torch.save(combat_bundle, combat_path)
    print(f"Saved combat bundle -> {combat_path}")


def save_rl_checkpoint(
    maneuver_policy, combat_policy, value_net, optimizer,
    episode, best_reward, total_episodes, path,
):
    #saves everything needed to resume training, including total_episodes for annealing
    torch.save({
        "maneuver_policy": maneuver_policy.state_dict(),
        "combat_policy": combat_policy.state_dict(),
        "value_net": value_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": episode,
        "best_reward": best_reward,
        "total_episodes": total_episodes,
    }, path)
    print(f"Saved RL checkpoint (ep {episode}) -> {path}")


def load_rl_checkpoint(
    maneuver_policy, combat_policy, value_net, optimizer, path, device,
):
    #restores full training state, returns (episode, best_reward, total_episodes)
    ckpt = torch.load(path, map_location=device)
    maneuver_policy.load_state_dict(ckpt["maneuver_policy"])
    combat_policy.load_state_dict(ckpt["combat_policy"])
    try:
        value_net.load_state_dict(ckpt["value_net"])
        optimizer.load_state_dict(ckpt["optimizer"])
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print("  (value_net/optimizer shape changed — reinitializing critic)")
        else:
            raise
    ep = ckpt.get("episode", 0)
    best = ckpt.get("best_reward", -float("inf"))
    total_ep = ckpt.get("total_episodes", None) # None if old checkpoint
    print(f"Resumed RL checkpoint from ep {ep}, best_reward={best:.2f}")
    return ep, best, total_ep



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--scenario", type=str, default="all")
    p.add_argument("--num_mfs", type=int, default=2,
                   help="Must match the warm-start model")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--init_log_std", type=float, default=-1.0,
                   help="Initial exploration noise (log scale). "
                        "-0.5 ≈ std=0.6, -1.0 ≈ std=0.37, -2.0 ≈ std=0.14")
    p.add_argument("--mini_batch_size", type=int, default=512)
    p.add_argument("--warmup_episodes", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="Resume training from rl_checkpoint.pt")
    args = p.parse_args()



    here = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(here, "models")
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_inputs = 8  # length of FEATURE_COLS

    # Build policies 
    maneuver_policy = StochasticManeuverPolicy(num_inputs, args.num_mfs, init_log_std=args.init_log_std).to(device)
    combat_policy = StochasticCombatPolicy(num_inputs, args.num_mfs).to(device)
    value_net = ValueNet(num_inputs + NUM_SCENARIOS).to(device)

    # Kill dropout in every SugenoNet for RL. Dropout makes forward()
    # stochastic wrt the network old_logp and new_logp diverge even
    # before any gradient step because different masks produce different
    # outputs from the same input, breaking PPO importance sampling.
    # Identity swap is permanent and can't be undone by an accidental .train().
    for net in [maneuver_policy.thrust_net, maneuver_policy.turn_net,
                combat_policy.fire_net, combat_policy.mine_net]:
        net.dropout = nn.Identity()

    #  Warm-start from expert-trained models 
    if args.eval:
        maneuver_path = os.path.join(model_dir, "maneuver_best.pt")
        combat_path   = os.path.join(model_dir, "combat_best.pt")
    else:
        maneuver_path = os.path.join(model_dir, "maneuver.pt")
        combat_path   = os.path.join(model_dir, "combat.pt")

    mu, sd, feature_cols = None, None, None
    if os.path.exists(maneuver_path):
        mu, sd, feature_cols = warm_start_maneuver(maneuver_policy, maneuver_path)
        # Restore trained log_std if available in the bundle
        _bundle = torch.load(maneuver_path, map_location="cpu")
        if "log_std" in _bundle:
            with torch.no_grad():
                maneuver_policy.log_std.copy_(torch.tensor(_bundle["log_std"]))
            print(f"Restored log_std: {maneuver_policy.log_std.data.tolist()}")
        print("Warm started maneuver policy from expert.")
    else:
        print("No maneuver.pt found —> training from scratch")
        
        feature_cols = [
            "dist", "ttc", "heading_err", "approach_speed",
            "ammo", "mines", "threat_density", "threat_angle",
        ]

    if os.path.exists(combat_path):
        mu_c, sd_c = warm_start_combat(combat_policy, combat_path)
        print("Warm started combat policy from expert.")
    else:
        print("No combat.pt found —> training from scratch")

    #Optimizer (all params together) 
    optimizer = optim.Adam(
        list(maneuver_policy.parameters()) +
        list(combat_policy.parameters()) +
        list(value_net.parameters()),
        lr=args.lr,
    )

    # Resume from RL checkpoint if requested
    start_ep = 1
    best_reward = -float("inf")
    total_episodes = args.episodes # save the original training horizon for annealing
    rl_ckpt_path = os.path.join(model_dir, "rl_checkpoint.pt")
    if args.resume and os.path.exists(rl_ckpt_path):
        start_ep, best_reward, saved_total = load_rl_checkpoint(
            maneuver_policy, combat_policy, value_net, optimizer,
            rl_ckpt_path, device,
        )
        start_ep += 1  # resume from the next episode
        if saved_total is not None:
            total_episodes = saved_total # keep the original schedule

    scenario_map = {
        "stock": sc.stock_scenario,
        "donut_ring": sc.donut_ring,
        "vertical_wall_left": sc.vertical_wall_left,
        "spiral_arms": sc.spiral_arms,
        "crossing_lanes": sc.crossing_lanes,
        "asteroid_rain": sc.asteroid_rain,
        "four_corner": sc.four_corner,
        "sniper_practice": sc.sniper_practice, #removed from training pool since it's too different and causes instability, but keep it available for evaluation since it's a fun stress test for the combat policy
    }
    # Fixed index for critic one-hot context. Must match NUM_SCENARIOS.
    scenario_to_idx = {name: i for i, name in enumerate(scenario_map.keys())}
    # Note: if args.scenario is "all", we train on all scenarios but still use the scenario ID as a one-hot context for the critic, so it can learn scenario-specific value estimates. The maneuver and combat policies don't get the scenario ID, so they have to learn a single policy that works across all scenarios. This is a form of multi-task learning that can improve generalization and stability, as the policies learn shared representations that work across different situations, while the critic can still distinguish scenarios to provide accurate value estimates.
    if args.scenario.lower() == "all":
        train_scenario_names = [
            "stock",
            "donut_ring",
            "vertical_wall_left",
            "spiral_arms",
            "crossing_lanes",
            "asteroid_rain",
            "four_corner",
        ]
        eval_scenario_names = list(scenario_map.keys())
    else:
        train_scenario_names = [args.scenario]
        eval_scenario_names = [args.scenario]

    game_settings = {
        "perf_tracker": True,
        "graphics_type": GraphicsType.NoGraphics if not args.eval else GraphicsType.Tkinter,
        "realtime_multiplier":1.0 if args.eval else 0.0, 
        "graphics_obj": None,
        "frequency": 30,
    }
    game = KesslerGame(settings=game_settings)#type:ignore

    #  Training/Eval loop 
    rng = np.random.default_rng()
    episode_pool = [] # list of trajectories (each is a list of dicts)
    pool_steps = 0
    pool_scenarios = set()# track distinct scenarios in current pool
    MIN_POOL_STEPS = 2048  # don't update PPO until this many steps
    MIN_POOL_SCENARIOS = 4 # require scenario diversity before updating
    base_lr = args.lr
    cooldown_until = 0 # episode at which LR cooldown expires
    train_start = time.perf_counter()

    # critic warmup: freeze policy params so only value_net trains, let the critic learn what states are worth before PPO starts pushing the actor around with bad advantage estimates
    policy_frozen = False
    if args.warmup_episodes > 0 and not args.eval:
        for p in maneuver_policy.parameters():
            p.requires_grad = False
        for p in combat_policy.parameters():
            p.requires_grad = False
        policy_frozen = True
        print(f"Policy frozen for first {args.warmup_episodes} episodes (critic warmup)")

    for ep in range(start_ep, args.episodes + 1):
        # unfreeze policy once warmup is done
        if policy_frozen and ep > args.warmup_episodes:
            for p in maneuver_policy.parameters():
                p.requires_grad = True
            for p in combat_policy.parameters():
                p.requires_grad = True
            policy_frozen = False
            print(f"Policy unfrozen at episode {ep} (critic warmup done)")

        if args.eval:
            scenario_name = eval_scenario_names[(ep - 1) % len(eval_scenario_names)]
        else:
            scenario_name = rng.choice(train_scenario_names)
        
        scenario = scenario_map[scenario_name]()

        controller = RLController(
            maneuver_policy, combat_policy,
            mu=mu, sd=sd,
            deterministic=args.eval,  # deterministic during evaluation
        )
        controller.reset()

        t0 = time.perf_counter()
        score, perf = game.run(scenario=scenario, controllers=[controller])
        controller.finalize_episode(score)
        dt = time.perf_counter() - t0

        #episode stats
        traj = controller.trajectory
        ep_reward = sum(t["reward"] for t in traj)
        ep_steps = len(traj)
        hits = sum(t.asteroids_hit for t in score.teams)
        deaths = sum(t.deaths for t in score.teams)

        if args.eval:
            print(f"[EVAL {ep}] {scenario_name}: reward={ep_reward:.1f} "
                  f"hits={hits} deaths={deaths} steps={ep_steps} time={dt:.1f}s")
            continue

        # Add episode to pool (keep separate for correct GAE)
        if ep_steps > 0:
            sc_idx = scenario_to_idx[scenario_name]
            for step in traj:
                step["scenario_id"] = sc_idx
            episode_pool.append(traj)
            pool_steps += ep_steps
            pool_scenarios.add(scenario_name)


        #PPO Update — only when pool has enough steps AND scenario diversity
        stats = None
        if pool_steps >= MIN_POOL_STEPS and len(pool_scenarios) >= MIN_POOL_SCENARIOS:
            stats = ppo_update_pooled(
                maneuver_policy,
                combat_policy,
                value_net,
                optimizer,
                episode_pool,
                clip_eps=args.clip_eps,
                entropy_coef=args.entropy_coef,
                epochs=args.ppo_epochs,
                mini_batch_size=args.mini_batch_size,
                gamma=args.gamma,
            )

            # Actively cool exploration after warmup.
            anneal_log_std_(
                maneuver_policy,
                ep,
                total_episodes,
                args.warmup_episodes,
            )

            episode_pool = []
            pool_steps = 0
            pool_scenarios = set()

            # Restore LR if cooldown expired
            if ep >= cooldown_until and cooldown_until > 0:
                for g in optimizer.param_groups:
                    g["lr"] = base_lr
                cooldown_until = 0

        if stats is None:
            stats = {
                "total_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "ratio": 1.0,
                "skipped": 0,
            }


        #Logging 
        log_std_val = maneuver_policy.log_std.exp().mean().item()
        log_std_raw = maneuver_policy.log_std.detach().cpu().tolist()
        skip_str = f" skip={stats['skipped']}" if stats["skipped"] > 0 else ""
        warmup_str = " [WARMUP]" if policy_frozen else ""
        print(
            f"[{ep:04d}/{args.episodes}] {scenario_name:20s} | " # scenario name padded to 20 chars for alignment
            f"R={ep_reward:7.2f} hits={hits} deaths={deaths} steps={ep_steps:4d} "# episode stats
            f"std={log_std_val:.3f} "# current exploration noise level
            f"loss={stats['total_loss']:.4f} "# PPO loss (for debugging, not a great performance metric on its own)
            f"pi={stats['policy_loss']:.4f} "# policy loss component (how much the actor is updating)
            f"v={stats['value_loss']:.4f} "# value loss component (how much the critic is updating)
            f"H={stats['entropy']:.4f} "# combat policy entropy (exploration bonus from discrete actions)
            f"ratio={stats['ratio']:.3f}{skip_str}{warmup_str} "# PPO ratio (clipped vs unclipped policy update magnitude, should be near 1.0 if updates are stable)
            f"log_std_raw={log_std_raw} "# raw log_std values for debugging
            f"({dt:.1f}s, elapsed={((time.perf_counter()-train_start)/60):.1f}m)"
        )

        #Save checkpoints 
        # Save rolling checkpoint every N episodes
        if ep % args.save_every == 0:
            save_bundle(
                maneuver_policy,
                combat_policy,
                mu,
                sd,
                feature_cols,
                args.num_mfs,
                os.path.join(model_dir, "maneuver_rl.pt"),
            )
            save_rl_checkpoint(
                maneuver_policy, combat_policy, value_net, optimizer,
                ep, best_reward, total_episodes, rl_ckpt_path,
            )

        # Run deterministic eval across ALL scenarios to pick best checkpoint.
        # Single-scenario eval was misleading because a policy good at one
        # scenario but bad at others could look much better or worse than it
        # really is overall.
        if ep % args.save_every == 0:
            total_det_reward = 0.0
            total_det_hits = 0
            total_det_deaths = 0
            eval_details = []

            for eval_name in eval_scenario_names:
                eval_scenario = scenario_map[eval_name]()
                eval_ctrl = RLController(
                    maneuver_policy, combat_policy,
                    mu=mu, sd=sd, deterministic=True,
                )
                eval_ctrl.reset()
                eval_score, _ = game.run(scenario=eval_scenario, controllers=[eval_ctrl])
                eval_ctrl.finalize_episode(eval_score)

                det_reward = sum(t["reward"] for t in eval_ctrl.trajectory)
                det_hits = sum(t.asteroids_hit for t in eval_score.teams)
                det_deaths = sum(t.deaths for t in eval_score.teams)

                total_det_reward += det_reward
                total_det_hits += det_hits
                total_det_deaths += det_deaths
                eval_details.append(f"{eval_name}={det_reward:.0f}")

            avg_det_reward = total_det_reward / len(eval_scenario_names)
            print(
                f" [DET-EVAL] avg={avg_det_reward:.1f} " # average deterministic reward across all eval scenarios, used for best checkpoint selection
                f"total_hits={total_det_hits} total_deaths={total_det_deaths}"
            )
            print(f"    {', '.join(eval_details)}")

            if avg_det_reward > best_reward:
                best_reward = avg_det_reward
                save_bundle(
                    maneuver_policy,
                    combat_policy,
                    mu,
                    sd,
                    feature_cols,
                    args.num_mfs,
                    os.path.join(model_dir, "maneuver_best.pt"),
                )
                print(f" New best (deterministic avg): {best_reward:.2f}")
                # Cooldown: halve LR for 75 ep, hopefully stop the next few updates from destroying this checkpoint before it can be evaluated.
                for g in optimizer.param_groups:
                    g["lr"] = base_lr * 0.5
                cooldown_until = ep + 75
                print(f" LR cooldown: {base_lr:.1e} -> {base_lr*0.5:.1e} until ep {cooldown_until}")

    if not args.eval:
        #Final save
        save_bundle(
            maneuver_policy, combat_policy,
            mu, sd, feature_cols, args.num_mfs,
            os.path.join(model_dir, "maneuver_rl.pt"),
        )
        save_rl_checkpoint(
            maneuver_policy, combat_policy, value_net, optimizer,
            args.episodes, best_reward, total_episodes, rl_ckpt_path,
        )
        print("\nDone. Models saved to models/")
    total_time = time.perf_counter() - train_start
    minutes = total_time / 60
    print(f"\nTotal training time: {minutes:.1f} min ({total_time:.0f}s)")


if __name__ == "__main__":
    main()