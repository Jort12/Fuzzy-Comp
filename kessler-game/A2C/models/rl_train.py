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
from rl_policy import (
    StochasticManeuverPolicy,
    StochasticCombatPolicy,
    ValueNet,
    warm_start_maneuver,
    warm_start_combat,
)
from rl_controller import RLController


#  PPO update 

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    #Generalized Advantage Estimation.
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        # Terminal step bootstraps with 0 (episode is over, no future reward)
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    return advantages, returns


def ppo_update(
    maneuver_policy,
    combat_policy,
    value_net,
    optimizer,
    trajectory,
    clip_eps=0.2,
    entropy_coef=0.03,
    value_coef=0.25,
    epochs=2,
    mini_batch_size=32,
    gamma=0.99,
    lam=0.95,
):
    device = next(maneuver_policy.parameters()).device

    features = torch.cat([t["features"] for t in trajectory], dim=0).to(device)
    raw_m = torch.cat([t["raw_sample_m"] for t in trajectory], dim=0).to(device)
    fire_acts = torch.stack([t["fire_action"] for t in trajectory]).to(device)
    mine_acts = torch.stack([t["mine_action"] for t in trajectory]).to(device)
    old_logp = torch.stack([t["log_prob"] for t in trajectory]).to(device).squeeze(-1)
    rewards = [t["reward"] for t in trajectory]

    with torch.no_grad():
        old_values = value_net(features)
        values_np = old_values.detach().cpu().numpy()

    advantages, returns = compute_gae(rewards, values_np, gamma, lam)

    adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)

    # Normalize only advantages
    if adv_t.std() > 1e-8:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    N = len(trajectory)

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

            mb_features = features[mb]
            mb_raw_m = raw_m[mb]
            mb_fire = fire_acts[mb]
            mb_mine = mine_acts[mb]
            mb_old_logp = old_logp[mb]
            mb_adv = adv_t[mb]
            mb_ret = ret_t[mb]

            new_logp_m, entropy_m = maneuver_policy.evaluate_action(mb_features, mb_raw_m)
            new_logp_c, entropy_c = combat_policy.evaluate_action(
                mb_features, mb_fire, mb_mine
            )

            new_logp_m = new_logp_m.squeeze(-1)
            new_logp_c = new_logp_c.squeeze(-1)
            entropy_m = entropy_m.squeeze(-1) if entropy_m.dim() > 1 else entropy_m
            entropy_c = entropy_c.squeeze(-1) if entropy_c.dim() > 1 else entropy_c

            new_logp = new_logp_m + new_logp_c
            entropy = entropy_m + entropy_c

            log_ratio = new_logp - mb_old_logp

            # Skip mini-batch if any sample has drifted too far.
            # With ±4 combat clamp + 1e-4 tanh epsilon, legitimate
            # single-step log-ratios stay well within ±5.
            if log_ratio.max().item() > 5.0 or log_ratio.min().item() < -5.0:
                skipped_batches += 1
                continue

            ratio = torch.exp(log_ratio)

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            v_pred = value_net(mb_features)
            value_loss = nn.SmoothL1Loss()(v_pred, mb_ret)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            nn.utils.clip_grad_norm_(
                list(maneuver_policy.parameters())
                + list(combat_policy.parameters())
                + list(value_net.parameters()),
                max_norm=0.5,
            )

            optimizer.step()

            # log_std clamping moved to training loop (annealed there)

            total_loss_sum += float(loss.item())
            policy_loss_sum += float(policy_loss.item())
            value_loss_sum += float(value_loss.item())
            entropy_sum += float(entropy.mean().item())
            ratio_sum += float(ratio.mean().item())
            ratio_count += 1

        # Early stop remaining epochs if policy has drifted too far
        if ratio_count > 0 and abs(ratio_sum / ratio_count - 1.0) > 0.2:
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
#  Save model in existing bundle format 

def ppo_update_pooled(
    maneuver_policy, combat_policy, value_net, optimizer,
    episode_pool, clip_eps=0.2, entropy_coef=0.03, value_coef=0.25,
    epochs=2, mini_batch_size=32, gamma=0.99, lam=0.95,
):
    """PPO update from multiple episodes. Computes GAE per-episode to avoid
    bootstrapping across episode boundaries, then concatenates for minibatch SGD."""
    device = next(maneuver_policy.parameters()).device

    all_features, all_raw_m, all_fire, all_mine, all_old_logp = [], [], [], [], []
    all_adv, all_ret = [], []

    for traj in episode_pool:
        if len(traj) == 0:
            continue
        features = torch.cat([t["features"] for t in traj], dim=0).to(device)
        raw_m = torch.cat([t["raw_sample_m"] for t in traj], dim=0).to(device)
        fire_acts = torch.stack([t["fire_action"] for t in traj]).to(device)
        mine_acts = torch.stack([t["mine_action"] for t in traj]).to(device)
        old_logp = torch.stack([t["log_prob"] for t in traj]).to(device).squeeze(-1)
        rewards = [t["reward"] for t in traj]

        with torch.no_grad():
            values_np = value_net(features).cpu().numpy()

        advantages, returns = compute_gae(rewards, values_np, gamma, lam)

        all_features.append(features)
        all_raw_m.append(raw_m)
        all_fire.append(fire_acts)
        all_mine.append(mine_acts)
        all_old_logp.append(old_logp)
        all_adv.append(torch.tensor(advantages, dtype=torch.float32, device=device))
        all_ret.append(torch.tensor(returns, dtype=torch.float32, device=device))

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
            entropy = entropy_m + entropy_c

            log_ratio = new_logp - old_logp[mb]
            if log_ratio.max().item() > 5.0 or log_ratio.min().item() < -5.0:
                skipped_batches += 1
                continue

            ratio = torch.exp(log_ratio)
            surr1 = ratio * adv_t[mb]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            v_pred = value_net(features[mb])
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
            ratio_sum += float(ratio.mean().item())
            ratio_count += 1

        if ratio_count > 0 and abs(ratio_sum / ratio_count - 1.0) > 0.3:
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


def save_bundle(maneuver_policy, combat_policy, mu, sd, feature_cols, num_mfs, path):
    """Save in the same format as nf_train.py so nf_infer.py can load it."""
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

    torch.save(bundle, path)
    print(f"Saved maneuver bundle → {path}")

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
    print(f"Saved combat bundle → {combat_path}")


def save_rl_checkpoint(
    maneuver_policy, combat_policy, value_net, optimizer,
    episode, best_reward, path,
):
    """Save full RL training state so training can be resumed exactly."""
    torch.save({
        "maneuver_policy": maneuver_policy.state_dict(),
        "combat_policy": combat_policy.state_dict(),
        "value_net": value_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": episode,
        "best_reward": best_reward,
    }, path)
    print(f"Saved RL checkpoint (ep {episode}) → {path}")


def load_rl_checkpoint(
    maneuver_policy, combat_policy, value_net, optimizer, path, device,
):
    """Restore full RL training state. Returns (start_episode, best_reward)."""
    ckpt = torch.load(path, map_location=device)
    maneuver_policy.load_state_dict(ckpt["maneuver_policy"])
    combat_policy.load_state_dict(ckpt["combat_policy"])
    value_net.load_state_dict(ckpt["value_net"])
    optimizer.load_state_dict(ckpt["optimizer"])
    ep = ckpt.get("episode", 0)
    best = ckpt.get("best_reward", -float("inf"))
    print(f"Resumed RL checkpoint from ep {ep}, best_reward={best:.2f}")
    return ep, best



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--scenario", type=str, default="all")
    p.add_argument("--num_mfs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ppo_epochs", type=int, default=2)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.005)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--init_log_std", type=float, default=-0.7,
                   help="Initial exploration noise (log scale). "
                        "-0.5 ≈ std=0.6, -1.0 ≈ std=0.37, -2.0 ≈ std=0.14")
    p.add_argument("--mini_batch_size", type=int, default=256)
    p.add_argument("--resume", action="store_true",
                   help="Resume training from rl_checkpoint.pt")
    args = p.parse_args()



    here = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(here, "models")
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_inputs = 8  # length of FEATURE_COLS

    # Build policies 
    maneuver_policy = StochasticManeuverPolicy(
        num_inputs, args.num_mfs, init_log_std=args.init_log_std
    ).to(device)
    combat_policy = StochasticCombatPolicy(num_inputs, args.num_mfs).to(device)
    value_net = ValueNet(num_inputs).to(device)

    # Kill dropout in every SugenoNet for RL. Dropout makes forward()
    # stochastic w.r.t. the network old_logp and new_logp diverge even
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
        print("Warm-started maneuver policy from expert.")
    else:
        print("No maneuver.pt found — training from scratch!")
        feature_cols = [
            "dist", "ttc", "heading_err", "approach_speed",
            "ammo", "mines", "threat_density", "threat_angle",
        ]
    """if args.eval:
        with torch.no_grad():
            maneuver_policy.log_std[:] = math.log(0.08)"""
    if os.path.exists(combat_path):
        mu_c, sd_c = warm_start_combat(combat_policy, combat_path)
        print("Warm-started combat policy from expert.")
    else:
        print("No combat.pt found — training from scratch!")

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
    rl_ckpt_path = os.path.join(model_dir, "rl_checkpoint.pt")
    if args.resume and os.path.exists(rl_ckpt_path):
        start_ep, best_reward = load_rl_checkpoint(
            maneuver_policy, combat_policy, value_net, optimizer,
            rl_ckpt_path, device,
        )
        start_ep += 1  # resume from the next episode

    scenario_map = {
        "stock": sc.stock_scenario,
        "donut_ring": sc.donut_ring,
        "vertical_wall_left": sc.vertical_wall_left,
        "spiral_arms": sc.spiral_arms,
        "crossing_lanes": sc.crossing_lanes,
        "asteroid_rain": sc.asteroid_rain,
        "four_corner": sc.four_corner,
    }

    if args.scenario.lower() == "all":
        scenario_names = list(scenario_map.keys())
    else:
        scenario_names = [args.scenario]

    game_settings = {
        "perf_tracker": True,
        "graphics_type": GraphicsType.NoGraphics if not args.eval else GraphicsType.Tkinter,
        "realtime_multiplier":0.0,
        "graphics_obj": None,
        "frequency": 30,
    }
    game = KesslerGame(settings=game_settings)#type:ignore

    #  Training / Eval loop 
    rng = np.random.default_rng()
    episode_pool = []       # list of trajectories (each is a list of dicts)
    pool_steps = 0
    MIN_POOL_STEPS = 2048    # don't update PPO until this many steps
    train_start = time.perf_counter()

    for ep in range(start_ep, args.episodes + 1):
        if args.eval:
            scenario_name = scenario_names[(ep - 1) % len(scenario_names)]
        else:
            scenario_name = rng.choice(scenario_names)
        
        scenario = scenario_map[scenario_name]()

        controller = RLController(
            maneuver_policy, combat_policy,
            mu=mu, sd=sd,
            deterministic= args.eval,  # deterministic actions during evaluation
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
            episode_pool.append(traj)
            pool_steps += ep_steps

        # PPO Update — only when pool is large enough
        stats = None
        if pool_steps >= MIN_POOL_STEPS:
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

            # Anneal log_std: allow less noise as training progresses
            progress = ep / args.episodes
            max_log_std = -1.0 * (1 - progress) + -3.0 * progress
            with torch.no_grad():
                maneuver_policy.log_std.clamp_(-3.0, max_log_std)

            episode_pool = []
            pool_steps = 0

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
        skip_str = f" skip={stats['skipped']}" if stats["skipped"] > 0 else ""
        print(
            f"[{ep:04d}/{args.episodes}] {scenario_name:20s} | "
            f"R={ep_reward:7.2f} hits={hits} deaths={deaths} steps={ep_steps:4d} "
            f"std={log_std_val:.3f} "
            f"loss={stats['total_loss']:.4f} "
            f"pi={stats['policy_loss']:.4f} "
            f"v={stats['value_loss']:.4f} "
            f"H={stats['entropy']:.4f} "
            f"ratio={stats['ratio']:.3f}{skip_str} "
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
                os.path.join(model_dir, "maneuver.pt"),
            )
            save_rl_checkpoint(
                maneuver_policy, combat_policy, value_net, optimizer,
                ep, best_reward, rl_ckpt_path,
            )

        # Save best checkpoint whenever a new best episode appears
        print(f"DEBUG: ep_reward={ep_reward:.2f}, best_reward={best_reward:.2f}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            save_bundle(
                maneuver_policy,
                combat_policy,
                mu,
                sd,
                feature_cols,
                args.num_mfs,
                os.path.join(model_dir, "maneuver_best.pt"),
            )
            print(f"New best: {best_reward:.2f}")

    if not args.eval:
        #Final save
        save_bundle(
            maneuver_policy, combat_policy,
            mu, sd, feature_cols, args.num_mfs,
            os.path.join(model_dir, "maneuver.pt"),
        )
        save_rl_checkpoint(
            maneuver_policy, combat_policy, value_net, optimizer,
            args.episodes, best_reward, rl_ckpt_path,
        )
        print("\nDone. Models saved to models/")
        print("To resume: python rl_train.py --resume --episodes 600")
        print("To evaluate: python rl_train.py --eval --scenario all")
    total_time = time.perf_counter() - train_start
    minutes = total_time / 60
    print(f"\nTotal training time: {minutes:.1f} min ({total_time:.0f}s)")


if __name__ == "__main__":
    main()