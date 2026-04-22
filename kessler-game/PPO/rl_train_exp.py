"""
PPO training with shared-trunk actor policy.
Usage:
  python rl_train.py --episodes 6000 --scenario all
  python rl_train.py --eval --scenario all
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
from rl_policy_exp import SharedActorPolicy, ValueNet
from rl_controller_exp import RLController, FEATURE_COLS

# Must match the number of entries in scenario_map in main().
NUM_SCENARIOS = 8


#  PPO update 
#Generalized Advantage Estimation, computes advantages and returns from rewards and value estimates for a single episode/trajectory. No bootstrapping across episode boundaries.
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    return advantages, returns



# PPO update from multiple episodes. One evaluate_action call handles all 4 actions
# through the shared trunk. GAE per-episode, per-episode advantage normalization,
# post-GAE episode cap, separate actor/critic backward passes, KL early stopping.
def ppo_update_pooled(
    actor, value_net, opt_policy, opt_critic,
    episode_pool, clip_eps=0.2, entropy_coef=0.01, value_coef=1.0,
    epochs=1, mini_batch_size=512, gamma=0.99, lam=0.95,
    max_steps_per_episode=None, num_scenarios=NUM_SCENARIOS):

    device = next(actor.parameters()).device
    all_features, all_raw_m, all_fire, all_mine, all_old_logp = [], [], [], [], []
    all_adv, all_ret = [], []
    all_scenario_ctx = []
    raw_adv_stds = []

    for traj in episode_pool:
        if len(traj) == 0:
            continue
        features = torch.cat([t["features"] for t in traj], dim=0).to(device)
        raw_m = torch.cat([t["raw_sample_m"] for t in traj], dim=0).to(device)
        fire_acts = torch.stack([t["fire_action"] for t in traj]).to(device)
        mine_acts = torch.stack([t["mine_action"] for t in traj]).to(device)
        old_logp = torch.stack([t["log_prob"] for t in traj]).to(device).squeeze(-1)
        rewards = [t["reward"] for t in traj]

        #Build scenario one-hot for critic AND actor (both see it now through the trunk)
        sc_id = traj[0].get("scenario_id", 0)
        sc_onehot = torch.zeros(features.shape[0], num_scenarios, device=device)
        sc_onehot[:, sc_id] = 1.0

        with torch.no_grad():
            values_np = value_net(torch.cat([features, sc_onehot], dim=1)).cpu().numpy()

        advantages, returns = compute_gae(rewards, values_np, gamma, lam)

        #Per-episode advantage normalization
        adv_ep = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_ep = torch.tensor(returns, dtype=torch.float32, device=device)
        raw_adv_stds.append(float(adv_ep.std()) if adv_ep.numel() > 1 else 0.0)
        if adv_ep.numel() > 1 and adv_ep.std() > 1e-8:#avoid division by zero
            adv_ep = (adv_ep - adv_ep.mean()) / (adv_ep.std() + 1e-8)
        adv_ep = torch.clamp(adv_ep, -5.0, 5.0)

        #Truncate AFTER GAE so boundary advantages have correct bootstraps
        T = features.shape[0]
        cap = max_steps_per_episode or 0
        if cap > 0 and T > cap:
            start = int(torch.randint(0, T - cap + 1, (1,)).item())
            sl = slice(start, start + cap)
            features  = features[sl]
            raw_m     = raw_m[sl]
            fire_acts = fire_acts[sl]
            mine_acts = mine_acts[sl]
            old_logp  = old_logp[sl]
            adv_ep    = adv_ep[sl]
            ret_ep    = ret_ep[sl]
            sc_onehot = sc_onehot[sl]

        all_features.append(features)
        all_raw_m.append(raw_m)
        all_fire.append(fire_acts)
        all_mine.append(mine_acts)
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
    fire_acts = torch.cat(all_fire, dim=0)
    mine_acts = torch.cat(all_mine, dim=0)
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
    target_kl = 0.15

    for _ in range(epochs):
        idxs = torch.randperm(N, device=device)
        stop_early = False

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
            mb_sc = scenario_ctx[mb]

            # One evaluate_action call through the shared trunk
            new_logp, entropy = actor.evaluate_action(
                mb_features, mb_raw_m, mb_fire, mb_mine, mb_sc)

            log_ratio = new_logp - mb_old_logp
            log_ratio = log_ratio.clamp(-4.0, 4.0)
            ratio = torch.exp(log_ratio)

            mean_ratio = float(ratio.mean().item())
            approx_kl = float((mb_old_logp - new_logp).mean().abs().item())

            # Skip guard
            if mean_ratio > 10.0 or mean_ratio < 0.1:
                skipped_batches += 1
                continue

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            v_pred = value_net(torch.cat([mb_features, mb_sc], dim=1))
            value_loss = nn.SmoothL1Loss()(v_pred, mb_ret)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            # critic step always runs
            opt_critic.zero_grad(set_to_none=True)
            critic_obj = value_coef * value_loss

            # policy step (only if unfrozen)
            policy_params = list(actor.parameters())
            policy_trainable = any(p.requires_grad for p in policy_params)

            if policy_trainable:
                opt_policy.zero_grad(set_to_none=True)
                policy_obj = policy_loss - entropy_coef * entropy.mean()
                policy_obj.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(policy_params, max_norm=0.5)
                opt_policy.step()

            critic_obj.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
            opt_critic.step()

            total_loss_sum += float(loss.item())
            policy_loss_sum += float(policy_loss.item())
            value_loss_sum += float(value_loss.item())
            entropy_sum += float(entropy.mean().item())
            ratio_sum += mean_ratio
            ratio_count += 1

            # Early stop if policy moved too much
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
Tracks ppo_update_count so exploration decays evenly regardless of how many episodes it takes to fill the pool.
"""
def anneal_log_std_(actor, ppo_update_count, expected_updates):
    progress = min(1.0, ppo_update_count / max(expected_updates, 1))
    target_log_std = -1.10 * (1.0 - progress) + -2.30 * progress
    with torch.no_grad():
        target = torch.full_like(actor.log_std, target_log_std)
        actor.log_std.lerp_(target, 0.25)
        actor.log_std.clamp_(-2.30, -0.90)


# Save actor checkpoint (full state dict + normalization stats)
def save_actor_bundle(actor, mu, sd, feature_cols, path):
    bundle = {
        "task": "rl_exp",
        "actor_state_dict": {k: v.cpu() for k, v in actor.state_dict().items()},
        "mu": mu.tolist() if mu is not None else None,
        "sd": sd.tolist() if sd is not None else None,
        "feature_cols": feature_cols,
    }
    torch.save(bundle, path)
    print(f"Saved actor bundle -> {path}")


def save_rl_checkpoint(
    actor, value_net, opt_policy, opt_critic,
    episode, best_reward, total_episodes, ppo_update_count, path,
):
    torch.save({
        "actor": actor.state_dict(),
        "value_net": value_net.state_dict(),
        "opt_policy": opt_policy.state_dict(),
        "opt_critic": opt_critic.state_dict(),
        "episode": episode,
        "best_reward": best_reward,
        "total_episodes": total_episodes,
        "ppo_update_count": ppo_update_count,
    }, path)
    print(f"Saved RL checkpoint (ep {episode}) -> {path}")


def load_rl_checkpoint(actor, value_net, opt_policy, opt_critic, path, device):
    ckpt = torch.load(path, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    value_net.load_state_dict(ckpt["value_net"])
    opt_policy.load_state_dict(ckpt["opt_policy"])
    opt_critic.load_state_dict(ckpt["opt_critic"])
    ep = ckpt.get("episode", 0)
    best = ckpt.get("best_reward", -float("inf"))
    total_ep = ckpt.get("total_episodes", None)
    ppo_count = ckpt.get("ppo_update_count", 0)
    print(f"Resumed RL checkpoint from ep {ep}, best_reward={best:.2f}, ppo_updates={ppo_count}")
    return ep, best, total_ep, ppo_count



def load_actor_bundle(actor, path, device):
    bundle = torch.load(path, map_location=device)
    actor.load_state_dict(bundle["actor_state_dict"])

    mu = np.array(bundle["mu"], dtype=np.float32) if bundle.get("mu") is not None else None
    sd = np.array(bundle["sd"], dtype=np.float32) if bundle.get("sd") is not None else None
    feature_cols = bundle.get("feature_cols")

    print(f"Loaded actor bundle from {path}")
    return mu, sd, feature_cols

def main():
    p = argparse.ArgumentParser()
    #absolute behemoth of a cli
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--scenario", type=str, default="all")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--critic_lr", type=float, default=3e-4,
                   help="Learning rate for value network")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--init_log_std", type=float, default=-1.0,
                   help="Initial exploration noise (log scale). "
                        "-0.5 = std=0.6, -1.0 = std=0.37, -2.0 = std=0.14")
    p.add_argument("--mini_batch_size", type=int, default=512)
    p.add_argument("--max_steps_per_episode", type=int, default=512)
    p.add_argument("--warmup_episodes", type=int, default=50,
                   help="Freeze actor for N episodes so critic can learn before PPO pushes the actor")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from rl_checkpoint.pt")
    p.add_argument("--trunk_hidden", type=int, default=128)
    p.add_argument("--trunk_out", type=int, default=64)
    args = p.parse_args()



    here = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(here, "models_exp/")
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = len(FEATURE_COLS)  # 8

    # Build shared-trunk actor + separate critic
    actor = SharedActorPolicy(
        num_features=num_features,
        num_scenarios=NUM_SCENARIOS,
        trunk_hidden=args.trunk_hidden,
        trunk_out=args.trunk_out,
        init_log_std=args.init_log_std,
    ).to(device)
    value_net = ValueNet(num_features + NUM_SCENARIOS).to(device)

    print(f"Actor params: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic params: {sum(p.numel() for p in value_net.parameters()):,}")

    mu, sd = None, None
    feature_cols = list(FEATURE_COLS)

    if args.eval:
        best_path = os.path.join(model_dir, "actor_best.pt")
        rl_path = os.path.join(model_dir, "actor_rl.pt")

        if os.path.exists(best_path):
            mu, sd, feature_cols = load_actor_bundle(actor, best_path, device)
        elif os.path.exists(rl_path):
            mu, sd, feature_cols = load_actor_bundle(actor, rl_path, device)
        else:
            print("No saved actor bundle found for eval; evaluating fresh model.")
    else:
        mu_path = os.path.join(model_dir, "norm_stats.npz")
        if os.path.exists(mu_path):
            stats = np.load(mu_path)
            mu = stats["mu"].astype(np.float32)
            sd = stats["sd"].astype(np.float32)
            print(f"Loaded normalization stats from {mu_path}")
        else:
            old_bundle_path = os.path.join(model_dir, "maneuver.pt")
            if os.path.exists(old_bundle_path):
                old_bundle = torch.load(old_bundle_path, map_location="cpu")
                info = old_bundle.get("heads", {}).get("thrust", {})
                if info.get("mu") is not None:
                    mu = np.array(info["mu"], dtype=np.float32)
                    sd = np.array(info["sd"], dtype=np.float32)
                    print(f"Extracted normalization stats from old bundle {old_bundle_path}")
            if mu is None:
                print("No normalization stats found. Training with raw features.")

    # Separate optimizers. Exclude log_std from policy optimizer (controlled by anneal only).
    policy_params = [p for n, p in actor.named_parameters() if "log_std" not in n]
    opt_policy = optim.Adam(policy_params, lr=args.lr)
    opt_critic = optim.Adam(value_net.parameters(), lr=args.critic_lr)

    # Resume from checkpoint if requested
    start_ep = 1
    best_reward = -float("inf")
    total_episodes = args.episodes
    ppo_update_count = 0
    rl_ckpt_path = os.path.join(model_dir, "rl_checkpoint.pt")
    if args.resume and os.path.exists(rl_ckpt_path):
        start_ep, best_reward, saved_total, ppo_update_count = load_rl_checkpoint(
            actor, value_net, opt_policy, opt_critic,
            rl_ckpt_path, device,
        )
        start_ep += 1
        if saved_total is not None:
            total_episodes = saved_total

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
        "realtime_multiplier": 1.0 if args.eval else 0.0,
        "graphics_obj": None,
        "frequency": 30,
    }
    game = KesslerGame(settings=game_settings)#type:ignore

    #  Training/Eval loop 
    rng = np.random.default_rng()
    episode_pool = []
    pool_steps = 0
    pool_scenarios = set()
    MIN_POOL_STEPS = 2048
    MIN_POOL_SCENARIOS = 4
    base_lr = args.lr
    cooldown_until = 0
    train_start = time.perf_counter()

    # Estimate expected PPO updates for anneal schedule
    expected_updates = max(1, (args.episodes - args.warmup_episodes) // 3)

    # Critic warmup: freeze actor so critic can learn value estimates before PPO pushes the actor.
    # More important now since we're training from scratch (no warm start).
    policy_frozen = False
    if args.warmup_episodes > 0 and not args.eval:
        for p in actor.parameters():
            p.requires_grad = False
        policy_frozen = True
        print(f"Actor frozen for first {args.warmup_episodes} episodes (critic warmup)")

    for ep in range(start_ep, args.episodes + 1):
        # unfreeze actor once warmup is done
        if policy_frozen and ep > args.warmup_episodes:
            for p in actor.parameters():
                p.requires_grad = True
            policy_frozen = False
            print(f"Actor unfrozen at episode {ep} (critic warmup done)")

        if args.eval:
            scenario_name = eval_scenario_names[(ep - 1) % len(eval_scenario_names)]
        else:
            scenario_name = rng.choice(train_scenario_names)
        
        scenario = scenario_map[scenario_name]()
        sc_idx = scenario_to_idx[scenario_name]

        controller = RLController(
            actor,
            mu=mu, sd=sd,
            deterministic=args.eval,
            scenario_id=sc_idx,
            num_scenarios=num_scenarios,
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

        # Add episode to pool
        if ep_steps > 0:
            for step in traj:
                step["scenario_id"] = sc_idx
            episode_pool.append(traj)
            cap = args.max_steps_per_episode or len(traj)
            pool_steps += min(len(traj), cap)
            pool_scenarios.add(scenario_name)


        #PPO Update — only when pool has enough steps AND scenario diversity
        stats = None
        if pool_steps >= MIN_POOL_STEPS and len(pool_scenarios) >= MIN_POOL_SCENARIOS:
            stats = ppo_update_pooled(
                actor,
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

            # Anneal exploration (tracks actual PPO updates, not episodes)
            if not policy_frozen:
                anneal_log_std_(actor, ppo_update_count, expected_updates)

            episode_pool = []
            pool_steps = 0
            pool_scenarios = set()

            # Restore LR if cooldown expired
            if ep >= cooldown_until and cooldown_until > 0:
                for g in opt_policy.param_groups:
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
                "n_episodes": 0,
                "adv_std_spread": 0.0,
            }


        #Logging 
        log_std_val = actor.log_std.exp().mean().item()
        log_std_raw = actor.log_std.detach().cpu().tolist()
        skip_str = f" skip={stats['skipped']}" if stats["skipped"] > 0 else ""
        warmup_str = " [WARMUP]" if policy_frozen else ""
        diag_str = ""
        if stats["n_episodes"] > 0:
            diag_str = (f" pool={stats['n_episodes']}ep"
                        f" adv_spread={stats['adv_std_spread']:.1f}x"
                        f" ppo#{ppo_update_count}")
        print(
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

        #Save checkpoints 
        if ep % args.save_every == 0:
            save_actor_bundle(
                actor, mu, sd, feature_cols,
                os.path.join(model_dir, "actor_rl.pt"),
            )
            save_rl_checkpoint(
                actor, value_net, opt_policy, opt_critic,
                ep, best_reward, total_episodes, ppo_update_count, rl_ckpt_path,
            )

        # Deterministic eval across ALL scenarios to pick best checkpoint
        if ep % args.save_every == 0:
            total_det_reward = 0.0
            total_det_hits = 0
            total_det_deaths = 0
            eval_details = []

            for eval_name in eval_scenario_names:
                eval_scenario = scenario_map[eval_name]()
                eval_sc_idx = scenario_to_idx[eval_name]
                eval_ctrl = RLController(
                    actor,
                    mu=mu, sd=sd, deterministic=True,
                    scenario_id=eval_sc_idx,
                    num_scenarios=num_scenarios,
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
                f" [DET-EVAL] avg={avg_det_reward:.1f} "
                f"total_hits={total_det_hits} total_deaths={total_det_deaths}"
            )
            print(f"    {', '.join(eval_details)}")

            if avg_det_reward > best_reward:
                best_reward = avg_det_reward
                save_actor_bundle(
                    actor, mu, sd, feature_cols,
                    os.path.join(model_dir, "actor_best.pt"),
                )
                print(f" New best (deterministic avg): {best_reward:.2f}")
                # Cooldown: halve LR for 75 ep to protect the checkpoint
                for g in opt_policy.param_groups:
                    g["lr"] = base_lr * 0.5
                cooldown_until = ep + 75
                print(f" LR cooldown: {base_lr:.1e} -> {base_lr*0.5:.1e} until ep {cooldown_until}")

    if not args.eval:
        #Final save
        save_actor_bundle(
            actor, mu, sd, feature_cols,
            os.path.join(model_dir, "actor_rl.pt"),
        )
        save_rl_checkpoint(
            actor, value_net, opt_policy, opt_critic,
            args.episodes, best_reward, total_episodes, ppo_update_count, rl_ckpt_path,
        )
        print("\nDone. Models saved to models_exp/")
    total_time = time.perf_counter() - train_start
    minutes = total_time / 60
    print(f"\nTotal training time: {minutes:.1f} min ({total_time:.0f}s)")


if __name__ == "__main__":
    main()
