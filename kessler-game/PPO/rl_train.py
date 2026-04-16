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
"""

import argparse
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

# Number of scenarios in the fixed scenario map. Used to size the
# critic's one-hot context input AND now the actor's scenario bias layers.
# Must match the scenario_map in main().
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
# Includes per-episode advantage normalization, episode cap with post-GAE truncation,
# separate policy/critic optimization, and KL early stopping.
def ppo_update_pooled(
    maneuver_policy, value_net, opt_policy, opt_critic,
    episode_pool, clip_eps=0.2, entropy_coef=0.01, value_coef=1.0,
    epochs=1, mini_batch_size=512, gamma=0.99, lam=0.95,
    max_steps_per_episode=None, num_scenarios=NUM_SCENARIOS):

    device = next(maneuver_policy.parameters()).device
    # Pool data across episodes, compute advantages and returns per episode to avoid bootstrapping across episode boundaries.
    all_features, all_raw_m, all_old_logp = [], [], []
    all_adv, all_ret = [], []
    all_scenario_ctx = []  # one-hot scenario context for both critic and actor now
    raw_adv_stds = []  # diagnostic: track pre-normalization advantage spread

    for traj in episode_pool:
        if len(traj) == 0:
            continue
        features = torch.cat([t["features"] for t in traj], dim=0).to(device)
        raw_m = torch.cat([t["raw_sample_m"] for t in traj], dim=0).to(device)
        old_logp = torch.stack([t["log_prob"] for t in traj]).to(device).squeeze(-1)
        rewards = [t["reward"] for t in traj]

        #Build scenario one-hot for critic context AND actor conditioning
        sc_id = traj[0].get("scenario_id", 0)#default to 0 if not present, but it should always be there
        sc_onehot = torch.zeros(features.shape[0], num_scenarios, device=device)#shape (T, num_scenarios)
        sc_onehot[:, sc_id] = 1.0#one-hot encoding of the scenario

        with torch.no_grad():
            values_np = value_net(torch.cat([features, sc_onehot], dim=1)).cpu().numpy()

        advantages, returns = compute_gae(rewards, values_np, gamma, lam)

        #Per-episode advantage normalization
        # Normalize advantages within each episode BEFORE concatenating.
        # This prevents high-reward scenarios (crossing_lanes) from drowning out low-reward ones (sniper_practice) after global
        # normalization, which was causing the scenario tug-of-war.
        adv_ep = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_ep = torch.tensor(returns, dtype=torch.float32, device=device)
        raw_adv_stds.append(float(adv_ep.std()) if adv_ep.numel() > 1 else 0.0)
        if adv_ep.numel() > 1 and adv_ep.std() > 1e-8:
            adv_ep = (adv_ep - adv_ep.mean()) / (adv_ep.std() + 1e-8)
        adv_ep = torch.clamp(adv_ep, -5.0, 5.0)

        #Truncate AFTER GAE so boundary advantages have correct bootstraps
        #before: truncation happened before GAE, which forced the last step of a mid-episode window to bootstrap with V=0 (as if the episode ended).
        #Now GAE sees the full episode, then slice the result.
        T = features.shape[0]
        cap = max_steps_per_episode or 0
        if cap > 0 and T > cap:
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
    #Kullback–Leibler divergence: a measure of how much the new policy diverged from the old one. Used for early stopping to prevent destructive updates.
    target_kl = 0.05

    for _ in range(epochs):
        idxs = torch.randperm(N, device=device)
        stop_early = False

        for start in range(0, N, mini_batch_size):
            end = min(start + mini_batch_size, N)
            mb = idxs[start:end]

            mb_features = features[mb]
            mb_raw_m = raw_m[mb]
            mb_old_logp = old_logp[mb]
            mb_adv = adv_t[mb]
            mb_ret = ret_t[mb]
            mb_sc = scenario_ctx[mb]

            # Actor evaluate_action now gets scenario context too
            new_logp_m, entropy_m = maneuver_policy.evaluate_action(
                mb_features, mb_raw_m, scenario_onehot=mb_sc)

            new_logp_m = new_logp_m.squeeze(-1)
            entropy_m = entropy_m.squeeze(-1) if entropy_m.dim() > 1 else entropy_m

            new_logp = new_logp_m
            entropy = entropy_m

            log_ratio = new_logp - mb_old_logp
            log_ratio = log_ratio.clamp(-4.0, 4.0)

            ratio = torch.exp(log_ratio)

            mean_ratio = float(ratio.mean().item())
            approx_kl = float((mb_old_logp - new_logp).mean().abs().item())

            # Skip guard: block destructive minibatches
            if mean_ratio > 3.0 or mean_ratio < 0.33:

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

            # only run policy backward if policy params are unfrozen
            policy_params = list(maneuver_policy.parameters())
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

            # Early stop this PPO pass if policy moved too much
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
def anneal_log_std_(maneuver_policy, ppo_update_count, expected_updates):

    progress = min(1.0, ppo_update_count / max(expected_updates, 1))

    target_log_std = -1.10 * (1.0 - progress) + -2.30 * progress

    with torch.no_grad():
        target = torch.full_like(maneuver_policy.log_std, target_log_std)
        maneuver_policy.log_std.lerp_(target, 0.25)
        maneuver_policy.log_std.clamp_(-2.30, -0.90)

#Save in the same format as nf_train.py so nf_infer.py can load it.
# Also saves scenario bias weights so they persist across runs.
def save_bundle(maneuver_policy, mu, sd, feature_cols, num_mfs, path):
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

    # Save scenario bias weights so they can be restored on resume
    if maneuver_policy.num_scenarios > 0:
        bundle["scenario_maneuver"] = {
            k: v.cpu() for k, v in maneuver_policy.state_dict().items()
            if "scenario_bias" in k
        }

    torch.save(bundle, path)
    print(f"Saved maneuver bundle -> {path}")



def save_rl_checkpoint(
    maneuver_policy, value_net, opt_policy, opt_critic,
    episode, best_reward, total_episodes, ppo_update_count, path,
):
    #saves everything needed to resume training, including total_episodes for annealing and ppo_update_count for the fixed anneal schedule
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


def load_rl_checkpoint(
    maneuver_policy, value_net, opt_policy, opt_critic, path, device,
):
    #restores full training state, returns (episode, best_reward, total_episodes, ppo_update_count)
    ckpt = torch.load(path, map_location=device)
    # strict=False: old checkpoints won't have scenario_bias_* keys, that's fine
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
    # Support loading from old single-optimizer checkpoints
    if "opt_policy" in ckpt:
        opt_policy.load_state_dict(ckpt["opt_policy"])
        if critic_reloaded:
            opt_critic.load_state_dict(ckpt["opt_critic"])
    elif "optimizer" in ckpt:
        print("  (old single-optimizer checkpoint — skipping optimizer state)")
    ep = ckpt.get("episode", 0)
    best = ckpt.get("best_reward", -float("inf"))
    total_ep = ckpt.get("total_episodes", None) # None if old checkpoint
    ppo_count = ckpt.get("ppo_update_count", 0) # 0 if old checkpoint
    print(f"Resumed RL checkpoint from ep {ep}, best_reward={best:.2f}, ppo_updates={ppo_count}")
    return ep, best, total_ep, ppo_count


def run_eval_sweep(game, scenario_names, scenario_map, scenario_to_idx,
                   maneuver_policy, mu, sd, num_scenarios):
    results = []

    for eval_name in scenario_names:
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



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--scenario", type=str, default="all")
    p.add_argument("--num_mfs", type=int, default=2,
                   help="Must match the warm-start model")
    p.add_argument("--init_bundle", type=str, default=None,
                   help="Optional warm-start bundle path. If omitted, training uses models/maneuver.pt and eval uses models/maneuver_best.pt.")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--critic_lr", type=float, default=3e-4,
                   help="Learning rate for value network (higher than policy LR)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--graphics", action="store_true",
                   help="Render the game window. Off by default for faster training and evaluation.")
    p.add_argument("--init_log_std", type=float, default=-1.0,
                   help="Initial exploration noise (log scale). "
                        "-0.5 ≈ std=0.6, -1.0 ≈ std=0.37, -2.0 ≈ std=0.14")
    p.add_argument("--mini_batch_size", type=int, default=512)
    p.add_argument("--max_steps_per_episode", type=int, default=512)
    p.add_argument("--warmup_episodes", type=int, default=0)
    p.add_argument("--min_pool_steps", type=int, default=2048)
    p.add_argument("--min_pool_scenarios", type=int, default=4)
    p.add_argument("--cooldown_episodes", type=int, default=25)
    p.add_argument("--cooldown_lr_scale", type=float, default=0.5)
    p.add_argument("--early_stop_patience", type=int, default=3,
                   help="Stop after this many save/eval sweeps without a new best train-scenario deterministic average. Set to 0 to disable.")
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--resume", action="store_true",
                   help="Resume training from rl_checkpoint.pt")
    args = p.parse_args()



    here = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(here, "models")
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_inputs = 8  # length of FEATURE_COLS

    # Build policies (actor now gets scenario context via num_scenarios)
    maneuver_policy = StochasticManeuverPolicy(
        num_inputs, args.num_mfs,
        num_scenarios=NUM_SCENARIOS,
        init_log_std=args.init_log_std,
    ).to(device)
    value_net = ValueNet(num_inputs + NUM_SCENARIOS).to(device)
    
    # Kill dropout in every warm-started SugenoNet for RL. Dropout makes forward()
    # stochastic wrt the network old_logp and new_logp diverge even
    # before any gradient step because different masks produce different
    # outputs from the same input, breaking PPO importance sampling.
    # Identity swap is permanent and can't be undone by an accidental .train().
    for net in [maneuver_policy.thrust_net, maneuver_policy.turn_net]:
        net.dropout = nn.Identity()

    #  Warm-start from expert-trained models 
    if args.init_bundle is not None:
        maneuver_path = args.init_bundle
        if not os.path.isabs(maneuver_path):
            maneuver_path = os.path.join(here, maneuver_path)
    elif args.eval:
        maneuver_path = os.path.join(model_dir, "maneuver_best.pt")
    else:
        maneuver_path = os.path.join(model_dir, "maneuver.pt")

    mu, sd, feature_cols = None, None, None
    if os.path.exists(maneuver_path):
        mu, sd, feature_cols = warm_start_maneuver(maneuver_policy, maneuver_path)
        # Restore trained log_std if available in the bundle
        _bundle = torch.load(maneuver_path, map_location="cpu")
        if "log_std" in _bundle:
            with torch.no_grad():
                maneuver_policy.log_std.copy_(torch.tensor(_bundle["log_std"]))
            print(f"Restored log_std: {maneuver_policy.log_std.data.tolist()}")
        # Restore scenario bias weights if present in bundle (from a previous v2 run)
        if "scenario_maneuver" in _bundle and maneuver_policy.num_scenarios > 0:
            maneuver_policy.load_state_dict(_bundle["scenario_maneuver"], strict=False)
            print("Restored maneuver scenario bias from bundle")
        print("Warm started maneuver policy from expert.")
    else:
        print("No maneuver.pt found —> training from scratch")
        
        feature_cols = [
            "dist", "ttc", "heading_err", "approach_speed",
            "ammo", "mines", "threat_density", "threat_angle",
        ]

    ###

    # Separate optimizers: critic needs a higher LR to keep up with
    # the changing policy across diverse scenarios.
    # Exclude log_std from the policy optimizer — it is controlled
    # entirely by anneal_log_std_() on a schedule. Having it in both
    # the optimizer (pulled up by entropy bonus) and the annealer
    # (pulled down) creates a tug-of-war that prevents clean decay.
    policy_params = (
        [p for n, p in maneuver_policy.named_parameters() if "log_std" not in n])
    opt_policy = optim.Adam(policy_params, lr=args.lr)
    opt_critic = optim.Adam(value_net.parameters(), lr=args.critic_lr)

    # Resume from RL checkpoint if requested
    start_ep = 1
    best_reward = -float("inf")
    total_episodes = args.episodes # save the original training horizon for annealing
    ppo_update_count = 0
    rl_ckpt_path = os.path.join(model_dir, "rl_checkpoint.pt")
    if args.resume and os.path.exists(rl_ckpt_path):
        start_ep, best_reward, saved_total, ppo_update_count = load_rl_checkpoint(
            maneuver_policy, value_net, opt_policy, opt_critic,
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
        "sniper_practice": sc.sniper_practice,
    }
    # Fixed index for critic one-hot context. Must match NUM_SCENARIOS.
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
        "graphics_type": GraphicsType.Tkinter if args.graphics else GraphicsType.NoGraphics,
        "realtime_multiplier": 1.0 if args.graphics else 0.0,
        "graphics_obj": None,
        "frequency": 30,
    }
    game = KesslerGame(settings=game_settings)#type:ignore

    #  Training/Eval loop 
    rng = np.random.default_rng()
    episode_pool = [] # list of trajectories (each is a list of dicts)
    pool_steps = 0
    pool_scenarios = set()# track distinct scenarios in current pool
    base_lr = args.lr
    cooldown_until = 0 # episode at which LR cooldown expires
    train_start = time.perf_counter()

    # Estimate expected PPO updates for anneal schedule.
    # With MIN_POOL_STEPS=2048 and ~1800 steps/episode, roughly 1 update
    # per 1-2 episodes. Conservative: ~1 per 3 episodes accounting for
    # the scenario diversity gate slowing things down.
    expected_updates = max(1, (args.episodes - args.warmup_episodes) // 3)
    if not (0.0 < args.cooldown_lr_scale <= 1.0):
        raise ValueError("--cooldown_lr_scale must be in (0, 1].")

    # critic warmup: freeze policy params so only value_net trains, let the critic learn what states are worth before PPO starts pushing the actor around with bad advantage estimates
    policy_frozen = False
    no_improve_evals = 0
    final_episode = start_ep - 1
    if args.warmup_episodes > 0 and not args.eval:
        for p in maneuver_policy.parameters():
            p.requires_grad = False

        policy_frozen = True
        print(f"Policy frozen for first {args.warmup_episodes} episodes (critic warmup)")

    if args.init_bundle is not None and not args.eval and not args.resume:
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
        # unfreeze policy once warmup is done
        if policy_frozen and ep > args.warmup_episodes:
            for p in maneuver_policy.parameters():
                p.requires_grad = True
            policy_frozen = False
            print(f"Policy unfrozen at episode {ep} (critic warmup done)")

        if args.eval:
            scenario_name = eval_scenario_names[(ep - 1) % len(eval_scenario_names)]
        else:
            scenario_name = rng.choice(train_scenario_names)
        
        scenario = scenario_map[scenario_name]()
        sc_idx = scenario_to_idx[scenario_name]

        controller = RLController(
            maneuver_policy,
            mu=mu, sd=sd,
            deterministic=args.eval,  # deterministic during evaluation
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

        # Add episode to pool (keep separate for correct GAE)
        if ep_steps > 0:
            for step in traj:
                step["scenario_id"] = sc_idx
            episode_pool.append(traj)
            # Count capped steps for pool-size gating
            cap = args.max_steps_per_episode or len(traj)
            pool_steps += min(len(traj), cap)
            pool_scenarios.add(scenario_name)


        #PPO Update — only when pool has enough steps AND scenario diversity
        stats = None
        if pool_steps >= args.min_pool_steps and len(pool_scenarios) >= args.min_pool_scenarios:
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

            # Anneal exploration (tracks actual PPO updates, not episodes)
            if not policy_frozen:
                anneal_log_std_(maneuver_policy, ppo_update_count, expected_updates)

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
        log_std_val = maneuver_policy.log_std.exp().mean().item()
        log_std_raw = maneuver_policy.log_std.detach().cpu().tolist()
        skip_str = f" skip={stats['skipped']}" if stats["skipped"] > 0 else ""
        warmup_str = " [WARMUP]" if policy_frozen else ""
        diag_str = ""
        if stats["n_episodes"] > 0:
            diag_str = (f" pool={stats['n_episodes']}ep"
                        f" adv_spread={stats['adv_std_spread']:.1f}x"
                        f" ppo#{ppo_update_count}")
        print(
            f"[{ep:04d}/{args.episodes}] {scenario_name:20s} | " # scenario name padded to 20 chars for alignment
            f"R={ep_reward:7.2f} hits={hits} deaths={deaths} steps={ep_steps:4d} "# episode stats
            f"std={log_std_val:.3f} "# current exploration noise level
            f"loss={stats['total_loss']:.4f} "# PPO loss (for debugging, not a great performance metric on its own)
            f"pi={stats['policy_loss']:.4f} "# policy loss component (how much the actor is updating)
            f"v={stats['value_loss']:.4f} "# value loss component (how much the critic is updating)
            f"H={stats['entropy']:.4f} "# maneuver policy entropy
            f"ratio={stats['ratio']:.3f}{skip_str}{diag_str}{warmup_str} "# PPO ratio (clipped vs unclipped policy update magnitude, should be near 1.0 if updates are stable)
            f"log_std_raw={log_std_raw} "# raw log_std values for debugging
            f"({dt:.1f}s, elapsed={((time.perf_counter()-train_start)/60):.1f}m)"
        )

        #Save checkpoints 
        # Save rolling checkpoint every N episodes
        if ep % args.save_every == 0:
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

        # Run deterministic eval across ALL scenarios to pick best checkpoint.
        # Single-scenario eval was misleading because a policy good at one
        # scenario but bad at others could look much better or worse than it
        # really is overall.
        if ep % args.save_every == 0:
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
                # Cooldown after a new best to reduce immediate regressions.
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

    if not args.eval:
        #Final save
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
    total_time = time.perf_counter() - train_start
    minutes = total_time / 60
    print(f"\nTotal training time: {minutes:.1f} min ({total_time:.0f}s)")


if __name__ == "__main__":
    main()
