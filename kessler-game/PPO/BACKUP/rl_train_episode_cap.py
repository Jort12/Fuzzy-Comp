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



# Number of scenarios in the fixed scenario map. Used to size the
# critic's one-hot context input. Must match the scenario_map in main().
NUM_SCENARIOS = 8

#Generalized Advantage Estimation.
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
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


"""
Experimental
PPO update from multiple episodes.

Computes GAE per episode, concatenates everything, then does minibatch SGD.
This version adds real safety checks so a warm-started policy does not get
shoved too far in one update.
"""
def ppo_update_pooled(
    maneuver_policy, combat_policy, value_net, opt_policy, opt_critic,
    episode_pool, clip_eps=0.2, entropy_coef=0.01, value_coef=0.25,
    epochs=1, mini_batch_size=512, gamma=0.99, lam=0.95,
    max_steps_per_episode=None, num_scenarios=NUM_SCENARIOS):

    device = next(maneuver_policy.parameters()).device

    all_features, all_raw_m, all_fire, all_mine, all_old_logp = [], [], [], [], []
    all_adv, all_ret = [], []
    all_scenario_ctx = []  # one-hot scenario context for critic only
    raw_adv_stds = []  # diagnostic: track pre-normalization advantage spread

    for traj in episode_pool:
        if len(traj) == 0:
            continue

        features = torch.cat([t["features"] for t in traj], dim=0).to(device)
        raw_m = torch.cat([t["raw_sample_m"] for t in traj], dim=0).to(device)
        fire_acts = torch.stack([t["fire_action"] for t in traj]).to(device)
        mine_acts = torch.stack([t["mine_action"] for t in traj]).to(device)
        old_logp = torch.stack([t["log_prob"] for t in traj]).to(device).squeeze(-1)
        rewards = [t["reward"] for t in traj]

        # Build scenario one-hot for critic context (policy doesn't see this).
        # This lets the critic learn separate value scales per scenario instead
        # of averaging across maps with very different reward dynamics.
        sc_id = traj[0].get("scenario_id", 0)
        sc_onehot = torch.zeros(features.shape[0], num_scenarios, device=device)
        sc_onehot[:, sc_id] = 1.0

        with torch.no_grad():
            value_features = torch.cat([features, sc_onehot], dim=1)
            values_np = value_net(value_features).cpu().numpy()

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

        #Truncate AFTER GAE so boundary advantages have correct bootstraps
        #before: runcation happened before GAE, which forced the last step of a mid-episode window to bootstrap with V=0 (as if the episode ended).
        #Now GAE sees the full episode, and we slice the result.
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
        return {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "ratio": 1.0,
            "skipped": 0,
        }

    features = torch.cat(all_features, dim=0)#Concatenate all episodes into one big batch for PPO updates. We already normalized advantages per episode, so we can just concat and go.
    raw_m = torch.cat(all_raw_m, dim=0)#Concatenate all episodes into one big batch for PPO updates. We already normalized advantages per episode, so we can just concat and go.
    fire_acts = torch.cat(all_fire, dim=0)#Concatenate all episodes into one big batch for PPO updates. We already normalized advantages per episode, so we can just concat and go.
    mine_acts = torch.cat(all_mine, dim=0)#Concatenate all episodes into one big batch for PPO updates. We already normalized advantages per episode, so we can just concat and go.
    old_logp = torch.cat(all_old_logp, dim=0)
    adv_t = torch.cat(all_adv, dim=0)
    ret_t = torch.cat(all_ret, dim=0)
    scenario_ctx = torch.cat(all_scenario_ctx, dim=0)  # (N, num_scenarios)

    N = features.shape[0]

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    entropy_sum = 0.0
    ratio_sum = 0.0
    ratio_count = 0
    skipped_batches = 0
    #Kullback–Leibler divergence: a measure of how much the new policy diverged from the old one. Used for early stopping to prevent destructive updates.
    # KL target for early stopping. Relaxed from 0.02 because we already
    # have ratio-skip guards, grad clipping, and conservative clip_eps.
    target_kl = 0.05

    for _ in range(epochs): # multiple epochs over the pooled episodes
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

            new_logp_m, entropy_m = maneuver_policy.evaluate_action(mb_features, mb_raw_m)
            new_logp_c, entropy_c = combat_policy.evaluate_action(
                mb_features, mb_fire, mb_mine
            )

            new_logp_m = new_logp_m.squeeze(-1)
            new_logp_c = new_logp_c.squeeze(-1)
            entropy_m = entropy_m.squeeze(-1) if entropy_m.dim() > 1 else entropy_m
            entropy_c = entropy_c.squeeze(-1) if entropy_c.dim() > 1 else entropy_c

            new_logp = new_logp_m + new_logp_c
            # Combat entropy dominates; small maneuver term gives the
            # continuous head some exploration pressure without the
            # mean-collapse issues seen at full weight.
            entropy = entropy_c + 0.25 * entropy_m

            log_ratio = new_logp - mb_old_logp
            log_ratio = log_ratio.clamp(-4.0, 4.0)
            ratio = torch.exp(log_ratio)

            mean_ratio = float(ratio.mean().item())
            approx_kl = float((mb_old_logp - new_logp).mean().abs().item())

            # Real batch skip. If the batch is already too far off-policy,
            # don't apply a destructive update.
            if mean_ratio > 1.5 or mean_ratio < 0.67:
                skipped_batches += 1
                continue

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            mb_vf = torch.cat([mb_features, scenario_ctx[mb]], dim=1)
            v_pred = value_net(mb_vf)
            value_loss = nn.SmoothL1Loss()(v_pred, mb_ret)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            # critic step always runs
            opt_critic.zero_grad(set_to_none=True)
            critic_obj = value_coef * value_loss

            # only run policy backward if policy params are unfrozen
            policy_params = list(maneuver_policy.parameters()) + list(combat_policy.parameters())
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

            # Early stop this PPO pass if policy moved too much.
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





def anneal_log_std_(maneuver_policy, ep, total_episodes, warmup_episodes):
    """
    Actively shrink exploration after warmup.

    The old code only clamped an upper bound on log_std. If gradients did not
    move log_std downward, std could sit near the same value for hundreds of
    episodes. This helper moves log_std toward a scheduled target each PPO update.

    Right after warmup: target log_std ~= -1.10 (std ~ 0.33)
    End of training:   target log_std ~= -2.30 (std ~ 0.10)
    """
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
    maneuver_policy, combat_policy, value_net, opt_policy, opt_critic,
    episode, best_reward, total_episodes, path,
):
    #saves everything needed to resume training, including total_episodes for annealing
    torch.save({
        "maneuver_policy": maneuver_policy.state_dict(),
        "combat_policy": combat_policy.state_dict(),
        "value_net": value_net.state_dict(),
        "opt_policy": opt_policy.state_dict(),
        "opt_critic": opt_critic.state_dict(),
        "episode": episode,
        "best_reward": best_reward,
        "total_episodes": total_episodes,
    }, path)
    print(f"Saved RL checkpoint (ep {episode}) -> {path}")


def load_rl_checkpoint(
    maneuver_policy, combat_policy, value_net, opt_policy, opt_critic, path, device,
):
    #restores full training state, returns (episode, best_reward, total_episodes)
    ckpt = torch.load(path, map_location=device)
    maneuver_policy.load_state_dict(ckpt["maneuver_policy"])
    combat_policy.load_state_dict(ckpt["combat_policy"])
    # Value net input size may have changed (e.g. added scenario context).
    # If shapes don't match, skip loading and let the critic reinitialize.
    try:
        value_net.load_state_dict(ckpt["value_net"])
        critic_reloaded = True
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"  (value_net shape changed — reinitializing critic from scratch)")
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
    print(f"Resumed RL checkpoint from ep {ep}, best_reward={best:.2f}")
    return ep, best, total_ep



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--scenario", type=str, default="all")
    p.add_argument("--num_mfs", type=int, default=2,
                   help="Must match the warm-start model")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--critic_lr", type=float, default=3e-4,
                   help="Learning rate for value network (higher than policy LR)")
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
    p.add_argument("--max_steps_per_episode", type=int, default=512)
    p.add_argument("--warmup_episodes", type=int, default=0)
    p.add_argument("--resume", action="store_true")
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
    # ValueNet gets scenario context as extra one-hot input so it can learn
    # different value scales per scenario. Policy networks stay at num_inputs.
    value_net = ValueNet(num_inputs + NUM_SCENARIOS).to(device)

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
        # Restore trained log_std if available in the bundle
        _bundle = torch.load(maneuver_path, map_location="cpu")
        if "log_std" in _bundle:
            with torch.no_grad():
                maneuver_policy.log_std.copy_(torch.tensor(_bundle["log_std"]))
            print(f"Restored log_std: {maneuver_policy.log_std.data.tolist()}")
        print("Warm-started maneuver policy from expert.")
    else:
        print("No maneuver.pt found — training from scratch")
        feature_cols = [
            "dist", "ttc", "heading_err", "approach_speed",
            "ammo", "mines", "threat_density", "threat_angle",
        ]

    if os.path.exists(combat_path):
        mu_c, sd_c = warm_start_combat(combat_policy, combat_path)
        print("Warm-started combat policy from expert.")
    else:
        print("No combat.pt found — training from scratch")

    # Separate optimizers: critic needs a higher LR to keep up with
    # the changing policy across diverse scenarios.
    # Exclude log_std from the policy optimizer — it is controlled
    # entirely by anneal_log_std_() on a schedule. Having it in both
    # the optimizer (pulled up by entropy bonus) and the annealer
    # (pulled down) creates a tug-of-war that prevents clean decay.
    policy_params = (
        [p for n, p in maneuver_policy.named_parameters() if "log_std" not in n]
        + list(combat_policy.parameters())
    )
    opt_policy = optim.Adam(policy_params, lr=args.lr)
    opt_critic = optim.Adam(value_net.parameters(), lr=args.critic_lr)

    # Resume from RL checkpoint if requested
    start_ep = 1
    best_reward = -float("inf")
    total_episodes = args.episodes # save the original training horizon for annealing
    rl_ckpt_path = os.path.join(model_dir, "rl_checkpoint.pt")
    if args.resume and os.path.exists(rl_ckpt_path):
        start_ep, best_reward, saved_total = load_rl_checkpoint(
            maneuver_policy, combat_policy, value_net, opt_policy, opt_critic,
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
        "sniper_practice": sc.sniper_practice, # static targets, trains aiming
    }

    # Fixed mapping so checkpoint dimensions are stable regardless of --scenario flag.
    # The critic sees a one-hot of this as extra context; the policy does not.
    scenario_to_idx = {name: i for i, name in enumerate(scenario_map.keys())}
    num_scenarios = len(scenario_map)
    assert num_scenarios == NUM_SCENARIOS, (
        f"scenario_map has {num_scenarios} entries but NUM_SCENARIOS={NUM_SCENARIOS}"
    )

    if args.scenario.lower() == "all":
        # Training set: leave sniper out for now
        train_scenario_names = [
            "stock",
            "donut_ring",
            "vertical_wall_left",
            "spiral_arms",
            "crossing_lanes",
            "asteroid_rain",
            "four_corner",
        ]

        # Eval set: keep sniper so we can still monitor it
        eval_scenario_names = list(scenario_map.keys())
    else:
        train_scenario_names = [args.scenario]
        eval_scenario_names = [args.scenario]

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

    # critic warmup: freeze policy params so only value_net trains
    # this lets the critic learn what states are worth before PPO starts
    # pushing the actor around with bad advantage estimates
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
            # Tag every step with the scenario index so the critic
            # can receive scenario context during PPO updates.
            sc_idx = scenario_to_idx[scenario_name]
            for step in traj:
                step["scenario_id"] = sc_idx
            episode_pool.append(traj)
            # Count capped steps for pool-size gating (actual truncation
            # happens inside ppo_update_pooled AFTER GAE is computed on
            # the full episode, so the bootstrap at the slice boundary
            # is correct instead of being forced to zero).
            cap = args.max_steps_per_episode or len(traj)
            pool_steps += min(len(traj), cap)

        # PPO Update — only when pool is large enough
        stats = None
        if pool_steps >= MIN_POOL_STEPS:
            stats = ppo_update_pooled(
                maneuver_policy,
                combat_policy,
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

            # Actively cool exploration after warmup.
            anneal_log_std_(
                maneuver_policy,
                ep,
                total_episodes,
                args.warmup_episodes,
            )

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
        log_std_raw = maneuver_policy.log_std.detach().cpu().tolist()
        skip_str = f" skip={stats['skipped']}" if stats["skipped"] > 0 else ""
        warmup_str = " [WARMUP]" if policy_frozen else ""
        diag_str = ""
        if "n_episodes" in stats and stats["n_episodes"] > 0:
            diag_str = f" pool={stats['n_episodes']}ep adv_spread={stats['adv_std_spread']:.1f}x"
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
                maneuver_policy, combat_policy, value_net, opt_policy, opt_critic,
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
                f"  [DET-EVAL] avg={avg_det_reward:.1f} "
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
                print(f"  New best (deterministic avg): {best_reward:.2f}")

    if not args.eval:
        #Final save
        save_bundle(
            maneuver_policy, combat_policy,
            mu, sd, feature_cols, args.num_mfs,
            os.path.join(model_dir, "maneuver_rl.pt"),
        )
        save_rl_checkpoint(
            maneuver_policy, combat_policy, value_net, opt_policy, opt_critic,
            args.episodes, best_reward, total_episodes, rl_ckpt_path,
        )
        print("\nDone. Models saved to models/")
    total_time = time.perf_counter() - train_start
    minutes = total_time / 60
    print(f"\nTotal training time: {minutes:.1f} min ({total_time:.0f}s)")


if __name__ == "__main__":
    main()