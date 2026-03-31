"""
rl_controller.py uses the stochastic RL policy.
Runs inside the game loop, collects (state, action, reward, log_prob) trajectories.
After each episode, the training script pulls the trajectory and does PPO updates.
"""

import math
import torch
import numpy as np
from kesslergame.controller import KesslerController
from util import wrap180, toro_dx_dy, SHIP_RADIUS
from rl_policy import (StochasticManeuverPolicy,StochasticCombatPolicy,)

FEATURE_COLS = [
    "dist", "ttc", "heading_err", "approach_speed",
    "ammo", "mines", "threat_density", "threat_angle",
]

# Reward coefficients (tunable)
HIT_REWARD = 7.0
DEATH_PENALTY = -4.0


def team_hits_and_deaths(game_or_score):
    hits = 0
    deaths = 0
    for t in getattr(game_or_score, "teams", []):
        hits += getattr(t, "asteroids_hit", 0)
        deaths += getattr(t, "deaths", 0)
    return hits, deaths

def compute_reward(ship_state, game_state, prev_hits, prev_deaths, prev_danger, prev_fire=False):
    reward = 0.0
    dt = float(getattr(game_state, "delta_time", 1 / 30))
    map_size = getattr(game_state, "map_size", (1000, 800))

    current_hits, current_deaths = team_hits_and_deaths(game_state)
    new_kills = max(0, current_hits - prev_hits)
    new_deaths = max(0, current_deaths - prev_deaths)

    # 1. High-Value Sparse Rewards
    reward += 8.0 * new_kills
    reward += -6.0 * new_deaths

    # 2. Movement & Engagement (The "Search" part)
    speed = math.hypot(*getattr(ship_state, "velocity", (0.0, 0.0)))
    if speed < 20.0:
        reward -= 0.40 * dt 
    elif speed > 100.0:
        reward += 0.05 * dt

    # 3. Dense Targeting Rewards (The "Destroy" part)
    asteroids = getattr(game_state, "asteroids", [])
    err = 180.0  # default: not aimed at anything
    if asteroids:
        sx, sy = ship_state.position
        closest = min(asteroids, key=lambda a: math.hypot(*toro_dx_dy(sx, sy, a.position[0], a.position[1], map_size)))
        ax, ay = closest.position
        dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)
        
        desired = math.degrees(math.atan2(dy, dx))
        err = abs(wrap180(desired - ship_state.heading))
        
        if err < 4:
            reward += 0.30 * dt
        elif err < 10:
            reward += 0.10 * dt

        dist = math.hypot(dx, dy)
        if dist < 220:
            reward += 0.08 * (1.0 - dist / 220.0) * dt

    # 4. Firing Encouragement
    if prev_fire and err < 20:
        reward += 0.02 * dt

    return reward, current_hits, current_deaths, compute_min_danger(ship_state, game_state)


# shared scoring formula, used by both find_priority_threat and compute_min_danger
def _threat_score(gap, ttc, closing, size):
    return (
        2.5 / max(gap, 20.0) +
        1.5 / max(ttc, 0.25) +
        0.015 * closing +
        0.20 * (5 - size)
    )


def find_priority_threat(asteroids, ship_state, map_size):
    sx, sy = ship_state.position
    svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))

    best = None
    best_gap = float("inf")
    best_score = -float("inf")

    for a in asteroids:
        ax, ay = a.position
        dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)
        center = math.hypot(dx, dy)

        radius = getattr(a, "radius", 0.0)
        gap = center - radius - SHIP_RADIUS
        gap = max(gap, 1.0)

        avx, avy = getattr(a, "velocity", (0.0, 0.0))
        rel_vx, rel_vy = avx - svx, avy - svy

        approach_speed = (rel_vx * dx + rel_vy * dy) / max(center, 1.0)
        closing = max(approach_speed, 0.0)

        ttc = gap / max(closing, 1e-6)
        ttc = min(ttc, 100.0)

        size = getattr(a, "size", 2)

        score = _threat_score(gap, ttc, closing, size)

        if score > best_score:
            best_score = score
            best = a
            best_gap = gap

    return best, max(best_gap, 1.0)


def compute_min_danger(ship_state, game_state):
    #returns the highest threat score across all asteroids
    asteroids = getattr(game_state, "asteroids", [])
    map_size = getattr(game_state, "map_size", (1000, 800))

    if not asteroids:
        return 0.0

    sx, sy = ship_state.position
    svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))

    best_score = 0.0

    for a in asteroids:
        ax, ay = a.position
        dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)
        center = math.hypot(dx, dy)

        radius = getattr(a, "radius", 0.0)
        gap = max(center - radius - SHIP_RADIUS, 1.0)

        avx, avy = getattr(a, "velocity", (0.0, 0.0))
        rel_vx, rel_vy = avx - svx, avy - svy

        approach_speed = (rel_vx * dx + rel_vy * dy) / max(center, 1.0)
        closing = max(approach_speed, 0.0)

        ttc = min(gap / max(closing, 1e-6), 100.0)
        size = getattr(a, "size", 2)

        score = _threat_score(gap, ttc, closing, size)
        best_score = max(best_score, score)

    return best_score

def calculate_context(ship_state, game_state):
    sx, sy = ship_state.position
    heading = ship_state.heading
    asteroids = getattr(game_state, "asteroids", [])
    map_size = getattr(game_state, "map_size", (1000, 800))

    if not asteroids:
        return {k: 0.0 for k in FEATURE_COLS}

    priority, dist = find_priority_threat(asteroids, ship_state, map_size)
    ax, ay = priority.position
    dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)

    avx, avy = getattr(priority, "velocity", (0.0, 0.0))
    svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))
    rel_vx, rel_vy = avx - svx, avy - svy

    raw_dist = math.hypot(dx, dy)
    approach_speed = (rel_vx * dx + rel_vy * dy) / max(raw_dist, 1.0)

    closing = max(approach_speed, 0.0)
    ttc = min(dist / max(closing, 1e-6), 100.0)

    heading_err = wrap180(math.degrees(math.atan2(dy, dx)) - heading)
    density = len(asteroids) / 10.0

    return {
        "dist": dist,
        "ttc": ttc,
        "heading_err": heading_err,
        "approach_speed": approach_speed,
        "ammo": getattr(ship_state, "ammo", 0),
        "mines": getattr(ship_state, "mines", 0),
        "threat_density": density,
        "threat_angle": math.degrees(math.atan2(dy, dx)),
    }

class RLController(KesslerController):
    name = "RLController"

    def __init__(
        self,
        maneuver_policy: StochasticManeuverPolicy,
        combat_policy: StochasticCombatPolicy,
        mu=None,
        sd=None,
        deterministic=False,):
        self.prev_danger = 0.0
        super().__init__()
        self.maneuver_policy = maneuver_policy
        self.combat_policy = combat_policy
        self.mu = mu
        self.sd = sd
        self.deterministic = deterministic

        self.device = next(maneuver_policy.parameters()).device

        self.trajectory = []
        self.prev_asteroids_hit = 0
        self.prev_deaths = 0
        self._pending = None

    def reset(self):
        self.trajectory = []
        self.prev_asteroids_hit = 0
        self.prev_deaths = 0
        self._pending = None
        self.prev_danger = 0.0

    def finalize_episode(self, score=None):
        if self._pending is None:
            return

        terminal_reward = 0.0
        if score is not None:
            final_hits, final_deaths = team_hits_and_deaths(score)

            terminal_reward += HIT_REWARD * max(0, final_hits - self.prev_asteroids_hit)
            terminal_reward += DEATH_PENALTY * max(0, final_deaths - self.prev_deaths)

            if final_hits == 0 and final_deaths == 0:
                terminal_reward -= 6.0

        self._pending["reward"] = terminal_reward
        self.trajectory.append(self._pending)
        self._pending = None

    def _normalize(self, ctx):
        x = np.array([ctx[k] for k in FEATURE_COLS], dtype=np.float32)
        if self.mu is not None and self.sd is not None:
            sd = self.sd.copy()
            sd[sd < 1e-6] = 1.0
            x = (x - self.mu) / sd
        return torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

    def actions(self, ship_state, game_state):
        ctx = calculate_context(ship_state, game_state)
        xb = self._normalize(ctx)

        # Finish previous transition reward using the current state
        if self._pending is not None:
            prev_fire = bool(self._pending["fire_action"].item() > 0.5)

            reward, self.prev_asteroids_hit, self.prev_deaths, self.prev_danger = compute_reward(
                ship_state,
                game_state,
                self.prev_asteroids_hit,
                self.prev_deaths,
                self.prev_danger,
                prev_fire=prev_fire,
            )
            self._pending["reward"] = reward
            self.trajectory.append(self._pending)
            self._pending = None

        # Current action
        if self.deterministic:
            with torch.no_grad():
                means, _ = self.maneuver_policy(xb)
                # Truly deterministic eval — no noise
                action = torch.tanh(means)
                thrust_norm = action[0, 0].item()
                turn_norm = action[0, 1].item()

                log_prob_m = torch.tensor(0.0, device=self.device)
                raw_sample_m = means

        else:
            action_m, log_prob_m, raw_sample_m = self.maneuver_policy.get_action(xb)
            thrust_norm = action_m[0, 0].item()
            turn_norm = action_m[0, 1].item()

        # FIX: apply gains AFTER sampling so log_prob stays clean.
        # Matches nf_infer.py GAIN=1.5 and TURN_GAIN=1.2.
        # This is part of the environment interface, not the policy distribution.
        thrust_scaled = max(-1.0, min(1.0, thrust_norm * 1.5))
        thrust = thrust_scaled * 150.0

        turn_scaled = max(-1.0, min(1.0, turn_norm * 1.2))
        turn_rate = turn_scaled * 180.0
        
        if self.deterministic:
            fire_logit, mine_logit = self.combat_policy(xb)
            fire = bool(torch.sigmoid(fire_logit).item() > 0.4)
            mine = bool(torch.sigmoid(mine_logit).item() > 0.4)
            log_prob_c = torch.tensor(0.0, device=self.device)
            fire_a = torch.tensor(1.0 if fire else 0.0, device=self.device)
            mine_a = torch.tensor(1.0 if mine else 0.0, device=self.device)
        else:
            fire_a, mine_a, log_prob_c = self.combat_policy.get_action(xb)
            fire = bool(fire_a.item())
            mine = bool(mine_a.item())

        # Recompute combat log_prob to match the (possibly overridden) actions,
        # so the stored old_logp is consistent with the stored fire_a / mine_a.
        if not self.deterministic:
            log_prob_c, _ = self.combat_policy.evaluate_action(
                xb, fire_a, mine_a.detach()
            )

        self._pending = {
            "features": xb.detach(),
            "raw_sample_m": raw_sample_m.detach(),
            "fire_action": fire_a.detach(),
            "mine_action": mine_a.detach(),
            "log_prob": (log_prob_m + log_prob_c).detach().squeeze(),
            "reward": 0.0,
        }

        if hasattr(ship_state, "thrust_range"):
            lo, hi = ship_state.thrust_range
            thrust = max(lo, min(hi, thrust))
        if hasattr(ship_state, "turn_rate_range"):
            lo, hi = ship_state.turn_rate_range
            turn_rate = max(lo, min(hi, turn_rate))

        return float(thrust), float(turn_rate), fire, mine