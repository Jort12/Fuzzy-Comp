"""
rl_controller.py uses the stochastic maneuver policy.
Runs inside the game loop, collects (state, action, reward, log_prob)
trajectories, and leaves combat decisions to heuristics.
"""

import math
import torch
import numpy as np
from kesslergame.controller import KesslerController
from util import wrap180, toro_dx_dy, SHIP_RADIUS
from rl_policy import StochasticManeuverPolicy

FEATURE_COLS = [
    "dist", "ttc", "heading_err", "approach_speed",
    "ammo", "mines", "threat_density", "threat_angle",
]



# Reward coefficients
HIT_REWARD = 7.0
DEATH_PENALTY = -4.0
SURVIVAL_SOFT_GAP = 160.0
SURVIVAL_HARD_GAP = 80.0
CROWDING_RADIUS = 220.0

#Parameter
EXTENDED_RANGE = 100.0


#Extracts total hits and deaths from the game score for reward calculation.
def team_hits_and_deaths(game_or_score):
    hits = 0
    deaths = 0
    for t in getattr(game_or_score, "teams", []):
        hits += getattr(t, "asteroids_hit", 0)
        deaths += getattr(t, "deaths", 0)
    return hits, deaths



#calculate the reward for the current state, including sparse rewards for kills/deaths and dense shaping rewards for movement, aiming, and firing quality.
# prev_danger removed: was computed but never read in the reward body (dead code from removed danger-shaping term)
def compute_reward(ship_state, game_state, prev_hits, prev_deaths, prev_fire=False, locked_target=None):
    reward = 0.0
    dt = float(getattr(game_state, "delta_time", 1 / 30))#Time delta since last frame, used to make rewards per-second consistent regardless of frame rate.
    map_size = getattr(game_state, "map_size", (1000, 800))

    current_hits, current_deaths = team_hits_and_deaths(game_state)
    new_kills = max(0, current_hits - prev_hits) #New kills since last step, used for sparse kill rewards.
    new_deaths = max(0, current_deaths - prev_deaths)#New deaths since last step, used for sparse death penalties.

    #High Value Sparse Rewards
    reward += HIT_REWARD * new_kills
    reward += DEATH_PENALTY * new_deaths

    # Movement & Engagement (The "Search" part)
    speed = math.hypot(*getattr(ship_state, "velocity", (0.0, 0.0)))
    if speed < 20.0:
        reward -= 0.40 * dt 
    elif speed > 100.0:
        reward += 0.05 * dt

    asteroids = getattr(game_state, "asteroids", [])
    nearest_gap, crowding_count, danger_pressure, nearest_approach = survival_metrics(
        ship_state, asteroids, map_size
    )

    if crowding_count >= 3:
        reward -= 0.05 * min(crowding_count - 2, 4) * dt

    if nearest_gap < SURVIVAL_SOFT_GAP:
        reward -= 0.22 * (1.0 - nearest_gap / SURVIVAL_SOFT_GAP) * dt
    if nearest_gap < SURVIVAL_HARD_GAP:
        reward -= 0.35 * (1.0 - nearest_gap / SURVIVAL_HARD_GAP) * dt

    reward -= 0.015 * min(danger_pressure, 12.0) * dt

    if nearest_gap < SURVIVAL_SOFT_GAP and nearest_approach < 0.0:
        separation_speed = min(-nearest_approach / 120.0, 1.0)
        reward += 0.10 * separation_speed * dt

    #Dense Targeting Rewards
    #continuous aiming reward so the agent always has a gradient toward the target
    # uses locked_target if provided so reward and features target the same asteroid
    err = 180.0  # default to max error if no asteroids
    combat_focus = 0.35 if nearest_gap < SURVIVAL_SOFT_GAP or crowding_count >= 3 else 1.0
    if asteroids:
        sx, sy = ship_state.position
        if locked_target is not None:
            target = locked_target
        else:
            target, _ = find_priority_threat(asteroids, ship_state, map_size)
        ax, ay = target.position
        dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)
        
        desired = math.degrees(math.atan2(dy, dx))
        err = abs(wrap180(desired - ship_state.heading))
        
        # smooth reward: 0 at 180 degrees, ramps up to 0.30 at 0 degrees
        # squaring makes it pull harder as you get close to on-target
        aim_reward = 0.30 * (1.0 - err / 180.0) ** 2
        aim_reward *= combat_focus
        reward += aim_reward * dt

        dist = math.hypot(dx, dy)
        if dist < 220:
            reward += 0.08 * combat_focus * (1.0 - dist / 220.0) * dt

    # Firing: reward good shots, punish spraying into empty space
    fire_focus = 0.25 if nearest_gap < SURVIVAL_HARD_GAP or crowding_count >= 4 else combat_focus
    if prev_fire and err < 8:
        reward += 0.14 * fire_focus * dt   # very good shot
    elif prev_fire and err < 15:
        reward += 0.08 * fire_focus * dt   # decent shot
    elif prev_fire and err < 22:
        reward += 0.03 * fire_focus * dt   # still acceptable

    if prev_fire and err > 35:
        reward -= 0.12 * max(fire_focus, 0.5) * dt   # bad spray

    return reward, current_hits, current_deaths


# shared scoring formula, used by both find_priority_threat and compute_min_danger
def _threat_score(gap, ttc, closing, size):
    return (
        2.5 / max(gap, 20.0) +
        1.5 / max(ttc, 0.25) +
        0.015 * closing +
        0.20 * (5 - size)
    )

#Find the highest-scoring threat based on a combination of distance, time-to-collision, closing speed, and size. This is used for both targeting and reward calculation, so it uses the same scoring formula as compute_min_danger.
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

        radius = getattr(a, "radius", 0.0) #The radius of the asteroid, used to calculate the gap between the ship and the asteroid's surface, which is more relevant for collision risk than center distance.
        gap = center - radius - SHIP_RADIUS
        gap = max(gap, 1.0)

        avx, avy = getattr(a, "velocity", (0.0, 0.0))#The velocity of the asteroid, used to calculate the relative velocity and approach speed for reward shaping and threat scoring.
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


def survival_metrics(ship_state, asteroids, map_size):
    sx, sy = ship_state.position
    svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))

    nearest_gap = float("inf")
    nearest_approach = 0.0
    crowding_count = 0
    danger_pressure = 0.0

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

        if gap < nearest_gap:
            nearest_gap = gap
            nearest_approach = approach_speed

        if gap < CROWDING_RADIUS:
            crowding_count += 1

        if gap < SURVIVAL_SOFT_GAP:
            danger_pressure += closing / max(gap, 20.0)

    if not asteroids:
        return float("inf"), 0, 0.0, 0.0

    return nearest_gap, crowding_count, danger_pressure, nearest_approach


def calculate_context(ship_state, game_state, locked_target=None):
    sx, sy = ship_state.position
    heading = ship_state.heading
    asteroids = getattr(game_state, "asteroids", [])
    map_size = getattr(game_state, "map_size", (1000, 800))

    if not asteroids:
        return {k: 0.0 for k in FEATURE_COLS}

    if locked_target is not None:
        priority = locked_target
        ax, ay = priority.position
        dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)
        center = math.hypot(dx, dy)
        radius = getattr(priority, "radius", 0.0)
        dist = max(center - radius - SHIP_RADIUS, 1.0)
    else:
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

def dot(ax, ay, bx, by):
    return (ax*bx) + (ay*by)

def length(x, y):
    return math.sqrt(x*x + y*y)

def should_fire(ship_state, game_state) -> bool:
    #If ammo is unlimited shoot until you are hit by an asteroid then stop
    asteroid_arr = getattr(game_state, "asteroids", [])
    map_size = getattr(game_state, "map_size", (1000, 800))
    sx, sy = ship_state.position
    if ship_state.bullets_remaining == -1:
        for a in asteroid_arr:
            ax, ay = a.position
            total_r = a.radius + ship_state.radius
            dx_wrap, dy_wrap = toro_dx_dy(sx, sy, ax, ay, map_size)
            dist = math.hypot(dx_wrap, dy_wrap)
            if dist <= total_r:
                return False
        return True
    
    heading = ship_state.heading
    heading_rad = math.radians(heading)
    dx = math.cos(heading_rad)
    dy = math.sin(heading_rad)

    best = None  

    for a in asteroid_arr:
        ax, ay = a.position

        vx, vy = toro_dx_dy(sx, sy, ax, ay, map_size)

        along = dot(vx, vy, dx, dy)

        perp = vx * dy - vy * dx  

        if along <= 0:
            continue

        if abs(perp) <= a.radius:
            if best is None or along < best["along"]:
                best = {
                    "asteroid": a,
                    "along": along,
                    "perp": perp
                }

    if best is not None:
        return True  
    return False


class RLController(KesslerController):
    name = "RLController"

    # how many frames to hold a target before re-evaluating
    TARGET_LOCK_FRAMES = 10
    # The target-locking mechanism is important for both the reward function and the features, to ensure they are consistent and stable. By locking onto a specific target asteroid for several frames, we prevent the priority target from flickering between multiple asteroids in symmetric scenarios, which was causing the heading error feature to jump around and making it hard for the maneuver policy to learn a consistent turning behavior. The reward function also uses the locked target to calculate aiming rewards, so it benefits from the same stability.

    def __init__(
        self,
        maneuver_policy: StochasticManeuverPolicy,
        mu=None,
        sd=None,
        deterministic=False,
        scenario_id: int = 0,
        num_scenarios: int = 8,
    ):
        super().__init__()
        self.maneuver_policy = maneuver_policy
        self.mu = mu
        self.sd = sd
        self.deterministic = deterministic
        self.scenario_id = scenario_id
        self.num_scenarios = num_scenarios

        self.device = next(maneuver_policy.parameters()).device

        self.trajectory = []
        self.prev_asteroids_hit = 0
        self.prev_deaths = 0
        self._pending = None

        # target-sticking state
        self._locked_target = None
        self._locked_pos = None
        self._lock_frames_left = 0

    def _scenario_onehot(self):
        oh = torch.zeros(1, self.num_scenarios, device=self.device)
        if 0 <= self.scenario_id < self.num_scenarios:
            oh[0, self.scenario_id] = 1.0
        return oh

    def reset(self):
        self.trajectory = []
        self.prev_asteroids_hit = 0
        self.prev_deaths = 0
        self._pending = None
        self._locked_target = None
        self._locked_pos = None
        self._lock_frames_left = 0

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

    def _get_locked_target(self, ship_state, game_state):
        """
        Returns a stable target asteroid, holding the same one for
        TARGET_LOCK_FRAMES before re-evaluating. This prevents the
        priority target from flickering between equidistant asteroids
        in symmetric scenarios (donut_ring, four_corner), which was
        causing heading_err to jump wildly and preventing the maneuver
        head from learning a consistent turn direction.
        """
        asteroids = getattr(game_state, "asteroids", [])
        map_size = getattr(game_state, "map_size", (1000, 800))

        if not asteroids:
            self._locked_target = None
            self._locked_pos = None
            self._lock_frames_left = 0
            return None

        # check if current lock is still valid
        still_alive = False
        if self._locked_target is not None and self._lock_frames_left > 0:
            # look for an asteroid at the locked position (destroyed asteroids
            # get removed from the list, so we match by position)
            lx, ly = self._locked_pos
            for a in asteroids:
                ax, ay = a.position
                # tolerance of 15 handles fast-moving asteroids (~7 units/frame)
                # while still being tight enough to not match a different asteroid
                dx, dy = toro_dx_dy(lx, ly, ax, ay, map_size)
                if math.hypot(dx, dy) < 15.0:
                    # found it, update reference in case the object changed
                    self._locked_target = a
                    self._locked_pos = (ax, ay)
                    still_alive = True
                    break

        if still_alive:
            self._lock_frames_left -= 1
            return self._locked_target

        # lock expired or target destroyed — pick a new one
        target, _ = find_priority_threat(asteroids, ship_state, map_size)
        self._locked_target = target
        self._locked_pos = tuple(target.position)
        self._lock_frames_left = self.TARGET_LOCK_FRAMES
        return target

    def actions(self, ship_state, game_state):
        # get a stable target for this frame
        locked = self._get_locked_target(ship_state, game_state)

        ctx = calculate_context(ship_state, game_state, locked_target=locked)
        xb = self._normalize(ctx)
        sc_oh = self._scenario_onehot()

        # Finish previous transition reward using the current state
        if self._pending is not None:
            prev_fire = bool(self._pending["fire"])

            reward, self.prev_asteroids_hit, self.prev_deaths = compute_reward(
                ship_state,
                game_state,
                self.prev_asteroids_hit,
                self.prev_deaths,
                prev_fire=prev_fire,
                locked_target=locked,
            )
            self._pending["reward"] = reward
            self.trajectory.append(self._pending)
            self._pending = None

        # Current action
        if self.deterministic:
            with torch.no_grad():
                means, _ = self.maneuver_policy(xb, sc_oh)
                # Truly deterministic eval — no noise
                action = torch.tanh(means)
                thrust_norm = action[0, 0].item()
                turn_norm = action[0, 1].item()

                log_prob_m = torch.tensor(0.0, device=self.device)
                raw_sample_m = means

        else:
            action_m, log_prob_m, raw_sample_m = self.maneuver_policy.get_action(xb, sc_oh)
            thrust_norm = action_m[0, 0].item()
            turn_norm = action_m[0, 1].item()

        # FIX apply gains after sampling so log_prob stays clean.
        # Matches nf_infer.py GAIN=1.5 and TURN_GAIN=1.2.
        # This is part of the environment interface, not the policy distribution.
        thrust_scaled = max(-1.0, min(1.0, thrust_norm * 1.5))
        thrust = thrust_scaled * 150.0

        turn_scaled = max(-1.0, min(1.0, turn_norm * 1.2))
        turn_rate = turn_scaled * 180.0
        fire = should_fire(ship_state, game_state)
        mine = False

        self._pending = {
            "features": xb.detach(),
            "raw_sample_m": raw_sample_m.detach(),
            "log_prob": (log_prob_m).detach().squeeze(),
            "fire": fire,
            "reward": 0.0,
        }

        if hasattr(ship_state, "thrust_range"):
            lo, hi = ship_state.thrust_range
            thrust = max(lo, min(hi, thrust))
        if hasattr(ship_state, "turn_rate_range"):
            lo, hi = ship_state.turn_rate_range
            turn_rate = max(lo, min(hi, turn_rate))

        return float(thrust), float(turn_rate), fire, mine
