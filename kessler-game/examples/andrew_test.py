# andrew_test.py
# Aggressive controller
# - Hunts asteroids with lead aim and high thrust
# - Drops mines when something is close
# - After dropping a mine, briefly steers away from it BUT still attacks
#   by blending away-from-mine and aim-to-target directions.

import math
from kesslergame.controller import KesslerController


# --------------------------
# Helpers
# --------------------------

def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def get_attr(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default

def get_heading_degrees(ship_state) -> float:
    if hasattr(ship_state, "heading"):
        try:
            return float(ship_state.heading)
        except Exception:
            pass
    if hasattr(ship_state, "angle"):
        try:
            return math.degrees(float(ship_state.angle))
        except Exception:
            pass
    return 0.0

def lead_time(rel_px, rel_py, rel_vx, rel_vy, proj_speed, eps=1e-6):
    a = (rel_vx * rel_vx + rel_vy * rel_vy) - proj_speed * proj_speed
    b = 2.0 * (rel_px * rel_vx + rel_py * rel_vy)
    c = rel_px * rel_px + rel_py * rel_py
    if abs(a) < eps:
        if abs(b) < eps:
            return None
        t = -c / b
        return t if t > 0 else None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sd = math.sqrt(disc)
    t1 = (-b - sd) / (2 * a)
    t2 = (-b + sd) / (2 * a)
    ts = [t for t in (t1, t2) if t > 0]
    return min(ts) if ts else None

def time_to_collision(px, py, vx, vy, radius, eps=1e-6):
    a = vx * vx + vy * vy
    if a < eps:
        return None
    b = 2 * (px * vx + py * vy)
    c = px * px + py * py - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sd = math.sqrt(disc)
    t1 = (-b - sd) / (2 * a)
    t2 = (-b + sd) / (2 * a)
    cand = [t for t in (t1, t2) if t >= 0]
    return min(cand) if cand else None

def angle_lerp_deg(a_deg: float, b_deg: float, w: float) -> float:
    """Slerp-like blend between angles in degrees, w in [0,1]."""
    a = math.radians(a_deg)
    b = math.radians(b_deg)
    d = math.atan2(math.sin(b - a), math.cos(b - a))
    out = a + clamp(w, 0.0, 1.0) * d
    return math.degrees(out)


# --------------------------
# Controller
# --------------------------

class AndrewTactic(KesslerController):
    name = "AndrewTacticBalancedAggro"

    FIRE_AIM_ERR_DEG   = 14.0
    FIRE_MAX_RANGE     = 900.0
    FIRE_MIN_RANGE     = 50.0

    APPROACH_DIST_HI   = 800.0
    APPROACH_DIST_MED  = 300.0
    APPROACH_DIST_LO   = 90.0

    DANGER_RADIUS      = 120.0
    COLLISION_TTC      = 0.6

    BASE_THRUST        = 70.0
    MAX_THRUST         = 160.0

    TURN_GAIN          = 6.0
    MAX_TURN_RATE      = 260.0

    SURROUND_RADIUS    = 230.0
    SURROUND_MIN_COUNT = 1
    MINE_COOLDOWN_STEPS = 18

    MINE_EVADE_STEPS     = 24
    MINE_SAFE_DIST       = 220.0
    MINE_EVADE_WEIGHT_MAX = 1.0

    DEFAULT_PROJECTILE_SPEED = 640.0

    def __init__(self):
        super().__init__()
        self._mine_evade_steps = 0
        self._mine_cooldown = 0
        self._mine_pos = None

    def actions(self, ship_state, game_state):
        # Ship
        sx, sy = get_attr(ship_state, ["position"], (0.0, 0.0))
        svx, svy = get_attr(ship_state, ["velocity", "vel"], (0.0, 0.0)) or (0.0, 0.0)
        heading = get_heading_degrees(ship_state)

        # World
        proj_speed = (
            get_attr(game_state, ["projectile_speed", "bullet_speed", "laser_speed"],
                     self.DEFAULT_PROJECTILE_SPEED)
            or self.DEFAULT_PROJECTILE_SPEED
        )
        asteroids = (
            get_attr(game_state, ["asteroids"], None)
            or get_attr(game_state, ["asteroid_states"], [])
            or []
        )

        if self._mine_cooldown > 0:
            self._mine_cooldown -= 1
        if self._mine_evade_steps > 0:
            self._mine_evade_steps -= 1

        # Roam if nothing to do
        if not asteroids:
            return self.BASE_THRUST, 40.0, False, False

        # Build targets: (dx, dy, rvx, rvy, dist, ahead_bias)
        targets = []
        for a in asteroids:
            pos = get_attr(a, ["position", "pos"], None)
            if not (isinstance(pos, (tuple, list)) and len(pos) >= 2):
                continue
            ax, ay = float(pos[0]), float(pos[1])
            vel = get_attr(a, ["velocity", "vel"], (0.0, 0.0)) or (0.0, 0.0)
            avx, avy = float(vel[0]), float(vel[1])
            dx, dy = ax - sx, ay - sy
            dist = math.hypot(dx, dy)
            rvx, rvy = avx - svx, avy - svy
            desired = math.degrees(math.atan2(dy, dx))
            err = wrap180(desired - heading)
            ahead_bias = math.cos(math.radians(abs(err)))
            targets.append((dx, dy, rvx, rvy, dist, ahead_bias))

        if not targets:
            return self.BASE_THRUST, 40.0, False, False

        # Imminent collision?
        imminent = any(
            (ttc := time_to_collision(dx, dy, rvx, rvy, self.DANGER_RADIUS)) is not None
            and ttc <= self.COLLISION_TTC
            for (dx, dy, rvx, rvy, _, _) in targets
        )

        # Choose target
        best = None
        best_score = -1e18
        for (dx, dy, rvx, rvy, dist, ahead_bias) in targets:
            score = -0.0020 * dist + 0.95 * ahead_bias
            if score > best_score:
                best_score = score
                best = (dx, dy, rvx, rvy, dist)
        dx, dy, rvx, rvy, dist = best

        # Lead aim
        t_hit = lead_time(dx, dy, rvx, rvy, proj_speed)
        if t_hit is not None and 0 < t_hit < 3.0:
            aim_x = dx + rvx * t_hit
            aim_y = dy + rvy * t_hit
        else:
            aim_x, aim_y = dx, dy
        aim_deg = math.degrees(math.atan2(aim_y, aim_x))

        # If recently dropped a mine, blend away from mine with aim
        desired_deg = aim_deg
        if self._mine_evade_steps > 0 and self._mine_pos is not None:
            mx, my = self._mine_pos
            away_x, away_y = (sx - mx), (sy - my)
            if abs(away_x) > 1e-6 or abs(away_y) > 1e-6:
                away_deg = math.degrees(math.atan2(away_y, away_x))
                d_from_mine = math.hypot(away_x, away_y)
                w = clamp((self.MINE_SAFE_DIST - d_from_mine) / self.MINE_SAFE_DIST, 0.0, self.MINE_EVADE_WEIGHT_MAX)
                desired_deg = angle_lerp_deg(aim_deg, away_deg, w)

        # Turn
        err = wrap180(desired_deg - heading)
        turn_rate = clamp(self.TURN_GAIN * err, -self.MAX_TURN_RATE, self.MAX_TURN_RATE)

        # Thrust & fire
        if imminent:
            side = -1.0 if err > 0 else 1.0
            turn_rate = clamp(turn_rate + side * 90.0, -self.MAX_TURN_RATE, self.MAX_TURN_RATE)
            thrust = self.MAX_THRUST
            aim_err = abs(wrap180(aim_deg - heading))
            fire = (self.FIRE_MIN_RANGE <= dist <= self.FIRE_MAX_RANGE) and (aim_err < self.FIRE_AIM_ERR_DEG / 2.0)
        else:
            if dist > self.APPROACH_DIST_HI:
                thrust = self.MAX_THRUST
            elif dist > self.APPROACH_DIST_MED:
                thrust = 0.85 * self.MAX_THRUST
            elif dist > self.APPROACH_DIST_LO:
                thrust = self.BASE_THRUST
            else:
                thrust = -0.30 * self.MAX_THRUST 
            aim_err = abs(wrap180(aim_deg - heading))
            fire = (self.FIRE_MIN_RANGE <= dist <= self.FIRE_MAX_RANGE) and (aim_err < self.FIRE_AIM_ERR_DEG)

        nearby = sum(1 for (_, _, _, _, d, _) in targets if d <= self.SURROUND_RADIUS)
        want_drop = (nearby >= self.SURROUND_MIN_COUNT) and (self._mine_cooldown == 0)

        drop_mine = False
        if want_drop:
            drop_mine = True
            self._mine_pos = (sx, sy)
            self._mine_evade_steps = self.MINE_EVADE_STEPS
            self._mine_cooldown = self.MINE_COOLDOWN_STEPS

        return float(clamp(thrust, -self.MAX_THRUST, self.MAX_THRUST)), float(turn_rate), bool(fire), bool(drop_mine)


__all__ = ["AndrewTactic"]
