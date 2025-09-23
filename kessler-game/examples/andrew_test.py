# andrew_test.py
# A controller for KesslerGame that dodges asteroids, shoots with lead aim,
# and drops mines when surrounded.

import math
from kesslergame.controller import KesslerController


# --------------------------
# Small helpers
# --------------------------

def wrap180(deg: float) -> float:
    """Normalize degrees to (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def get_attr(obj, names, default=None):
    """Return first attribute that exists on obj from names; else default."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def get_heading_degrees(ship_state) -> float:
    """Gracefully read heading in degrees (fallback from radians if needed)."""
    if hasattr(ship_state, "heading"):
        try:
            return float(ship_state.heading)
        except Exception:
            pass
    if hasattr(ship_state, "angle"):  # radians
        try:
            return math.degrees(float(ship_state.angle))
        except Exception:
            pass
    return 0.0


def lead_time(rel_px, rel_py, rel_vx, rel_vy, proj_speed, eps=1e-6):
    """
    Solve |p + v t| = s t for earliest t > 0 ; None if projectile too slow.
    p = relative position, v = relative velocity, s = projectile speed.
    """
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
    """
    Solve |p + v t| = r for earliest t >= 0 ; None if miss.
    Useful for quick “imminent collision” checks with a safety bubble radius.
    """
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


# --------------------------
# Andrew tactic (controller)
# --------------------------

class AndrewTactic(KesslerController):
    

    name = "AndrewTactic"

    # --- Tunables ---
    FIRE_AIM_ERR_DEG   = 8.0
    FIRE_MAX_RANGE     = 700.0
    FIRE_MIN_RANGE     = 120.0
    APPROACH_DIST_HI   = 500.0
    APPROACH_DIST_LO   = 170.0

    DANGER_RADIUS      = 135.0      # “safety bubble” around ship for TTC
    COLLISION_TTC      = 1.0        # seconds

    BASE_THRUST        = 35.0
    MAX_THRUST         = 100.0
    TURN_GAIN          = 4.0        # proportional heading controller (deg -> deg/s)
    MAX_TURN_RATE      = 180.0      # deg/s clamp

    SURROUND_RADIUS    = 180.0
    SURROUND_MIN_COUNT = 3

    DEFAULT_PROJECTILE_SPEED = 620.0  # fallback if not in game_state

    def actions(self, ship_state, game_state):
        # Ship state
        sx, sy = get_attr(ship_state, ["position"], (0.0, 0.0))
        svx, svy = get_attr(ship_state, ["velocity", "vel"], (0.0, 0.0)) or (0.0, 0.0)
        heading = get_heading_degrees(ship_state)

        # Projectile speed (engine dependent)
        proj_speed = (
            get_attr(game_state, ["projectile_speed", "bullet_speed", "laser_speed"],
                     self.DEFAULT_PROJECTILE_SPEED)
            or self.DEFAULT_PROJECTILE_SPEED
        )

        # Asteroids list (schema tolerant)
        asteroids = (
            get_attr(game_state, ["asteroids"], None)
            or get_attr(game_state, ["asteroid_states"], [])
            or []
        )

        # idle spin if nothing to do
        if not asteroids:
            return 15.0, 25.0, False, False

        # Build normalized target list: (dx, dy, rvx, rvy, dist)
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
            targets.append((dx, dy, rvx, rvy, dist))

        if not targets:
            return 15.0, 25.0, False, False

        # Evasion: any asteroid with TTC into our DANGER_RADIUS soon?
        imminent = False
        for (dx, dy, rvx, rvy, dist) in targets:
            ttc = time_to_collision(dx, dy, rvx, rvy, self.DANGER_RADIUS)
            if ttc is not None and ttc <= self.COLLISION_TTC:
                imminent = True
                break

        # Offensive target selection: nearer + more “ahead” is better 
        best = None
        best_score = -1e18
        for (dx, dy, rvx, rvy, dist) in targets:
            desired = math.degrees(math.atan2(dy, dx))
            err = wrap180(desired - heading)
            ahead_bias = math.cos(math.radians(abs(err)))
            score = -0.0035 * dist + 0.7 * ahead_bias
            if score > best_score:
                best_score = score
                best = (dx, dy, rvx, rvy, dist)

        dx, dy, rvx, rvy, dist = best

        #  Predictive lead aim
        t_hit = lead_time(dx, dy, rvx, rvy, proj_speed)
        if t_hit is not None and t_hit < 3.0:
            aim_x = dx + rvx * t_hit
            aim_y = dy + rvy * t_hit
        else:
            aim_x, aim_y = dx, dy

        desired = math.degrees(math.atan2(aim_y, aim_x))
        err = wrap180(desired - heading)

        # Turn control
        turn_rate = clamp(self.TURN_GAIN * err, -self.MAX_TURN_RATE, self.MAX_TURN_RATE)

        # Thrust & fire logic
        if imminent:
            # Strafe by biasing turn hard perpendicular to aim direction,
            # and full thrust to escape.
            side = -1.0 if err > 0 else 1.0
            turn_rate = clamp(turn_rate + side * 80.0, -self.MAX_TURN_RATE, self.MAX_TURN_RATE)
            thrust = self.MAX_THRUST
            fire = False  #survival over shooting while dodging
        else:
            # Distance-based thrust profile
            if dist > self.APPROACH_DIST_HI:
                thrust = self.MAX_THRUST
            elif dist < self.APPROACH_DIST_LO:
                thrust = -0.55 * self.MAX_THRUST  # back off if too close
            else:
                thrust = self.BASE_THRUST

            # Fire only if aligned and at a sane range
            aim_err = abs(err)
            fire = (self.FIRE_MIN_RANGE <= dist <= self.FIRE_MAX_RANGE) and (aim_err < self.FIRE_AIM_ERR_DEG)

        # Drop a mine when crowded
        nearby = sum(1 for (_, _, _, _, d) in targets if d <= self.SURROUND_RADIUS)
        drop_mine = nearby >= self.SURROUND_MIN_COUNT

        return float(clamp(thrust, -self.MAX_THRUST, self.MAX_THRUST)), float(turn_rate), bool(fire), bool(drop_mine)


__all__ = ["AndrewTactic"]
