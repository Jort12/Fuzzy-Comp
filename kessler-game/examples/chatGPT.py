import math
from kesslergame.controller import KesslerController

# --------------------------
# Helpers
# --------------------------

def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def get_attr(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def get_heading_degrees(ship_state):
    # Supports either .heading (deg) or .angle (rad)
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


def hypot2(dx, dy):
    return dx * dx + dy * dy


def time_to_collision(px, py, vx, vy, radius, eps=1e-6):
    """
    Approximate time to collision with the origin (ship at 0,0) assuming
    circular safety radius. Solves |p + v t| = r for smallest t >= 0.
    Returns None if diverging or no real solution.
    """
    a = vx * vx + vy * vy
    if a < eps:
        return None
    b = 2 * (px * vx + py * vy)
    c = px * px + py * py - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    # choose the earliest non-negative time
    candidates = [t for t in (t1, t2) if t >= 0]
    return min(candidates) if candidates else None


def lead_time(rel_px, rel_py, rel_vx, rel_vy, proj_speed, eps=1e-6):
    """Compute intercept time t for projectile speed s, solving
    |p + v t| = s t. Returns None if no solution (projectile too slow)."""
    # (vx^2 + vy^2 - s^2) t^2 + 2(p·v) t + (px^2 + py^2) = 0
    a = (rel_vx * rel_vx + rel_vy * rel_vy) - proj_speed * proj_speed
    b = 2.0 * (rel_px * rel_vx + rel_py * rel_vy)
    c = rel_px * rel_px + rel_py * rel_py

    if abs(a) < eps:
        # Linear case: treat as closest approach along v
        if abs(b) < eps:
            return None
        t = -c / b
        return t if t > 0 else None

    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    # pick earliest positive time
    ts = [t for t in (t1, t2) if t > 0]
    return min(ts) if ts else None


# --------------------------
# Advanced Tactic
# --------------------------

class AdvancedTactic(KesslerController):
    """
    A significantly upgraded controller with:
      • Threat prioritization (closes-on-us first)
      • Predictive aiming (lead shots)
      • Evasive maneuvers for near-term collisions
      • Smooth turn & thrust control (approach, brake, cruise)
    Safe against missing fields and works with multiple game-state schemas.
    """

    name = "AdvancedTactic"

    # Tunables (feel free to tweak)
    FIRE_AIM_ERR_DEG = 9.0
    FIRE_MAX_RANGE = 800.0
    APPROACH_DIST_HI = 450.0
    APPROACH_DIST_LO = 160.0
    DANGER_RADIUS = 135.0
    COLLISION_TTC = 1.1  # seconds
    BASE_THRUST = 40.0
    MAX_THRUST = 100.0
    TURN_GAIN = 4.0
    MAX_TURN_RATE = 180.0

    def actions(self, ship_state, game_state):
        # ------------------
        # Read ship state
        # ------------------
        sx, sy = get_attr(ship_state, ["position"], (0.0, 0.0))
        svx, svy = get_attr(ship_state, ["velocity", "vel"], (0.0, 0.0)) or (0.0, 0.0)
        heading = get_heading_degrees(ship_state)

        # Projectile speed (fallback)
        proj_speed = (
            get_attr(game_state, ["projectile_speed", "bullet_speed", "laser_speed"], 620.0)
            or 620.0
        )

        # ------------------
        # Collect asteroids
        # ------------------
        asteroids = (
            get_attr(game_state, ["asteroids"], None)
            or get_attr(game_state, ["asteroid_states"], [])
            or []
        )

        # Normalize asteroid list to (px,py,vx,vy,dist,closing)
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

            # Closing speed along line of sight (positive if closing)
            rel_vx, rel_vy = avx - svx, avy - svy
            los = (dx * rel_vx + dy * rel_vy) / (dist + 1e-6)
            closing = -los  # positive means moving toward us

            targets.append((dx, dy, rel_vx, rel_vy, dist, closing))

        if not targets:
            # No targets visible: slow patrol spin
            return 15.0, 25.0, False, False

        # ------------------
        # Threat prioritization
        # ------------------
        # Score: nearer + more closing gets higher priority
        best = None
        best_score = -1e18
        for (dx, dy, rvx, rvy, dist, closing) in targets:
            # Favor smaller distance and larger closing
            # Also a slight bias if target is in front (cosine of angle to heading)
            desired = math.degrees(math.atan2(dy, dx))
            err = wrap180(desired - heading)
            front_bias = math.cos(math.radians(abs(err)))  # ~1 if ahead, ~0 if sideways

            score = 2.5 * closing - 0.003 * dist + 0.5 * front_bias
            if score > best_score:
                best_score = score
                best = (dx, dy, rvx, rvy, dist)

        dx, dy, rvx, rvy, dist = best

        # ------------------
        # Predictive aim (lead point)
        # ------------------
        # Relative position p = (dx,dy), relative velocity v = (rvx,rvy)
        t_hit = lead_time(dx, dy, rvx, rvy, proj_speed)
        if t_hit is not None and t_hit < 3.0:
            aim_x = dx + rvx * t_hit
            aim_y = dy + rvy * t_hit
        else:
            aim_x, aim_y = dx, dy  # fallback to pure LOS aim

        desired = math.degrees(math.atan2(aim_y, aim_x))
        err = wrap180(desired - heading)

        # ------------------
        # Evasion: if something is about to hit us, strafe
        # ------------------
        imminent = False
        for (tx, ty, rvx, rvy, d, _) in targets:
            ttc = time_to_collision(tx, ty, rvx, rvy, self.DANGER_RADIUS)
            if ttc is not None and ttc <= self.COLLISION_TTC:
                imminent = True
                break

        # ------------------
        # Turn control
        # ------------------
        turn_rate = clamp(self.TURN_GAIN * err, -self.MAX_TURN_RATE, self.MAX_TURN_RATE)

        # ------------------
        # Thrust control
        # ------------------
        if imminent:
            # Strafe roughly perpendicular to aim to dodge, with strong thrust
            # Sign: move to the side that reduces |err|
            side = -1.0 if err > 0 else 1.0
            turn_rate = clamp(turn_rate + side * 75.0, -self.MAX_TURN_RATE, self.MAX_TURN_RATE)
            thrust = self.MAX_THRUST
            fire = False
        else:
            if dist > self.APPROACH_DIST_HI:
                thrust = self.MAX_THRUST  # close distance quickly
            elif dist < self.APPROACH_DIST_LO:
                # Brake / back off slightly when too close
                thrust = -0.5 * self.MAX_THRUST
            else:
                thrust = self.BASE_THRUST  # cruise

            # Fire when aligned and in range
            aim_err = abs(err)
            fire = (aim_err < self.FIRE_AIM_ERR_DEG) and (dist < self.FIRE_MAX_RANGE)

        return float(clamp(thrust, -self.MAX_THRUST, self.MAX_THRUST)), float(turn_rate), bool(fire), False
