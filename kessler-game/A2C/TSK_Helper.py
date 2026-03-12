from util import *
from TSK import *
import math
import numpy as np
import os 
import json


DEFAULT_MAP_W = 1000.0
DEFAULT_MAP_H = 800.0
DEFAULT_SHIP_RADIUS = 20.0
def get_map_size(game_state):
    map_size = getattr(game_state, "map_size", None)

    if isinstance(map_size, (tuple, list)) and len(map_size) >= 2:
        return float(map_size[0]), float(map_size[1])

    if hasattr(game_state, "width") and hasattr(game_state, "height"):
        return float(game_state.width), float(game_state.height)

    return DEFAULT_MAP_W, DEFAULT_MAP_H


def wrap_delta(d, size):
    if d > size / 2.0:
        d -= size
    elif d < -size / 2.0:
        d += size
    return d


def wrapped_vector(from_pos, to_pos, map_w, map_h):
    dx = wrap_delta(to_pos[0] - from_pos[0], map_w)
    dy = wrap_delta(to_pos[1] - from_pos[1], map_h)
    return dx, dy


def wrapped_distance(from_pos, to_pos, map_w, map_h):
    dx, dy = wrapped_vector(from_pos, to_pos, map_w, map_h)
    return math.hypot(dx, dy)


def asteroid_radius(asteroid):
    if hasattr(asteroid, "radius"):
        return float(getattr(asteroid, "radius", 0.0))

    size = getattr(asteroid, "size", 2)
    if size == 1:
        return 8.0
    if size == 2:
        return 16.0
    if size == 3:
        return 24.0
    return 32.0


def surface_gap(ship_pos, asteroid, map_w, map_h, ship_radius=DEFAULT_SHIP_RADIUS):
    center_dist = wrapped_distance(ship_pos, asteroid.position, map_w, map_h)
    return center_dist - ship_radius - asteroid_radius(asteroid)


def wrapped_angle_deg(from_pos, to_pos, map_w, map_h):
    dx, dy = wrapped_vector(from_pos, to_pos, map_w, map_h)
    return math.degrees(math.atan2(dy, dx))


def mu_dist(d):
    return {
        "very_close": triag(d, 0, 80, 160),
        "close": triag(d, 120, 200, 300),
        "medium": triag(d, 250, 400, 600),
        "far": triag(d, 500, 700, 1000)
    }

def mu_approach(v):
    return {
        "away":                   triag(v, -150, -50,   0),
        "stable":                 triag(v,  -60,   0,   60),
        "approaching":            triag(v,   30, 100,  200),
        "fast_approaching":       triag(v,  150, 250,  300),
        "very_fast_approaching":  triag(v,  250, 350, 400),
    }
    
    
def mu_ttc(t):
    # Missing / invalid TTC -> no confidence (or choose far_future if you prefer)
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return {
            "imminent": 0.0,
            "soon": 0.0,
            "later": 0.0,
            "far_future": 0.0
        }

    # Not closing or undefined TTC -> treat as far future
    if isinstance(t, float) and math.isinf(t):
        return {
            "imminent": 0.0,
            "soon": 0.0,
            "later": 0.0,
            "far_future": 1.0
        }

    return {
        "imminent": triag(t, 0, 1, 3),
        "soon": triag(t, 2, 4, 6),
        "later": triag(t, 5, 8, 12),
        "far_future": triag(t, 10, 15, 25)
    }

def mu_heading_err(e):
    return {
        "sharp_left":   triag(e, -180, -90,  -40),
        "left":         triag(e,  -60, -35,   -8),
        "slight_left":  triag(e,  -20,  -8,    0),
        "straight":     triag(e,  -10,   0,   10),
        "slight_right": triag(e,    0,   8,   20),
        "right":        triag(e,    8,  35,   60),
        "sharp_right":  triag(e,   40,  90,  180),
    }
    
def mu_ammo(a):
    return {
        "none": triag(a, 0, 0, 1),
        "very_low": triag(a, 0, 1, 2),
        "low": triag(a, 1, 2, 4),
        "medium": triag(a, 3, 5, 7),
        "high": triag(a, 6, 8, 10),
        "full": triag(a, 9, 10, 10)
    }
    
def mu_mine(m):
    return {
        "none": triag(m, 0, 0, 1),
        "low": triag(m, 0, 1, 2),
        "medium": triag(m, 1, 2, 3),
        "high": triag(m, 2, 3, 4),
    }
    
    
def mu_threat_density(density):
    return {
        "clear": triag(density, 0, 0, 1),
        "low": triag(density, 0, 1, 3),
        "moderate": triag(density, 2, 4, 6),
        "dense": triag(density, 5, 7, 10)
    }
    
def mu_threat_angle(angle):
    return {
        "left_side": triag(angle, -180, -90, 0),
        "ahead": triag(angle, -45, 0, 45),
        "right_side": triag(angle, 0, 90, 180),
    }

def norm(x, lo, hi):
    if hi == lo:
        return 0.0
    x = max(lo, min(hi, x))
    return (x - lo) / (hi - lo)

def heading_norm(x):
    return max(-1.0, min(1.0, x / 180.0))
def dist_norm(x):
    return norm(x, 0.0, 1000.0)

def reward(ship_state, game_state):
    dt = float(getattr(game_state, "delta_time", 1.0))
    ctx = context(ship_state, game_state)
    if ctx is None:
        return 0.0

    r = 0.0

    dist = float(ctx["dist"])
    ttc = ctx["ttc"]
    heading_err = abs(float(ctx["heading_err"]))
    density = float(ctx["threat_density"])
    approach = float(ctx.get("approach_speed", 0.0))

    # small alive reward
    r += 0.02 * dt

    # reward breathing room
    if dist > 120:
        r += 0.03 * dt
    if dist > 220:
        r += 0.03 * dt

    # reward lower crowding
    if density <= 2:
        r += 0.02 * dt

    # danger shaping
    if dist < 80:
        r -= 0.08 * dt
    if dist < 40:
        r -= 0.20 * dt

    # TTC danger shaping
    if math.isfinite(ttc):
        if ttc < 2.0:
            r -= 0.15 * dt
        elif ttc < 4.0:
            r -= 0.07 * dt

    # if threat is close, reward being roughly pointed toward solution / target
    if dist < 180 and heading_err < 15:
        r += 0.03 * dt

    # penalize freezing when danger is real
    speed = math.hypot(ship_state.velocity[0], ship_state.velocity[1])
    if dist < 120 and speed < 20:
        r -= 0.05 * dt

    # wrapped collision penalty
    if not getattr(ship_state, "is_respawning", False):
        map_w, map_h = get_map_size(game_state)
        ship_r = float(getattr(ship_state, "radius", DEFAULT_SHIP_RADIUS))

        for ast in getattr(game_state, "asteroids", []):
            gap = surface_gap(ship_state.position, ast, map_w, map_h, ship_r)
            if gap <= 0.0:
                r -= 25.0
                break

    return float(r)

# Parsing helper for both actor and critic consequents, supporting constant and linear types with optional input features
def parse_consequent_params(spec):
    spec_type = spec.get("type", "linear")

    if spec_type == "constant":
        return {
            "bias": float(spec.get("value", 0.0)),
            "weights": {}
        }

    if spec_type == "linear":
        return {
            "bias": float(spec.get("bias", 0.0)),
            "weights": {k: float(v) for k, v in spec.get("weights", {}).items()}
        }

    raise ValueError(f"Unknown consequent type: {spec_type}")


def context(ship_state, game_state):
    asteroids = getattr(game_state, "asteroids", [])
    sx, sy = ship_state.position
    svx, svy = ship_state.velocity
    heading = ship_state.heading

    map_w, map_h = get_map_size(game_state)
    ship_r = float(getattr(ship_state, "radius", DEFAULT_SHIP_RADIUS))

    if not asteroids:
        return {
            "dist": 1000.0,
            "approach_speed": 0.0,
            "ttc": float("inf"),
            "heading_err": 0.0,
            "ammo": getattr(ship_state, "bullets_remaining", getattr(ship_state, "ammo", 0)),
            "mines": getattr(ship_state, "mines_remaining", getattr(ship_state, "mines", 0)),
            "threat_density": 0.0,
            "threat_angle": 0.0,
            "escape_x": 0.0,
            "dist_norm": float(dist_norm(1000.0)),
            "heading_norm": 0.0,
        }

    best_target = None
    best_score = -float("inf")
    best_heading_err = 0.0
    best_heading_abs = 0.0
    best_closing = 0.0
    best_ttc = float("inf")
    best_dist = float("inf")

    closest_gap = float("inf")
    threat_angle = 0.0

    for ast in asteroids:
        dx, dy = wrapped_vector((sx, sy), ast.position, map_w, map_h)
        center_dist = math.hypot(dx, dy)
        gap = center_dist - ship_r - asteroid_radius(ast)

        avx, avy = getattr(ast, "velocity", (0.0, 0.0))

        # closest threat for danger features
        if gap < closest_gap:
            closest_gap = gap
            angle_to_ast = math.degrees(math.atan2(dy, dx))
            threat_angle = wrap180(angle_to_ast - heading)

        # ignore very far targets for engagement scoring
        if center_dist > 800:
            continue

        ux, uy = (dx / max(center_dist, 1e-6), dy / max(center_dist, 1e-6))
        rel_los = -((avx - svx) * ux + (avy - svy) * uy)  # positive when closing

        # unwrap asteroid into local coordinates before intercept
        local_target_pos = (sx + dx, sy + dy)
        intercept = intercept_point(
            (sx, sy),
            (svx, svy),
            local_target_pos,
            (avx, avy)
        )
        angle_to_intercept = angle_between((sx, sy), intercept)

        signed_err = wrap180(angle_to_intercept - heading)
        abs_err = abs(signed_err)

        cluster_score = 0.0
        for other in asteroids:
            if other is ast:
                continue
            od = wrapped_distance(ast.position, other.position, map_w, map_h)
            if od < 150:
                cluster_score += (150 - od) / 150.0

        score = (1000.0 / max(gap + 1.0, 1.0)) + cluster_score * 100.0 - abs_err * 2.0

        if score > best_score:
            best_score = score
            best_target = ast
            best_dist = max(gap, 0.0)
            best_heading_err = signed_err
            best_heading_abs = abs_err
            best_closing = max(0.0, rel_los)
            best_ttc = (gap / max(best_closing, 1e-3)) if best_closing > 0.0 else float("inf")

    if best_target is None:
        # fallback: nearest by wrapped gap
        nearest = min(
            asteroids,
            key=lambda a: surface_gap((sx, sy), a, map_w, map_h, ship_r)
        )

        ndx, ndy = wrapped_vector((sx, sy), nearest.position, map_w, map_h)
        ngap = surface_gap((sx, sy), nearest, map_w, map_h, ship_r)
        navx, navy = getattr(nearest, "velocity", (0.0, 0.0))

        intercept = intercept_point(
            (sx, sy),
            (svx, svy),
            (sx + ndx, sy + ndy),
            (navx, navy)
        )
        angle_to_intercept = angle_between((sx, sy), intercept)
        signed_err = wrap180(angle_to_intercept - heading)

        best_dist = max(ngap, 0.0)
        best_heading_err = signed_err
        best_heading_abs = abs(signed_err)
        best_closing = 0.0
        best_ttc = float("inf")
        closest_gap = min(closest_gap, ngap)

    escape_x = max(-1.0, min(1.0, threat_angle / 90.0))

    nearby_count = sum(
        1 for ast in asteroids
        if wrapped_distance((sx, sy), ast.position, map_w, map_h) < 200
    )

    return {
        "dist": float(max(closest_gap, 0.0)),
        "approach_speed": float(best_closing),
        "ttc": float(best_ttc),
        "heading_err": float(best_heading_err),
        "ammo": getattr(ship_state, "bullets_remaining", getattr(ship_state, "ammo", 0)),
        "mines": getattr(ship_state, "mines_remaining", getattr(ship_state, "mines", 0)),
        "threat_density": float(nearby_count),
        "threat_angle": float(threat_angle),
        "escape_x": float(escape_x),
        "dist_norm": float(dist_norm(max(best_dist, 0.0))),
        "heading_norm": float(heading_norm(best_heading_abs)),
    }
    
    
    
    

#########################################
#build the rules
MF_REGISTRY = {
    "ttc": mu_ttc,
    "dist": mu_dist,
    "threat_angle": mu_threat_angle,
    "approach_speed": mu_approach,
    "threat_density": mu_threat_density,
    "heading_err": mu_heading_err
}

SAFE_FUNCS = {
    "sign": lambda x: -1 if x < 0 else (1 if x > 0 else 0),
    "abs": abs,
    "min": min,
    "max": max,
    "K_TURN": 1.5
}

def load_sugeno_json(name):
    base_dir = os.path.dirname(__file__)
    rules_path = os.path.join(base_dir, name)

    with open(rules_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def build_actor_rules(cfg, params_dict):
    #Builds actor fuzzy rules with learnable consequents
    profiles = cfg["profiles"]
    rules = []

    for rule_idx, rule_spec in enumerate(cfg["rules"]):
        prof_name = rule_spec["use"]
        profile = profiles[prof_name]

        antecedents = []
        for var, mf_name in rule_spec["if"].items():
            if var not in MF_REGISTRY:
                raise ValueError(f"Variable {var} not in MF_REGISTRY")
            mf_funce = MF_REGISTRY[var]
            antecedents.append((var, lambda x, f=mf_funce, m=mf_name:f(x)[m]))

        consequents = {}
        for output_name, spec in profile["out"].items():
            param_key = f"actor_rule{rule_idx}_{output_name}"

            params_dict[param_key] = parse_consequent_params(spec)

            def make_consequent(pk):
                def _consequent(x):
                    params = params_dict[pk]
                    total = params['bias']
                    for var_name, weight in params['weights'].items():
                        if var_name in x:
                            total += weight * x[var_name]
                    return total
                return _consequent
            
            consequents[output_name] = make_consequent(param_key)

        rules.append(SugenoRule(
            antecedents,
            consequents,
            weight=float(profile.get("weight", 1.0))
        ))

    return rules

def build_critic_rules(cfg, params_dict):
    #Builds critic fuzzy rules with learnable consequents
    profiles = cfg["profiles"]
    rules = []

    for rule_idx, rule_spec in enumerate(cfg["rules"]):
        prof_name = rule_spec["use"]
        profile = profiles[prof_name]

        antecedents = []
        for var, mf_name in rule_spec["if"].items():
            if var not in MF_REGISTRY:
                raise ValueError(f"Variable {var} not in MF_REGISTRY")
            mf_funce = MF_REGISTRY[var]
            antecedents.append((var, lambda x, f=mf_funce, m=mf_name:f(x)[m]))

        consequents = {}
        for output_name, spec in profile["out"].items():
            param_key = f"critic_rule{rule_idx}_{output_name}"

            params_dict[param_key] = parse_consequent_params(spec)

            def make_consequent(pk):
                def _consequent(x):
                    params = params_dict[pk]
                    total = params['bias']
                    for var_name, weight in params['weights'].items():
                        if var_name in x:
                            total += weight * x[var_name]
                    return total
                return _consequent
            
            consequents[output_name] = make_consequent(param_key)

        rules.append(SugenoRule(
            antecedents,
            consequents,
            weight=float(profile.get("weight", 1.0))
        ))

    return rules

