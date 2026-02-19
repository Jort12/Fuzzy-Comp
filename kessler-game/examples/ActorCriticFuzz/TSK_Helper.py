from .util import *
from .TSK import *
import math
import numpy as np
import os 
import json

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
    reward = 0
    dt = game_state.delta_time
    ctx = context(ship_state, game_state)
  
    reward += 0.01 * dt

    if ctx["dist"] > 300: reward += 0.1 * dt
    
    if ctx["threat_density"] < 3: reward += 0.2 * dt

    if not ship_state.is_respawning:
        for asteroids in game_state.asteroids:
            distance  = np.sqrt((asteroids.position[0]-ship_state["position"][0])**2 + (asteroids.position[1]-ship_state["position"][1])**2)
            if distance <= asteroids.radius + ship_state.radius:
                reward -= 20 * dt
                break

    return reward

def context(ship_state, game_state):
    best_target = None
    best_score = -float("inf")
    best_heading_err = 0.0
    best_heading_abs = 0.0
    best_closing = 0.0
    best_ttc = float("inf")
    best_dist = float("inf")

    closest_dist = float("inf")
    threat_angle = 0.0

    sx, sy = ship_state.position
    svx, svy = ship_state.velocity

    for ast in game_state.asteroids:
        ax, ay = ast.position
        avx, avy = ast.velocity
        dx, dy = ax - sx, ay - sy
        dist = math.hypot(dx, dy)

        # closest asteroid for threat info / dist input
        if dist < closest_dist:
            closest_dist = dist
            angle_to_ast = angle_between(ship_state.position, ast.position)
            threat_angle = wrap180(angle_to_ast - ship_state.heading)

        if dist > 800:
            continue

        ux, uy = (dx / max(dist, 1e-6), dy / max(dist, 1e-6))
        rel_los = -((avx - svx) * ux + (avy - svy) * uy)  # positive when closing

        intercept = intercept_point(ship_state.position, ship_state.velocity,
                                    ast.position, ast.velocity)
        angle_to_intercept = angle_between(ship_state.position, intercept)

        signed_err = wrap180(angle_to_intercept - ship_state.heading)
        abs_err = abs(signed_err)

        cluster_score = 0.0
        for other in game_state.asteroids:
            if other is ast:
                continue
            od = distance(ast.position, other.position)
            if od < 150:
                cluster_score += (150 - od) / 150.0

        score = (1000.0 / max(dist, 1.0)) + cluster_score * 100.0 - abs_err * 2.0

        if score > best_score:
            best_score = score
            best_target = ast
            best_dist = dist
            best_heading_err = signed_err
            best_heading_abs = abs_err
            best_closing = max(0.0, rel_los)
            best_ttc = (dist / max(best_closing, 1e-3)) if best_closing > 0.0 else float("inf")

    if best_target is None:
        nearest = find_nearest_asteroid(ship_state, game_state)
        if nearest is None:
            return None

        dist = distance(ship_state.position, nearest.position)
        intercept = intercept_point(ship_state.position, ship_state.velocity,
                                    nearest.position, nearest.velocity)
        angle_to_intercept = angle_between(ship_state.position, intercept)
        signed_err = wrap180(angle_to_intercept - ship_state.heading)

        best_dist = dist
        best_heading_err = signed_err
        best_heading_abs = abs(signed_err)
        best_closing = 0.0
        best_ttc = float("inf")
        closest_dist = min(closest_dist, dist)  # keep dist input sane

    escape_x = max(-1.0, min(1.0, threat_angle / 90.0))
    nearby_count = sum(
        1 for ast in game_state.asteroids
        if distance(ship_state.position, ast.position) < 200
    )

    return {
        "dist": closest_dist,
        "approach_speed": best_closing,
        "ttc": best_ttc,
        "heading_err": best_heading_err,
        "ammo": ship_state.bullets_remaining,
        "mines": ship_state.mines_remaining,
        "threat_density": nearby_count,
        "threat_angle": threat_angle,
        "escape_x": float(escape_x),
        "dist_norm": float(dist_norm(best_dist)),
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

            params_dict[param_key] = {
                'bias': float(spec.get("bias",0.0)),
                'weights': {k: float(v) for k, v in spec.get("weights", {}).items()}
            }

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

            params_dict[param_key] = {
                'bias': float(spec.get("bias",0.0)),
                'weights': {k: float(v) for k, v in spec.get("weights", {}).items()}
            }

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