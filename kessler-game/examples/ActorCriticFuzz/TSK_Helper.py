from .util import *
import math
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
    return{
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
    return (x["heading_err"] / 180.0)
def dist_norm(x):
    return norm(x["dist"], 0, 1000)

def context(ship_state, game_state):
    ctx = {}
    
    best_target = None
    best_score = -float("inf")
    best_heading_err = 0.0
    best_closing = 0.0
    best_ttc = float("inf")
    
    closest_ast = None
    closest_dist = float("inf")
    threat_angle = 0

    sx, sy = ship_state.position
    svx, svy = ship_state.velocity

    for ast in game_state.asteroids:
        ax, ay = ast.position
        avx, avy = ast.velocity
        dx, dy = ax - sx, ay - sy
        dist = math.hypot(dx, dy)
        
        if dist < closest_dist:
            closest_dist = dist
            closest_ast = ast
            angle_to_ast = angle_between(ship_state.position, ast.position)
            threat_angle = wrap180(angle_to_ast - ship_state.heading)
        
        if dist > 800:
            continue

        ux, uy = (dx / max(dist, 1e-6), dy / max(dist, 1e-6))
        rel_los = -((avx - svx) * ux + (avy - svy) * uy)

        intercept = intercept_point(ship_state.position, ship_state.velocity, ast.position, ast.velocity)
        angle_to_intercept = angle_between(ship_state.position, intercept)
        heading_error = abs(wrap180(angle_to_intercept - ship_state.heading))

        cluster_score = 0.0
        for other in game_state.asteroids:
            if other is ast:
                continue
            od = distance(ast.position, other.position)
            if od < 150:
                cluster_score += (150 - od) / 150.0

        score = (1000.0 / max(dist, 1.0)) + cluster_score * 100.0 - heading_error * 2.0

        if score > best_score:
            best_score = score
            best_target = ast
            best_heading_err = wrap180(angle_to_intercept - ship_state.heading)
            best_closing = max(0.0, rel_los)
            best_ttc = (dist / max(best_closing, 1e-3)) if best_closing > 0.0 else float('inf')

    if best_target is None:
        nearest = find_nearest_asteroid(ship_state, game_state)
        if nearest is None:
            return None
        intercept = intercept_point(ship_state.position, ship_state.velocity, nearest.position, nearest.velocity)
        angle_to_intercept = angle_between(ship_state.position, intercept)
        best_heading_err = wrap180(angle_to_intercept - ship_state.heading)
        dist = distance(ship_state.position, nearest.position)
        best_closing = 0.0
        best_ttc = float('inf')
    
    escape_x = -1 if threat_angle > 0 else 1
    
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
        "escape_x": escape_x,
        "dist_norm": float(dist_norm({"dist": dist})),
        "heading_norm": float(heading_norm({"heading_err": heading_error})),
    }