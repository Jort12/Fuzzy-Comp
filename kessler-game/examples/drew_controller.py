#Author: Kyle Nguyen
#Description: A full fuzzy logic controller for the Kessler game.

from kesslergame.controller import KesslerController
from util import wrap180, intercept_point, side_score, triag, find_nearest_asteroid, angle_between, distance
from fuzzy_system import *
import math
import numpy as np


"""
mu_dist: distance to nearest asteroid
mu_approach: approach speed to nearest asteroid
mu_ttc: time to collision with nearest asteroid
mu_ammo: amount of ammo left
mu_mine
mu_clearance: how clear the area is around the ship
mu_headng_err: angle(target or intercept) - current heading,
    angle difference between ship heading and target heading

Create classes for rules:
SugenoRule: for Sugeno-type rules with numerical outputs
MamdaniRule: for Mamdani-type rules with fuzzy set outputs
mu_thrust: thrust level
mu_turn_rate: turn rate
mu_fire: whether to fire or not
mu_drop_mine: whether to drop a mine or not
mu_evade: whether to evade or not

Create defuzzification methods:
sugeno_defuzzify: for Sugeno-type rules

mamdani_defuzzify: for Mamdani-type rules



"""
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

def build_rules():
    K_TURN = 1.5
    K_THRUST = 0.8
    
    def heading_norm(x):
        return (x["heading_err"] / 180.0)

    def dist_norm(x):
        return norm(x["dist"], 0, 1000)

    def approach_norm(x):
        return norm(x["approach_speed"], -400, 400) * 2 - 1

    rules = [
        # CRITICAL EVASION - THREAT ON RIGHT, TURN LEFT
        SugenoRule(
            antecedents=[
                ("ttc", lambda t: mu_ttc(t)["imminent"]),
                ("threat_angle", lambda a: mu_threat_angle(a)["right_side"]),
                ("dist", lambda d: mu_dist(d)["very_close"])
            ],
            consequents={
                "thrust": lambda x: 30,
                "turn_rate": lambda x: -180
            },
            weight=2.0
        ),
        
        # CRITICAL EVASION - THREAT ON LEFT, TURN RIGHT
        SugenoRule(
            antecedents=[
                ("ttc", lambda t: mu_ttc(t)["imminent"]),
                ("threat_angle", lambda a: mu_threat_angle(a)["left_side"]),
                ("dist", lambda d: mu_dist(d)["very_close"])
            ],
            consequents={
                "thrust": lambda x: 30,
                "turn_rate": lambda x: 180
            },
            weight=2.0
        ),
        
        # CRITICAL EVASION - THREAT AHEAD
        SugenoRule(
            antecedents=[
                ("ttc", lambda t: mu_ttc(t)["imminent"]),
                ("threat_angle", lambda a: mu_threat_angle(a)["ahead"]),
                ("dist", lambda d: mu_dist(d)["very_close"])
            ],
            consequents={
                "thrust": lambda x: -50,  # Reverse
                "turn_rate": lambda x: 180 if x.get("threat_angle", 0) > 0 else -180
            },
            weight=2.0
        ),
        
        # URGENT EVASION - VERY FAST APPROACHING
        SugenoRule(
            antecedents=[
                ("approach_speed", lambda v: mu_approach(v)["very_fast_approaching"]),
                ("dist", lambda d: mu_dist(d)["close"])
            ],
            consequents={
                "thrust": lambda x: 40,
                "turn_rate": lambda x: 180 if x.get("threat_angle", 0) > 0 else -180
            },
            weight=1.8
        ),
        
        # SOON COLLISION - THREAT RIGHT
        SugenoRule(
            antecedents=[
                ("ttc", lambda t: mu_ttc(t)["soon"]),
                ("threat_angle", lambda a: mu_threat_angle(a)["right_side"])
            ],
            consequents={
                "thrust": lambda x: 60,
                "turn_rate": lambda x: -150
            },
            weight=1.5
        ),
        
        # SOON COLLISION - THREAT LEFT
        SugenoRule(
            antecedents=[
                ("ttc", lambda t: mu_ttc(t)["soon"]),
                ("threat_angle", lambda a: mu_threat_angle(a)["left_side"])
            ],
            consequents={
                "thrust": lambda x: 60,
                "turn_rate": lambda x: 150
            },
            weight=1.5
        ),
        
        # DENSE AREA - ESCAPE
        SugenoRule(
            antecedents=[
                ("threat_density", lambda d: mu_threat_density(d)["dense"]),
                ("dist", lambda d: mu_dist(d)["close"])
            ],
            consequents={
                "thrust": lambda x: 80,
                "turn_rate": lambda x: 180 if x.get("threat_angle", 0) > 0 else -180
            },
            weight=1.3
        ),
        
        # MODERATE DENSITY - AVOID
        SugenoRule(
            antecedents=[
                ("threat_density", lambda d: mu_threat_density(d)["moderate"])
            ],
            consequents={
                "thrust": lambda x: 100,
                "turn_rate": lambda x: 120 * np.sign(x.get("escape_vector", 1))
            },
            weight=1.0
        ),
        
        # FAST APPROACHING - SLOW DOWN
        SugenoRule(
            antecedents=[
                ("approach_speed", lambda v: mu_approach(v)["fast_approaching"]),
                ("dist", lambda d: mu_dist(d)["close"])
            ],
            consequents={
                "thrust": lambda x: 20,
                "turn_rate": lambda x: 150 * np.sign(x.get("escape_vector", 1))
            },
            weight=1.2
        ),

        # SAFE TO AIM - LEFT
        SugenoRule(
            antecedents=[
                ("heading_err", lambda e: mu_heading_err(e)["left"]),
                ("dist", lambda d: mu_dist(d)["medium"]),
                ("ttc", lambda t: mu_ttc(t)["later"])
            ],
            consequents={
                "thrust": lambda x: 80 + 40 * dist_norm(x),
                "turn_rate": lambda x: K_TURN * 90 * heading_norm(x)
            },
            weight=0.5
        ),
        
        # SAFE TO AIM - RIGHT
        SugenoRule(
            antecedents=[
                ("heading_err", lambda e: mu_heading_err(e)["right"]),
                ("dist", lambda d: mu_dist(d)["medium"]),
                ("ttc", lambda t: mu_ttc(t)["later"])
            ],
            consequents={
                "thrust": lambda x: 80 + 40 * dist_norm(x),
                "turn_rate": lambda x: -K_TURN * 90 * abs(heading_norm(x))
            },
            weight=0.5
        ),

        # SAFE TO GO STRAIGHT
        SugenoRule(
            antecedents=[
                ("heading_err", lambda e: mu_heading_err(e)["straight"]),
                ("dist", lambda d: mu_dist(d)["far"]),
                ("ttc", lambda t: mu_ttc(t)["far_future"])
            ],
            consequents={
                "thrust": lambda x: 120 + 60 * dist_norm(x),
                "turn_rate": 0.0
            },
            weight=0.5
        ),

        # FAR TARGETS
        SugenoRule(
            antecedents=[
                ("dist", lambda d: mu_dist(d)["far"]),
                ("ttc", lambda t: mu_ttc(t)["far_future"])
            ],
            consequents={
                "thrust": lambda x: 100 + 40 * dist_norm(x),
                "turn_rate": 0.0
            },
            weight=0.3
        ),
        
        # DEFAULT
        SugenoRule(
            antecedents=[("dist", lambda d: mu_dist(d)["medium"])],
            consequents={
                "thrust": lambda x: 60 + 20 * dist_norm(x),
                "turn_rate": 0.0
            },
            weight=0.2
        ),
    ]

    return rules


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
    
    escape_vector = -1 if threat_angle > 0 else 1
    
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
        "escape_vector": escape_vector
    }


class KyleController(KesslerController):
    name = "Kyle's Fuzzy Controller"
    def __init__(self):
        self.debug_counter = 0
        self.system = SugenoSystem(rules=build_rules())
    
        
    def actions(self, ship_state, game_state): 
        ctx = context(ship_state, game_state)
        
        outputs = self.system.evaluate(ctx)
        thrust = outputs.get("thrust", 0.0)
        turn_rate = outputs.get("turn_rate", 0.0)
        
        fire = (
            ship_state.can_fire and
            ctx["ammo"] != 0 and
            abs(ctx["heading_err"]) < 6 and
            ctx["ttc"] > 3 and
            ctx["dist"] > 150
        )
        drop_mine = False
        
        thrust = max(ship_state.thrust_range[0], min(ship_state.thrust_range[1], thrust))
        turn_rate = max(ship_state.turn_rate_range[0], min(ship_state.turn_rate_range[1], turn_rate))
    
        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)