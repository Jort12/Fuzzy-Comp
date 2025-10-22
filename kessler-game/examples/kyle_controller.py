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
        "very_fast_approaching":  triag(v,  150, 200, 300),
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
    
def mu_ammo(a): #ammo left
    return {
        "none": triag(a, 0, 0, 1),
        "very_low": triag(a, 0, 1, 2),
        "low": triag(a, 1, 2, 4),
        "medium": triag(a, 3, 5, 7),
        "high": triag(a, 6, 8, 10),
        "full": triag(a, 9, 10, 10)
    }
    
def mu_mine(m): #mines left
    return {
        "none": triag(m, 0, 0, 1),
        "low": triag(m, 0, 1, 2),
        "medium": triag(m, 1, 2, 3),
        "high": triag(m, 2, 3, 4),

    }
    
    
def mu_threat_density(density):#how many asteroids are nearby
    return {
        "clear": triag(density, 0, 0, 1),
        "low": triag(density, 0, 1, 3),
        "moderate": triag(density, 2, 4, 6),
        "dense": triag(density, 5, 7, 10)
    }
    
def mu_escape_angle(angle):#angle between ship heading and best escape direction
    return {
        "blocked": triag(angle, 0, 15, 30),
        "tight": triag(angle, 25, 45, 60),
        "open": triag(angle, 55, 90, 180)
    }

def norm(x, lo, hi): # normalize x to [0, 1] between lo and hi
    if hi == lo:
        return 0.0
    x = max(lo, min(hi, x))
    return (x - lo) / (hi - lo)

def build_rules():
    K_TURN = 1.2   # how aggressively to steer (try 0.8–1.5)
    K_THRUST = 0.4 # base thrust gain
    SAFE_DIST = 150.0

    def heading_norm(x):
        return (x["heading_err"] / 180.0)  # -1..1 range

    def dist_norm(x):
        return norm(x["dist"], 0, 1000)

    def approach_norm(x):
        return norm(x["approach_speed"], -400, 400) * 2 - 1 #

    rules = [

        #EMERGENCY AVOIDANCE
        SugenoRule(
            antecedents=[
                ("ttc", lambda t: mu_ttc(t)["imminent"]),
                ("heading_err", lambda e: mu_heading_err(e)["left"])
            ],
            consequents={
                "thrust": lambda x: -100 + 50 * dist_norm(x),
                "turn_rate": lambda x: +180 * (1 - abs(heading_norm(x)))
            },
            weight=1.0
        ),
        
        SugenoRule( 
            antecedents=[
                ("ttc", lambda t: mu_ttc(t)["imminent"]),
                ("heading_err", lambda e: mu_heading_err(e)["right"])
            ],
            consequents={
                "thrust": lambda x: -100 + 50 * dist_norm(x),
                "turn_rate": lambda x: -180 * (1 - abs(heading_norm(x)))
            },
            weight=1.0
        ),

        #HEADING ALIGNMENT — continuous steering toward target
        SugenoRule(
            antecedents=[("heading_err", lambda e: mu_heading_err(e)["left"])],
            consequents={
                "thrust": lambda x: 40 + 40 * dist_norm(x),
                "turn_rate": lambda x: +K_TURN * 180 * heading_norm(x)
            },
            weight=0.9
        ),
        SugenoRule(
            antecedents=[("heading_err", lambda e: mu_heading_err(e)["right"])],
            consequents={
                "thrust": lambda x: 40 + 40 * dist_norm(x),
                "turn_rate": lambda x: -K_TURN * 180 * abs(heading_norm(x))
            },
            weight=0.8
        ),

        #STRAIGHT ALIGNMENT — reduce turning, maintain steady thrust
        SugenoRule(
            antecedents=[("heading_err", lambda e: mu_heading_err(e)["straight"])],
            consequents={
                "thrust": lambda x: 60 + 40 * dist_norm(x),
                "turn_rate": 0.0
            },
            weight=1.0
        ),

        #DISTANCE-BASED SPEED — go faster when far
        SugenoRule(
            antecedents=[("dist", lambda d: mu_dist(d)["far"])],
            consequents={
                "thrust": lambda x: 60 + 60 * dist_norm(x),
                "turn_rate": 0.0
            },
            weight=0.5
        ),
        SugenoRule(
            antecedents=[("dist", lambda d: mu_dist(d)["medium"])],
            consequents={
                "thrust": lambda x: 40 + 40 * dist_norm(x),
                "turn_rate": 0.0
            },
            weight=0.5
        ),

        #APPROACH SPEED CONTROL — slow down smoothly
        SugenoRule(
            antecedents=[("approach_speed", lambda v: mu_approach(v)["fast_approaching"])],
            consequents={
                "thrust": lambda x: 40 - 40 * approach_norm(x),
                "turn_rate": 0.0
            },
            weight=0.6
        ),

        #DEFAULT CRUISE — gentle forward motion
        SugenoRule(
            antecedents=[("dist", lambda d: mu_dist(d)["medium"])],
            consequents={
                "thrust": lambda x: 20 + 20 * dist_norm(x),
                "turn_rate": 0.0
            },
            weight=0.4
        ),
    ]

    return rules



def fire_decision(ctx, ship_state):
#TODOL: Implement Fuzzy Logic for firing decision
    if ctx["ammo"] > 0 and abs(ctx["heading_err"]) < 5 and ctx["ttc"] < 20.0:
        return True
    return False



def context(ship_state, game_state):
    ctx = {}
    
    best_target = None
    best_score = -float("inf")
    best_heading_err = 0.0
    best_closing = 0.0
    best_ttc = float("inf")

    sx, sy = ship_state.position
    svx, svy = ship_state.velocity

    for ast in game_state.asteroids:
        ax, ay = ast.position
        avx, avy = ast.velocity
        dx, dy = ax - sx, ay - sy
        dist = math.hypot(dx, dy)
        if dist > 800:  # skip far targets
            continue

        # line-of-sight unit vector
        ux, uy = (dx / max(dist, 1e-6), dy / max(dist, 1e-6))
        # approach speed along line of sight (positive if closing)
        rel_los = (avx - svx) * ux + (avy - svy) * uy

        # intercept and heading error
        intercept = intercept_point(ship_state.position, ship_state.velocity, ast.position, ast.velocity)
        angle_to_intercept = angle_between(ship_state.position, intercept)
        heading_error = abs(wrap180(angle_to_intercept - ship_state.heading))

        # cluster bonus
        cluster_score = 0.0
        for other in game_state.asteroids:
            if other is ast:
                continue
            od = distance(ast.position, other.position)
            if od < 150:
                cluster_score += (150 - od) / 150.0

        # simple score: prefer near, clustered, small heading_err
        score = (1000.0 / max(dist, 1.0)) + cluster_score * 100.0 - heading_error * 2.0

        if score > best_score:
            best_score = score
            best_target = ast
            best_heading_err = wrap180(angle_to_intercept - ship_state.heading)
            best_closing = max(0.0, rel_los)  # only “closing” part
            best_ttc = (dist / max(best_closing, 1e-3)) if best_closing > 0.0 else float('inf')

    # fallback if nothing qualified
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
    
    # Momentum analysis
    if best_target:
        target_vec = np.array(best_target.position) - np.array(ship_state.position)
        ship_vel_normalized = np.array(ship_state.velocity) / max(np.linalg.norm(ship_state.velocity), 1)
        target_normalized = target_vec / max(np.linalg.norm(target_vec), 1)
        momentum_alignment = np.dot(ship_vel_normalized, target_normalized)
    else:
        momentum_alignment = 0
    
    return {
        "dist": distance(ship_state.position, best_target.position) if best_target else dist,
        "approach_speed": best_closing,            # consistent and bounded
        "ttc": best_ttc,                           # now available for firing logic
        "heading_err": best_heading_err,
        "ammo": ship_state.bullets_remaining,
        "mines": ship_state.mines_remaining,
    }


class KyleController(KesslerController):
    name = "Kyle's Fuzzy Controller"
    def __init__(self):
        self.debug_counter = 0  # just to not spam too much
        self.system = SugenoSystem(rules=build_rules())
    
        
    def actions(self, ship_state, game_state): 
        ctx = context(ship_state, game_state)
        
        # Get thrust and turn rate from fuzzy system
        outputs = self.system.evaluate(ctx)
        thrust = outputs.get("thrust", 0.0)
        turn_rate = outputs.get("turn_rate", 0.0)
        
        # Decide when to fire: aim well + have ammo + target is close enough
        fire = (
            ship_state.can_fire and              # Ship is ready to fire
            ctx["ammo"] != 0 and                 # Have bullets remaining
            abs(ctx["heading_err"]) < 6 and      # Aimed close enough to target
            ctx["ttc"] < 2.5                     # Target is close in time
        )
        drop_mine = False  # Not implemented yet - placeholder for future mine logic
        
        #clamp to valid ranges
        thrust = max(ship_state.thrust_range[0], min(ship_state.thrust_range[1], thrust))
        turn_rate = max(ship_state.turn_rate_range[0], min(ship_state.turn_rate_range[1], turn_rate))
    
        
        
        
        
        
        
        
        
        
        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)

