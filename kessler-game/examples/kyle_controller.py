 #Author: Kyle Nguyen
#Description: A full fuzzy logic controller for the Kessler game.

from kesslergame.controller import KesslerController
from util import wrap180, intercept_point, side_score, triag, find_nearest_asteroid, angle_between, distance
from fuzzy_system import SugenoRule

import math

"""
PLAN:

Create fuzzy sets for inputs:
mu_dist: distance to nearest asteroid
mu_approach: approach speed to nearest asteroid
mu_ttc: time to collision with nearest asteroid
mu_ammo: amount of ammo left
mu_mine
mu_clearance: how clear the area is around the ship

mu_heading_err: angle(target or intercept) - current heading,
    angle difference between ship heading and target heading

Create classes for rules:
SugenoRule: for Sugeno-type rules with numerical outputs
MamdaniRule: for Mamdani-type rules with fuzzy set outputs
Create fuzzy sets for outputs:
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
        "away": triag(v, -5, -2, 0),
        "stable": triag(v, -1, 0, 1),
        "approaching": triag(v, 0, 2, 5),
        "fast_approaching": triag(v, 3, 6, 10),
        "very_fast_approaching": triag(v, 8, 12, 20)
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
        "sharp_left": triag(e, -180, -90, -30),
        "left": triag(e, -60, -30, 0),
        "slight_left": triag(e, -20, -5, 0),
        "straight": triag(e, -5, 0, 5),
        "slight_right": triag(e, 0, 5, 20),
        "right": triag(e, 0, 30, 60),
        "sharp_right": triag(e, 30, 90, 180)
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

def build_rules():
    return [
        #close 


    ]

def context(ship_state, game_state):
    # find nearest asteroid
    asteroids = game_state.asteroids
    if not asteroids:
        return None  # no asteroids, nothing to do
    nearest_asteroid = find_nearest_asteroid(ship_state, game_state)
    dist = distance(ship_state.position, nearest_asteroid.position)
    
    rel_vel = (nearest_asteroid.velocity[0] - ship_state.velocity[0], nearest_asteroid.velocity[1] - ship_state.velocity[1])
    approach_speed = (rel_vel[0] * (nearest_asteroid.position[0] - ship_state.position[0]) + rel_vel[1] * (nearest_asteroid.position[1] - ship_state.position[1])) / dist if dist != 0 else 0

    ttc = dist / approach_speed if approach_speed > 0 else float('inf')

    intercept = intercept_point(ship_state.position, ship_state.velocity, nearest_asteroid.position, nearest_asteroid.velocity)
    target_angle = angle_between(ship_state.position, intercept)
    heading_err = wrap180(target_angle - ship_state.angle)


    return {
        "dist": dist,
        "approach_speed": approach_speed,
        "ttc": ttc,
        "heading_err": heading_err,
        "ammo": ship_state.bullets_remaining,
        "mines": ship_state.mines_remaining
    }


class KyleController(KesslerController):
    name = "Kyle's Fuzzy Controller"
    def __init__(self):
        self.debug_counter = 0  # just to not spam too much
        

        
    def actions(self, ship_state, game_state): 
        
        
        
        
        
        
        
        
        
        
        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)

