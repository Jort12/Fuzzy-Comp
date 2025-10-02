#Author: Kyle Nguyen
#Description: A full fuzzy logic controller for the Kessler game.

from kesslergame import Controller
from util import wrap180, intercept_point, side_score, triag
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

mu_heading_err: angle(target or intercept) − current heading,
    angle difference between ship heading and target heading


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
    
