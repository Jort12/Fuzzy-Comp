from kesslergame.controller import KesslerController
from util import wrap180, intercept_point, side_score, triag, find_nearest_asteroid, angle_between, distance
from TSK import *
import math
import numpy as np
import json

def build_rules():
    with open("rules.json", "r") as f:
        data = json.load(f)

    rules = data.get("rules",[])
    return rules

class ActorController(KesslerController):
    def __init__(self):
        super().__init__()
        self.system = SugenoSystem(rules=build_rules())
