from kesslergame.controller import KesslerController
from util import *
from TSK import *
import math
import numpy as np
import json

def load_sugeno_json():
    with open("rules.json", "r") as f:
        data = json.load(f)

    rules = [build_rules(r) for r in data["rules"]]
    return rules

def build_antecedents(rules_ants):
    antecedents = []

    for a in rules_ants:
        var = a["var"]
        mf = a["mf"]

        if var not in MF_REGISTRY:
            raise KeyError(
                f"Antecedent var '{var}' not found in MF_REGISTRY. "
            )

        func = MF_REGISTRY[var]

        
    return antecedents

def build_consequents(spec):
    consequents = []

    return consequents

def build_rules(rules):
    consequents = {
        name: build_consequents(spec)
        for name, spec in rules["consequents"].items()
    }

    rulesList = SugenoRule(
        antecedents=build_antecedents(rules["antecdents"]),
        consequents=consequents,
        weight=rules.get("weight", 1.0)
    )
    return rulesList

MF_REGISTRY = {
    "ttc": mu_ttc,
    "dist": mu_dist,
    "threat_angle": mu_threat_angle,
    "approach_speed": mu_approach,
    "threat_density": mu_threat_density,
    "heading_err": mu_heading_err
}

class ActorController(KesslerController):
    def __init__(self):
        super().__init__()
        self.system = SugenoSystem(rules=build_rules())

        
